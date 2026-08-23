/* Qwen3-TTS native path (row 31). Drives the vendored libmtmd audio
 * generation helpers against the same shared ggml as whisper/llama.
 *
 * Flow mirrors upstream tools/tts/tts.cpp at the pinned vendor commit:
 *   set_input -> step_prompt until 0 -> [sample semantic code -> step_gen]*
 *   -> get_output. Model-specific logic lives inside mtmd-helper-gen; this
 *   shim stays generic on purpose so future vendor bumps stay mechanical.
 */

#include "syllabix_native.h"

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Upstream tools/tts sampling defaults. Seed comes from the caller so the
 * native round-trip test can pin reproducible fixtures. */
#define SYLLABIX_QWEN_TOP_K 40
#define SYLLABIX_QWEN_TOP_P 0.95f
#define SYLLABIX_QWEN_TEMP 0.8f
/* tts.cpp caps generation at 512 frames (~42 s of audio at 12 Hz/frame). */
#define SYLLABIX_QWEN_MAX_FRAMES 512
#define SYLLABIX_QWEN_N_BATCH 512

/* Row 32 voice anchor. The Base backbones are speaker-unconditioned: every
 * cold-start generation samples a new speaker, so per-sentence generation
 * changed voices mid-reply. Fix: at load, synthesize one short clip with
 * this fixed seed, run it through the mmproj speaker encoder, and prepend
 * the resulting x-vector to every later prompt (mtmd-helper gen_audio
 * already supports `speaker_ref` and keeps the voice primed across chunk
 * rewinds). Fixed seed => same anchor => the same Syllabix voice on every
 * machine, every run, both backbone sizes. */
#define SYLLABIX_QWEN_VOICE_SEED 20260822u
static const char SYLLABIX_QWEN_ANCHOR_TEXT[] =
    "Hello, this is your local assistant. I keep one steady voice.";

struct syllabix_qwen_tts {
    struct llama_model *model;
    struct llama_context *ctx;
    mtmd_context *mctx;
    struct llama_sampler *smpl;
    mtmd_helper_gen_audio *gen;
    mtmd_bitmap *voice; /* self-generated speaker reference; NULL = fallback */
};

static int qwen_generate(
    struct syllabix_qwen_tts *tts,
    const char *text,
    const char *lang,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples,
                  int is_last, void *user),
    void *pcm_user,
    int32_t *out_sample_rate,
    int16_t **out_pcm,
    int64_t *out_n_samples);

/* Upstream tools/tts chain (top-k -> top-p -> temp -> dist(seed)). Built per
 * use so the voice anchor can pin its own seed without touching the runtime
 * sampler state. */
static struct llama_sampler *qwen_make_sampler(unsigned int seed) {
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (smpl == NULL) {
        return NULL;
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(SYLLABIX_QWEN_TOP_K));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(SYLLABIX_QWEN_TOP_P, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(SYLLABIX_QWEN_TEMP));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    return smpl;
}

static void qwen_silent_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static int qwen_aborted(bool (*abort_cb)(void *user), void *abort_user) {
    return abort_cb != NULL && abort_cb(abort_user);
}

static int qwen_debug(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *flag = getenv("SYLLABIX_QWEN_DEBUG");
        enabled = flag != NULL && flag[0] != '\0' && flag[0] != '0';
    }
    return enabled;
}

static void qwen_debug_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)user_data;
    if (text != NULL) {
        fputs(text, stderr);
    }
}

#define QWEN_LOG(...)                                                     \
    do {                                                                  \
        if (qwen_debug()) {                                               \
            fprintf(stderr, "syllabix_qwen_tts: " __VA_ARGS__);           \
        }                                                                 \
    } while (0)

struct syllabix_qwen_tts *syllabix_qwen_tts_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed) {
    if (model_path == NULL || mmproj_path == NULL || n_threads < 1) {
        return NULL;
    }

    /* Diagnostics: when SYLLABIX_QWEN_DEBUG is set, mtmd/clip helper errors
     * stream to stderr instead of vanishing into the silent logger. */
    mtmd_helper_log_set(qwen_debug() ? qwen_debug_log : qwen_silent_log, NULL);
    llama_backend_init();

    /* Row 31 decision: the Qwen3-TTS stack stays on CPU everywhere. Its
     * gen_code graph wants a ~870 MiB Metal compute buffer per frame batch,
     * which ggml_backend_sched fails to place once the STT and LLM contexts
     * are also resident (the normal Syllabix configuration). CPU keeps the
     * provider correct alongside the rest of the pipeline; revisit Metal
     * when upstream splits the audio graph smaller. */
    const int tts_gpu_layers = 0;

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = tts_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        return NULL;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    /* Embeddings must be on: the audio helper consumes the backbone hidden
     * state (llama_get_embeddings_ith) as its cross-frame state. Context is
     * the GGUF trained window, same rule as the chat LLM. */
    ctx_params.embeddings = true;
    ctx_params.n_ctx = 0;
    ctx_params.n_batch = SYLLABIX_QWEN_N_BATCH;
    ctx_params.n_ubatch = SYLLABIX_QWEN_N_BATCH;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.offload_kqv = tts_gpu_layers != 0;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        llama_model_free(model);
        return NULL;
    }

    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = tts_gpu_layers != 0 && syllabix_whisper_use_gpu() != 0;
    mtmd_params.print_timings = false;
    mtmd_context *mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (mctx == NULL) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    if (mtmd_gen_audio_get_info(mctx).type != MTMD_GEN_AUDIO_TYPE_QWEN3TTS) {
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    mtmd_helper_gen_audio *gen = mtmd_helper_gen_audio_init(ctx, mctx);
    if (gen == NULL) {
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    struct syllabix_qwen_tts *tts = (struct syllabix_qwen_tts *)malloc(sizeof(struct syllabix_qwen_tts));
    if (tts == NULL) {
        mtmd_helper_gen_audio_free(gen);
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    tts->model = model;
    tts->ctx = ctx;
    tts->mctx = mctx;
    tts->smpl = qwen_make_sampler(seed);
    tts->gen = gen;
    tts->voice = NULL;
    if (tts->smpl == NULL) {
        syllabix_qwen_tts_free(tts);
        return NULL;
    }
    QWEN_LOG("loaded; n_ctx=%d n_embd=%d\n", (int)llama_n_ctx(ctx), (int)llama_model_n_embd(model));

    /* Row 32 voice anchor: one unconditioned clip generated on a chain
     * pinned to SYLLABIX_QWEN_VOICE_SEED becomes the speaker reference
     * every later sentence is conditioned on, so the voice does not drift
     * with the runtime sampler seed. Any failure degrades to the row-31
     * free-sampling behavior; it never fails the load. */
    {
        int32_t rate = 0;
        int16_t *pcm = NULL;
        int64_t n_samples = 0;
        struct llama_sampler *anchor_smpl = qwen_make_sampler(SYLLABIX_QWEN_VOICE_SEED);
        if (anchor_smpl != NULL) {
            struct llama_sampler *runtime_smpl = tts->smpl;
            tts->smpl = anchor_smpl;
            const int rc = qwen_generate(tts, SYLLABIX_QWEN_ANCHOR_TEXT, "en", NULL, NULL,
                                         NULL, NULL, &rate, &pcm, &n_samples);
            tts->smpl = runtime_smpl;
            llama_sampler_free(anchor_smpl);
            if (rc == 0 && pcm != NULL && n_samples > 0 && rate > 0) {
                float *f32 = (float *)malloc((size_t)n_samples * sizeof(float));
                if (f32 != NULL) {
                    for (int64_t i = 0; i < n_samples; i++) {
                        f32[i] = (float)pcm[i] / 32767.0f;
                    }
                    /* mtmd_bitmap_init_from_audio copies; the scratch buffer
                     * is ours to free. */
                    tts->voice = mtmd_bitmap_init_from_audio((size_t)n_samples, f32);
                    free(f32);
                }
            }
        }
        if (pcm != NULL) {
            syllabix_qwen_tts_pcm_free(pcm);
        }
        QWEN_LOG("voice anchor %s\n",
                 tts->voice != NULL ? "engaged" : "unavailable (unconditioned fallback)");
    }
    return tts;
}

int syllabix_qwen_tts_has_voice(const struct syllabix_qwen_tts *tts) {
    return tts != NULL && tts->voice != NULL ? 1 : 0;
}

void syllabix_qwen_tts_free(struct syllabix_qwen_tts *tts) {
    if (tts == NULL) {
        return;
    }
    if (tts->gen != NULL) {
        mtmd_helper_gen_audio_free(tts->gen);
    }
    if (tts->smpl != NULL) {
        llama_sampler_free(tts->smpl);
    }
    if (tts->voice != NULL) {
        mtmd_bitmap_free(tts->voice);
    }
    if (tts->mctx != NULL) {
        mtmd_free(tts->mctx);
    }
    if (tts->ctx != NULL) {
        llama_free(tts->ctx);
    }
    if (tts->model != NULL) {
        llama_model_free(tts->model);
    }
    free(tts);
}

/* 0 = ok, 1 = cancelled, -1 = error. One independent generation; the
 * speaker reference comes from tts->voice (NULL during anchor generation
 * itself). `out_pcm` receives malloc'd mono i16 (clipped exactly like the
 * upstream WAV writer: clamp to [-1, 1], scale by 32767). Free with
 * syllabix_qwen_tts_pcm_free. `lang` may be NULL or empty for `en`. */
static int qwen_generate(
    struct syllabix_qwen_tts *tts,
    const char *text,
    const char *lang,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples,
                  int is_last, void *user),
    void *pcm_user,
    int32_t *out_sample_rate,
    int16_t **out_pcm,
    int64_t *out_n_samples) {
    if (tts == NULL || tts->ctx == NULL || text == NULL
        || (pcm_cb == NULL && (out_pcm == NULL || out_sample_rate == NULL || out_n_samples == NULL))) {
        return -1;
    }
    if (qwen_aborted(abort_cb, abort_user)) {
        return 1;
    }

    /* Each spoken sentence is an independent generation on this context. */
    llama_memory_clear(llama_get_memory(tts->ctx), true);
    mtmd_helper_gen_audio_reset(tts->gen);

    struct mtmd_helper_gen_audio_inp inp;
    memset(&inp, 0, sizeof(inp));
    inp.seq_id = 0;
    inp.prompt = text;
    inp.prompt_len = strlen(text);
    inp.speaker_ref = tts->voice;
    inp.lang = (lang != NULL && lang[0] != '\0') ? lang : "en";
    inp.top_k = SYLLABIX_QWEN_TOP_K;
    inp.top_p = SYLLABIX_QWEN_TOP_P;
    inp.seed = 0; /* unused by set_input; sampling seed lives in the chain */
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;

    if (mtmd_helper_gen_audio_set_input(tts->gen, &inp) != 0) {
        QWEN_LOG("set_input failed\n");
        return -1;
    }

    for (;;) {
        if (qwen_aborted(abort_cb, abort_user)) {
            return 1;
        }
        const int32_t remaining = mtmd_helper_gen_audio_step_prompt(tts->gen, SYLLABIX_QWEN_N_BATCH);
        if (remaining < 0) {
            QWEN_LOG("step_prompt failed (remaining=%d)\n", remaining);
            return -2;
        }
        if (remaining == 0) {
            break;
        }
    }

    const float *h_state = llama_get_embeddings_ith(tts->ctx, -1);
    if (h_state == NULL) {
        QWEN_LOG("embeddings are NULL after prompt (embedding flag off?)\n");
        return -6;
    }
    llama_token sampled = llama_sampler_sample(tts->smpl, tts->ctx, -1);

    bool stop = false;
    int n_frames = 0;
    while (!stop && n_frames < SYLLABIX_QWEN_MAX_FRAMES) {
        if (qwen_aborted(abort_cb, abort_user)) {
            return 1;
        }
        const float *h_next = NULL;
        if (mtmd_helper_gen_audio_step_gen(tts->gen, sampled, h_state, &h_next, &stop) != 0) {
            QWEN_LOG("step_gen failed at frame %d\n", n_frames);
            return -3;
        }
        if (pcm_cb != NULL) {
            int32_t partial_rate = 0;
            const float *partial_pcm = NULL;
            int64_t partial_samples = 0;
            if (mtmd_helper_gen_audio_take_output(tts->gen, &partial_rate, &partial_pcm,
                                                  &partial_samples) != 0) {
                return -4;
            }
            if (partial_samples > 0 && pcm_cb(partial_rate, partial_pcm, partial_samples,
                                               0, pcm_user) != 0) {
                return 1;
            }
        }
        if (h_next == NULL) {
            break; /* stopped without producing a frame */
        }
        h_state = h_next;
        n_frames++;
        if (stop) {
            break;
        }
        sampled = llama_sampler_sample(tts->smpl, tts->ctx, -1);
    }

    int32_t rate = 0;
    const char *data = NULL;
    size_t data_len = 0;
    int64_t n_samples = 0;
    if (mtmd_helper_gen_audio_get_output(tts->gen, &rate, &data, &data_len, &n_samples) != 0) {
        QWEN_LOG("get_output failed after %d frames\n", n_frames);
        return -4;
    }
    if (pcm_cb != NULL) {
        const float *partial_pcm = NULL;
        int64_t partial_samples = 0;
        if (mtmd_helper_gen_audio_take_output(tts->gen, &rate, &partial_pcm,
                                              &partial_samples) != 0) {
            return -4;
        }
        return pcm_cb(rate, partial_pcm, partial_samples, 1, pcm_user) == 0 ? 0 : 1;
    }
    if (rate <= 0 || data == NULL || n_samples <= 0 || data_len < (size_t)n_samples * sizeof(float)) {
        QWEN_LOG("bad output rate=%d samples=%lld len=%zu\n", rate, (long long)n_samples, data_len);
        return -5;
    }

    int16_t *pcm = (int16_t *)malloc((size_t)n_samples * sizeof(int16_t));
    if (pcm == NULL) {
        return -1;
    }
    const float *src = (const float *)data;
    for (int64_t i = 0; i < n_samples; i++) {
        float v = src[i];
        if (v > 1.0f) {
            v = 1.0f;
        } else if (v < -1.0f) {
            v = -1.0f;
        }
        pcm[i] = (int16_t)(v * 32767.0f);
    }

    *out_sample_rate = rate;
    *out_pcm = pcm;
    *out_n_samples = n_samples;
    return 0;
}

/* Public entry: same contract, with the pinned voice reference engaged. */
int syllabix_qwen_tts_synthesize(
    struct syllabix_qwen_tts *tts,
    const char *text,
    const char *lang,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int32_t *out_sample_rate,
    int16_t **out_pcm,
    int64_t *out_n_samples) {
    return qwen_generate(tts, text, lang, abort_cb, abort_user, NULL, NULL,
                         out_sample_rate, out_pcm, out_n_samples);
}

int syllabix_qwen_tts_synthesize_streaming(
    struct syllabix_qwen_tts *tts, const char *text, const char *lang,
    bool (*abort_cb)(void *user), void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples,
                  int is_last, void *user), void *pcm_user) {
    if (pcm_cb == NULL) {
        return -1;
    }
    return qwen_generate(tts, text, lang, abort_cb, abort_user, pcm_cb, pcm_user,
                         NULL, NULL, NULL);
}

void syllabix_qwen_tts_pcm_free(int16_t *pcm) {
    free(pcm);
}
