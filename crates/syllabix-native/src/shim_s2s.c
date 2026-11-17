/* LFM3-Audio turn-based S2S. Feeds one VAD-committed utterance through
 * mtmd_audio_stream_* then mtmd_helper_gen_audio_* on the shared ggml.
 *
 * Mirrors shim_tts.c: abort callbacks between steps and streaming native-rate
 * f32 PCM. CPU-only in this stack (n_gpu_layers = 0).
 */

#include "syllabix_native.h"

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ~42 s at 24 kHz, same order of magnitude as Qwen's 512-frame cap. */
#define SYLLABIX_S2S_DEFAULT_MAX_OUT_SAMPLES 1024000
#define SYLLABIX_S2S_N_BATCH 512
/* Empty prompt tokenizes to zero ids; continuing needs audio-start as the last piece. */
#define SYLLABIX_S2S_AUDIO_START_PROMPT "<|audio_start|>"
#define SYLLABIX_S2S_AUDIO_START_PROMPT_LEN 15

struct syllabix_s2s {
    struct llama_model *model;
    struct llama_context *ctx;
    mtmd_context *mctx;
    struct llama_sampler *smpl;
    mtmd_helper_gen_audio *gen;
    int in_rate;
    int out_rate;
    unsigned int seed;
};

static void s2s_silent_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static int s2s_aborted(bool (*abort_cb)(void *user), void *abort_user) {
    return abort_cb != NULL && abort_cb(abort_user);
}

static struct llama_sampler *s2s_make_sampler(const struct mtmd_context *mctx, unsigned int seed) {
    const struct mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (smpl == NULL) {
        return NULL;
    }
    if (def.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(def.top_k));
    }
    if (def.top_p > 0.0f && def.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(def.top_p, 1));
    }
    if (def.temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(def.temp));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    return smpl;
}

static int s2s_decode_embd_rows(
    struct syllabix_s2s *s2s,
    const float *rows,
    int32_t n_rows,
    llama_pos *n_past,
    int logits_on_last_row,
    bool (*abort_cb)(void *user),
    void *abort_user) {
    if (n_rows <= 0) {
        return 0;
    }
    const int32_t n_embd = llama_model_n_embd_inp(s2s->model);
    for (int32_t offset = 0; offset < n_rows;) {
        if (s2s_aborted(abort_cb, abort_user)) {
            return 1;
        }
        int32_t chunk = n_rows - offset;
        if (chunk > SYLLABIX_S2S_N_BATCH) {
            chunk = SYLLABIX_S2S_N_BATCH;
        }
        struct llama_batch_ext *batch = llama_batch_ext_init(s2s->ctx);
        if (batch == NULL) {
            return -1;
        }
        for (int32_t i = 0; i < chunk; i++) {
            const int32_t row = offset + i;
            struct llama_embd e = {rows + (size_t)row * (size_t)n_embd, 1, (size_t)n_embd};
            const int32_t idx = llama_batch_ext_add_embd(batch, 0, e);
            if (idx < 0) {
                llama_batch_ext_free(batch);
                return -1;
            }
            const llama_pos p[GGML_MROPE_SECTIONS] = {*n_past, *n_past, *n_past, 0};
            llama_batch_ext_set_pos(batch, idx, p);
            (*n_past)++;
            if (logits_on_last_row && row == n_rows - 1) {
                llama_batch_ext_set_output_logits(batch, idx, true);
            }
        }
        const int32_t rc = llama_process(s2s->ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        llama_batch_ext_free(batch);
        if (rc != 0) {
            return -1;
        }
        offset += chunk;
    }
    return 0;
}

/* Public entry is syllabix_s2s_load in shim_vulkan.cpp (C++ exception guard). */
struct syllabix_s2s *syllabix_s2s_load_unguarded(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed) {
    if (model_path == NULL || mmproj_path == NULL || n_threads < 1) {
        return NULL;
    }

    mtmd_helper_log_set(s2s_silent_log, NULL);
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        return NULL;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.n_ctx = 0;
    ctx_params.n_batch = SYLLABIX_S2S_N_BATCH;
    ctx_params.n_ubatch = SYLLABIX_S2S_N_BATCH;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.offload_kqv = false;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        llama_model_free(model);
        return NULL;
    }

    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = false;
    mtmd_params.print_timings = false;
    mtmd_context *mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (mctx == NULL) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    if (!mtmd_support_audio_stream(mctx)) {
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    const int in_rate = mtmd_get_audio_sample_rate(mctx);
    if (in_rate != 16000) {
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    const struct mtmd_gen_audio_info info = mtmd_gen_audio_get_info(mctx);
    if (info.type != MTMD_GEN_AUDIO_TYPE_LFM3A || info.sample_rate <= 0) {
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

    struct llama_sampler *smpl = s2s_make_sampler(mctx, seed);
    if (smpl == NULL) {
        mtmd_helper_gen_audio_free(gen);
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    struct syllabix_s2s *s2s = (struct syllabix_s2s *)malloc(sizeof(struct syllabix_s2s));
    if (s2s == NULL) {
        llama_sampler_free(smpl);
        mtmd_helper_gen_audio_free(gen);
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    s2s->model = model;
    s2s->ctx = ctx;
    s2s->mctx = mctx;
    s2s->smpl = smpl;
    s2s->gen = gen;
    s2s->in_rate = in_rate;
    s2s->out_rate = info.sample_rate;
    s2s->seed = seed;
    return s2s;
}

int syllabix_s2s_in_rate(const struct syllabix_s2s *s2s) {
    return s2s != NULL ? s2s->in_rate : 0;
}

int syllabix_s2s_out_rate(const struct syllabix_s2s *s2s) {
    return s2s != NULL ? s2s->out_rate : 0;
}

void syllabix_s2s_free(struct syllabix_s2s *s2s) {
    if (s2s == NULL) {
        return;
    }
    if (s2s->gen != NULL) {
        mtmd_helper_gen_audio_free(s2s->gen);
    }
    if (s2s->smpl != NULL) {
        llama_sampler_free(s2s->smpl);
    }
    if (s2s->mctx != NULL) {
        mtmd_free(s2s->mctx);
    }
    if (s2s->ctx != NULL) {
        llama_free(s2s->ctx);
    }
    if (s2s->model != NULL) {
        llama_model_free(s2s->model);
    }
    free(s2s);
}

static int s2s_emit_pcm(
    int32_t rate,
    const float *pcm,
    int64_t n_samples,
    int is_last,
    int (*pcm_cb)(int32_t, const float *, int64_t, int, void *),
    void *pcm_user) {
    if (pcm_cb == NULL) {
        return 0;
    }
    return pcm_cb(rate, pcm, n_samples, is_last, pcm_user);
}

int syllabix_s2s_generate(
    struct syllabix_s2s *s2s,
    const float *pcm_in,
    int n_samples,
    int max_out_samples,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples, int is_last, void *user),
    void *pcm_user) {
    if (s2s == NULL || pcm_in == NULL || n_samples < 1 || pcm_cb == NULL) {
        return -1;
    }
    if (s2s_aborted(abort_cb, abort_user)) {
        return 1;
    }

    const int cap = max_out_samples > 0 ? max_out_samples : SYLLABIX_S2S_DEFAULT_MAX_OUT_SAMPLES;

    llama_memory_clear(llama_get_memory(s2s->ctx), true);
    llama_sampler_reset(s2s->smpl);

    struct mtmd_audio_stream_params sparams = mtmd_audio_stream_params_default();
    sparams.n_prompt = 0;
    sparams.prompt = NULL;
    mtmd_audio_stream *stream = mtmd_audio_stream_init(s2s->mctx, sparams);
    if (stream == NULL) {
        return -1;
    }

    if (mtmd_audio_stream_push(stream, pcm_in, (size_t)n_samples) != 0) {
        mtmd_audio_stream_free(stream);
        return -1;
    }
    if (mtmd_audio_stream_flush(stream, true) != 0) {
        mtmd_audio_stream_free(stream);
        return -1;
    }

    llama_pos n_past = 0;
    int64_t out_pcm = 0;

    for (;;) {
        if (s2s_aborted(abort_cb, abort_user)) {
            mtmd_audio_stream_free(stream);
            return 1;
        }
        const int32_t n = mtmd_audio_stream_step(stream, LLAMA_TOKEN_NULL);
        if (n < 0) {
            if (n == -2) {
                mtmd_audio_stream_free(stream);
                return -1;
            }
            break;
        }
        if (n == 0) {
            continue;
        }
        const float *rows = mtmd_audio_stream_get_output_embd(stream);
        if (rows == NULL) {
            mtmd_audio_stream_free(stream);
            return -1;
        }
        const int dec = s2s_decode_embd_rows(s2s, rows, n, &n_past, 0, abort_cb, abort_user);
        if (dec != 0) {
            mtmd_audio_stream_free(stream);
            return dec == 1 ? 1 : -1;
        }
    }
    mtmd_audio_stream_free(stream);

    if (n_past <= 0) {
        return -1;
    }

    llama_sampler_reset(s2s->smpl);
    mtmd_helper_gen_audio_reset(s2s->gen);

    struct mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.seq_id = 0;
    inp.prompt = SYLLABIX_S2S_AUDIO_START_PROMPT;
    inp.prompt_len = SYLLABIX_S2S_AUDIO_START_PROMPT_LEN;
    inp.n_past = (int32_t)n_past;
    inp.out_mode = MTMD_HELPER_GEN_AUDIO_OUT_AUTO;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    inp.stream = true;
    inp.seed = s2s->seed;
    if (mtmd_helper_gen_audio_set_input(s2s->gen, &inp) != 0) {
        return -1;
    }

    for (;;) {
        if (s2s_aborted(abort_cb, abort_user)) {
            return 1;
        }
        const int32_t remaining = mtmd_helper_gen_audio_step_prompt(s2s->gen, SYLLABIX_S2S_N_BATCH);
        if (remaining < 0) {
            return -1;
        }
        if (remaining == 0) {
            break;
        }
    }

    const float *h_state = llama_get_embeddings_ith(s2s->ctx, -1);
    if (h_state == NULL) {
        return -1;
    }

    llama_token sampled = LLAMA_TOKEN_NULL;
    bool stop = false;
    while (!stop && out_pcm < cap) {
        if (s2s_aborted(abort_cb, abort_user)) {
            return 1;
        }
        if (mtmd_helper_gen_audio_next_is_text(s2s->gen)) {
            sampled = llama_sampler_sample(s2s->smpl, s2s->ctx, -1);
            llama_sampler_accept(s2s->smpl, sampled);
        } else {
            sampled = LLAMA_TOKEN_NULL;
        }
        const float *h_next = NULL;
        if (mtmd_helper_gen_audio_step_gen(s2s->gen, sampled, h_state, &h_next, &stop) != 0) {
            return -1;
        }
        int32_t rate = 0;
        const char *data = NULL;
        size_t data_len = 0;
        int64_t n_out = 0;
        if (mtmd_helper_gen_audio_get_output(s2s->gen, &rate, &data, &data_len, &n_out) != 0) {
            return -1;
        }
        if (n_out > 0 && data != NULL && data_len >= (size_t)n_out * sizeof(float)) {
            out_pcm += n_out;
            if (s2s_emit_pcm(rate, (const float *)data, n_out, 0, pcm_cb, pcm_user) != 0) {
                return 1;
            }
        }
        if (h_next != NULL) {
            h_state = h_next;
        } else if (!stop) {
            h_state = llama_get_embeddings_ith(s2s->ctx, -1);
            if (h_state == NULL) {
                break;
            }
        }
        if (out_pcm >= cap) {
            stop = true;
        }
    }

    if (mtmd_helper_gen_audio_flush(s2s->gen) != 0) {
        return s2s_aborted(abort_cb, abort_user) ? 1 : -1;
    }
    int32_t rate = 0;
    const char *data = NULL;
    size_t data_len = 0;
    int64_t n_out = 0;
    if (mtmd_helper_gen_audio_get_output(s2s->gen, &rate, &data, &data_len, &n_out) != 0) {
        return -1;
    }
    if (n_out > 0 && data != NULL && data_len >= (size_t)n_out * sizeof(float)) {
        if (s2s_emit_pcm(rate, (const float *)data, n_out, 1, pcm_cb, pcm_user) != 0) {
            return 1;
        }
    } else {
        if (s2s_emit_pcm(rate, NULL, 0, 1, pcm_cb, pcm_user) != 0) {
            return 1;
        }
    }
    return 0;
}
