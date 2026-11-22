/* LFM3-Audio turn-based S2S. Feeds one VAD-committed utterance through
 * mtmd_audio_stream_* then mtmd_helper_gen_audio_* on the shared ggml.
 *
 * Mirrors shim_tts.c: abort callbacks between steps and streaming native-rate
 * f32 PCM. GPU placement follows syllabix_llama_n_gpu_layers / whisper_use_gpu.
 */

#include "syllabix_native.h"

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SYLLABIX_S2S_VOICE_CHAT_SYSTEM_DEFAULT "Respond with interleaved text and audio."
/* ~42 s at 24 kHz, same order of magnitude as Qwen's 512-frame cap. */
#define SYLLABIX_S2S_DEFAULT_MAX_OUT_SAMPLES 1024000
#define SYLLABIX_S2S_N_BATCH 512
/* mtmd-helper-gen.cpp max_chunk_tokens (MAX_TOKEN_PER_CHUNK in the LFM3-Audio reference). */
#define SYLLABIX_S2S_MIXED_MAX_CHUNK_TOKENS 50
#define SYLLABIX_S2S_LAST_TEXT_INIT 256
/* mtmd-helper-gen.cpp lfm3a_gen_audio_pipeline::frame_rate (codec steps per second). */
#define SYLLABIX_S2S_LFM3A_CODEC_FRAME_RATE 12.5
#define SYLLABIX_S2S_INTERLEAVE_N_TEXT 6
#define SYLLABIX_S2S_INTERLEAVE_N_AUDIO 12
#define SYLLABIX_S2S_ASR_MAX_TOKENS 512

typedef enum {
    S2S_ASSIST_IDLE = 0,
    S2S_ASSIST_GEN_TEXT,
    S2S_ASSIST_GEN_AUDIO,
} s2s_assist_phase;

struct syllabix_s2s {
    struct llama_model *model;
    struct llama_context *ctx;
    mtmd_context *mctx;
    struct llama_sampler *smpl;
    mtmd_helper_gen_audio *gen;
    int in_rate;
    int out_rate;
    int n_gpu_layers;
    unsigned int seed;
    char *last_text;
    size_t last_text_len;
    size_t last_text_cap;
    llama_token tok_im_end;
    llama_token tok_text_end;
    llama_token tok_audio_start;
    llama_token tok_mixed_start;
    char *system_prompt;
    llama_pos n_past;
    bool assist_open;
    s2s_assist_phase assist_phase;
    int (*gen_text_cb)(const char *text, int is_last, void *user);
    void *gen_text_user;
    char last_error[256];
};

static void s2s_silent_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static int s2s_aborted(bool (*abort_cb)(void *user), void *abort_user) {
    return abort_cb != NULL && abort_cb(abort_user);
}

static void s2s_clear_last_error(struct syllabix_s2s *s2s) {
    if (s2s != NULL) {
        s2s->last_error[0] = '\0';
    }
}

static void s2s_set_last_error(struct syllabix_s2s *s2s, const char *msg) {
    if (s2s == NULL || msg == NULL) {
        return;
    }
    snprintf(s2s->last_error, sizeof(s2s->last_error), "%s", msg);
}

static int read_meta_float(const struct llama_model *model, const char *key, float *out) {
    char buf[64];
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) <= 0) {
        return 0;
    }
    *out = (float)atof(buf);
    return 1;
}

static int read_meta_int32(const struct llama_model *model, const char *key, int32_t *out) {
    char buf[64];
    if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) <= 0) {
        return 0;
    }
    *out = (int32_t)atoi(buf);
    return 1;
}

static llama_token s2s_token_from_text(const struct llama_vocab *vocab, const char *text) {
    if (text == NULL) {
        return LLAMA_TOKEN_NULL;
    }
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n_vocab; t++) {
        if (strcmp(llama_vocab_get_text(vocab, t), text) == 0) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static int s2s_decode_prompt_tokens(
    struct syllabix_s2s *s2s,
    const llama_token *tokens,
    int32_t n_tokens,
    llama_pos *n_past,
    bool (*abort_cb)(void *user),
    void *abort_user) {
    if (n_tokens <= 0) {
        return 0;
    }
    int32_t offset = 0;
    while (offset < n_tokens) {
        if (s2s_aborted(abort_cb, abort_user)) {
            return 1;
        }
        int32_t chunk = n_tokens - offset;
        if (chunk > SYLLABIX_S2S_N_BATCH) {
            chunk = SYLLABIX_S2S_N_BATCH;
        }
        if (llama_decode(s2s->ctx, llama_batch_get_one(tokens + offset, chunk)) != 0) {
            return -1;
        }
        offset += chunk;
        *n_past += chunk;
    }
    return 0;
}

static struct llama_sampler *s2s_make_sampler(const struct llama_model *model,
                                            const struct llama_vocab *vocab,
                                            unsigned int seed,
                                            const struct llama_logit_bias *bias,
                                            size_t n_bias) {
    float temp = 0.0f;
    int32_t top_k = 1;
    float repeat_penalty = 1.0f;
    int32_t repeat_last_n = 0;
    read_meta_float(model, "general.sampling.temp", &temp);
    read_meta_int32(model, "general.sampling.top_k", &top_k);
    read_meta_float(model, "general.sampling.penalty_repeat", &repeat_penalty);
    read_meta_int32(model, "general.sampling.penalty_last_n", &repeat_last_n);

    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (smpl == NULL) {
        return NULL;
    }
    if (n_bias > 0 && bias != NULL) {
        llama_sampler_chain_add(
            smpl, llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), (int32_t)n_bias,
                                                bias));
    }
    if (repeat_last_n > 0 && repeat_penalty != 1.0f) {
        llama_sampler_chain_add(smpl,
                                llama_sampler_init_penalties(
                                    llama_vocab_n_tokens(vocab), repeat_last_n, repeat_penalty,
                                    0.0f, 0.0f));
    }
    if (top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    }
    if (temp > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    return smpl;
}

static void s2s_last_text_clear(struct syllabix_s2s *s2s) {
    if (s2s == NULL) {
        return;
    }
    s2s->last_text_len = 0;
    if (s2s->last_text != NULL) {
        s2s->last_text[0] = '\0';
    }
}

static int s2s_last_text_append_piece_cb(struct syllabix_s2s *s2s,
                                         const char *piece,
                                         size_t piece_len,
                                         int (*text_cb)(const char *, int, void *),
                                         void *text_user) {
    if (s2s == NULL || piece == NULL || piece_len == 0) {
        return 0;
    }
    const size_t need = s2s->last_text_len + piece_len + 1;
    if (need > s2s->last_text_cap) {
        size_t cap = s2s->last_text_cap > 0 ? s2s->last_text_cap : SYLLABIX_S2S_LAST_TEXT_INIT;
        while (cap < need) {
            cap *= 2;
        }
        char *next = (char *)realloc(s2s->last_text, cap);
        if (next == NULL) {
            return -1;
        }
        s2s->last_text = next;
        s2s->last_text_cap = cap;
    }
    const size_t piece_off = s2s->last_text_len;
    memcpy(s2s->last_text + piece_off, piece, piece_len);
    s2s->last_text_len += piece_len;
    s2s->last_text[s2s->last_text_len] = '\0';
    if (text_cb != NULL) {
        return text_cb(s2s->last_text + piece_off, 0, text_user);
    }
    if (s2s->gen_text_cb != NULL) {
        return s2s->gen_text_cb(s2s->last_text + piece_off, 0, s2s->gen_text_user);
    }
    return 0;
}

static int s2s_last_text_append_piece(struct syllabix_s2s *s2s, const char *piece, size_t piece_len) {
    return s2s_last_text_append_piece_cb(s2s, piece, piece_len, NULL, NULL);
}

static int s2s_maybe_append_sampled_text(struct syllabix_s2s *s2s, llama_token tok) {
    if (s2s == NULL || tok == LLAMA_TOKEN_NULL) {
        return 0;
    }
    if (tok == s2s->tok_im_end || tok == s2s->tok_text_end || tok == s2s->tok_audio_start ||
        tok == s2s->tok_mixed_start) {
        return 0;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(s2s->model);
    char piece[256];
    const int32_t n = llama_token_to_piece(vocab, tok, piece, (int32_t)sizeof(piece), 0, true);
    if (n < 0) {
        return -1;
    }
    if (n > 0) {
        return s2s_last_text_append_piece(s2s, piece, (size_t)n);
    }
    return 0;
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

static llama_pos s2s_gen_reserve_positions(int cap_samples, int out_rate) {
    if (cap_samples <= 0 || out_rate <= 0) {
        return (llama_pos)SYLLABIX_S2S_MIXED_MAX_CHUNK_TOKENS + 64;
    }
    const double n_audio_frames_fixed =
        ceil((double)cap_samples * SYLLABIX_S2S_LFM3A_CODEC_FRAME_RATE / (double)out_rate);
    const double interleave =
        1.0 + (double)SYLLABIX_S2S_INTERLEAVE_N_TEXT / (double)SYLLABIX_S2S_INTERLEAVE_N_AUDIO;
    return (llama_pos)(n_audio_frames_fixed * interleave) + SYLLABIX_S2S_MIXED_MAX_CHUNK_TOKENS + 4;
}

static int s2s_tokenize_prompt(const struct llama_vocab *vocab, const char *text, llama_token **out,
                               int32_t *n_out) {
    *out = NULL;
    *n_out = 0;
    if (text == NULL) {
        return 0;
    }
    int32_t need =
        llama_tokenize(vocab, text, (int32_t)strlen(text), NULL, 0, false, true);
    if (need == INT32_MIN) {
        return -1;
    }
    if (need < 0) {
        need = -need;
    }
    if (need == 0) {
        return 0;
    }
    llama_token *buf = (llama_token *)malloc((size_t)need * sizeof(llama_token));
    if (buf == NULL) {
        return -1;
    }
    const int32_t got = llama_tokenize(vocab, text, (int32_t)strlen(text), buf, need, false, true);
    if (got < 0) {
        free(buf);
        return -1;
    }
    *out = buf;
    *n_out = got;
    return 0;
}

static void s2s_conversation_reset(struct syllabix_s2s *s2s) {
    llama_memory_clear(llama_get_memory(s2s->ctx), true);
    mtmd_helper_gen_audio_reset(s2s->gen);
    s2s->n_past = 0;
    s2s->assist_open = false;
    s2s->assist_phase = S2S_ASSIST_IDLE;
}

static int s2s_decode_one_token(struct syllabix_s2s *s2s, llama_token tok, llama_pos *n_past) {
    if (tok == LLAMA_TOKEN_NULL) {
        return 0;
    }
    const int rc = s2s_decode_prompt_tokens(s2s, &tok, 1, n_past, NULL, NULL);
    if (rc != 0) {
        return -1;
    }
    return 0;
}

static int s2s_decode_text(struct syllabix_s2s *s2s, const char *text, llama_pos *n_past) {
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(s2s->model);
    llama_token *tokens = NULL;
    int32_t n_tokens = 0;
    if (s2s_tokenize_prompt(vocab, text, &tokens, &n_tokens) != 0) {
        return -1;
    }
    const int rc = s2s_decode_prompt_tokens(s2s, tokens, n_tokens, n_past, NULL, NULL);
    free(tokens);
    return rc != 0 ? -1 : 0;
}

/* Barge-in assistant close: change this one string to adjust the decoded marker. */
#define SYLLABIX_S2S_INTERRUPT_MARKER_TEXT "(interrupted)"

static int s2s_close_assistant_on_interrupt(struct syllabix_s2s *s2s, llama_pos *n_past) {
    if (!s2s->assist_open) {
        return 0;
    }
    const uint32_t n_ctx_seq = llama_n_ctx_seq(s2s->ctx);
    llama_pos need = 12;
    if ((llama_pos)*n_past + need > (llama_pos)n_ctx_seq) {
        s2s_conversation_reset(s2s);
        *n_past = 0;
        s2s->assist_open = false;
        s2s->assist_phase = S2S_ASSIST_IDLE;
        return 0;
    }

    (void)mtmd_helper_gen_audio_flush(s2s->gen);
    if (s2s->assist_phase == S2S_ASSIST_GEN_AUDIO && s2s->tok_text_end != LLAMA_TOKEN_NULL) {
        if (s2s_decode_one_token(s2s, s2s->tok_text_end, n_past) != 0) {
            return -1;
        }
    }
    if (s2s_decode_text(s2s, SYLLABIX_S2S_INTERRUPT_MARKER_TEXT, n_past) != 0) {
        return -1;
    }
    /* Segment close only; template EOT comes from the next turn's render_next prefix. */
    mtmd_helper_gen_audio_reset(s2s->gen);
    s2s->assist_open = false;
    s2s->assist_phase = S2S_ASSIST_IDLE;
    return 0;
}

static int s2s_render_turn(struct syllabix_s2s *s2s, char **prefix_out, char **suffix_out,
                           char *err_buf, size_t err_cap) {
    const char *voice_system =
        s2s->system_prompt != NULL ? s2s->system_prompt : SYLLABIX_S2S_VOICE_CHAT_SYSTEM_DEFAULT;
    if (s2s->n_past <= 0) {
        return syllabix_s2s_voice_chat_render(s2s->model, voice_system, prefix_out, suffix_out,
                                              err_buf, err_cap);
    }
    return syllabix_s2s_voice_chat_render_next(s2s->model, voice_system, prefix_out, suffix_out,
                                             err_buf, err_cap);
}

/* Public entry is syllabix_s2s_load in shim_vulkan.cpp (C++ exception guard). */
struct syllabix_s2s *syllabix_s2s_load_unguarded(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed,
    int n_ctx,
    const char *system_prompt) {
    if (model_path == NULL || mmproj_path == NULL || n_threads < 1) {
        return NULL;
    }
    const char *prompt_src =
        (system_prompt != NULL && system_prompt[0] != '\0') ? system_prompt
                                                          : SYLLABIX_S2S_VOICE_CHAT_SYSTEM_DEFAULT;

    mtmd_helper_log_set(s2s_silent_log, NULL);
    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    const int n_gpu_layers = syllabix_llama_n_gpu_layers();
    model_params.n_gpu_layers = n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        return NULL;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.embeddings = true;
    ctx_params.n_ctx = (uint32_t)(n_ctx > 0 ? n_ctx : 0);
    ctx_params.n_batch = SYLLABIX_S2S_N_BATCH;
    ctx_params.n_ubatch = SYLLABIX_S2S_N_BATCH;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.offload_kqv = n_gpu_layers != 0;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        llama_model_free(model);
        return NULL;
    }

    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = syllabix_whisper_use_gpu() != 0;
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
    if (info.n_voices > 0) {
        fprintf(stderr, "lfm3-audio s2s voices:");
        for (int32_t i = 0; i < info.n_voices; i++) {
            const char *name = mtmd_gen_audio_voice_name(mctx, i);
            if (name != NULL && name[0] != '\0') {
                fprintf(stderr, " %s", name);
            }
        }
        fprintf(stderr, "\n");
    }

    mtmd_helper_gen_audio *gen = mtmd_helper_gen_audio_init(ctx, mctx);
    if (gen == NULL) {
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    struct llama_sampler *smpl = s2s_make_sampler(model, vocab, seed, NULL, 0);
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
    memset(s2s, 0, sizeof(*s2s));
    s2s->model = model;
    s2s->ctx = ctx;
    s2s->mctx = mctx;
    s2s->smpl = smpl;
    s2s->gen = gen;
    s2s->in_rate = in_rate;
    s2s->out_rate = info.sample_rate;
    s2s->n_gpu_layers = n_gpu_layers;
    s2s->seed = seed;
    s2s->tok_im_end = s2s_token_from_text(vocab, "<|im_end|>");
    s2s->tok_text_end = s2s_token_from_text(vocab, "<|text_end|>");
    s2s->tok_audio_start = s2s_token_from_text(vocab, "<|audio_start|>");
    s2s->tok_mixed_start = s2s_token_from_text(vocab, "<|mixed_start|>");
    s2s->system_prompt = strdup(prompt_src);
    if (s2s->system_prompt == NULL) {
        syllabix_s2s_free(s2s);
        return NULL;
    }
    return s2s;
}

int syllabix_s2s_in_rate(const struct syllabix_s2s *s2s) {
    return s2s != NULL ? s2s->in_rate : 0;
}

int syllabix_s2s_out_rate(const struct syllabix_s2s *s2s) {
    return s2s != NULL ? s2s->out_rate : 0;
}

int syllabix_s2s_n_gpu_layers(const struct syllabix_s2s *s2s) {
    return s2s != NULL ? s2s->n_gpu_layers : 0;
}

const char *syllabix_s2s_last_text(const struct syllabix_s2s *s2s) {
    if (s2s == NULL || s2s->last_text == NULL || s2s->last_text_len == 0) {
        return "";
    }
    return s2s->last_text;
}

const char *syllabix_s2s_last_error(const struct syllabix_s2s *s2s) {
    if (s2s == NULL || s2s->last_error[0] == '\0') {
        return NULL;
    }
    return s2s->last_error;
}

void syllabix_s2s_reset(struct syllabix_s2s *s2s) {
    if (s2s == NULL) {
        return;
    }
    s2s_conversation_reset(s2s);
}

int syllabix_s2s_n_past(const struct syllabix_s2s *s2s) {
    return s2s != NULL ? (int)s2s->n_past : 0;
}

void syllabix_s2s_free(struct syllabix_s2s *s2s) {
    if (s2s == NULL) {
        return;
    }
    free(s2s->last_text);
    free(s2s->system_prompt);
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

static int s2s_emit_text(int (*text_cb)(const char *, int, void *),
                         void *text_user,
                         const char *text,
                         int is_last) {
    if (text_cb == NULL) {
        return 0;
    }
    return text_cb(text != NULL ? text : "", is_last, text_user);
}

int syllabix_s2s_generate(
    struct syllabix_s2s *s2s,
    const float *pcm_in,
    int n_samples,
    int max_out_samples,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples, int is_last, void *user),
    void *pcm_user,
    int (*text_cb)(const char *text, int is_last, void *user),
    void *text_user) {
    if (s2s == NULL || pcm_in == NULL || n_samples < 1 || pcm_cb == NULL) {
        return -1;
    }
    if (s2s_aborted(abort_cb, abort_user)) {
        return 1;
    }

    const int cap = max_out_samples > 0 ? max_out_samples : SYLLABIX_S2S_DEFAULT_MAX_OUT_SAMPLES;

    s2s_last_text_clear(s2s);
    s2s->assist_open = false;
    s2s->assist_phase = S2S_ASSIST_IDLE;
    s2s->gen_text_cb = text_cb;
    s2s->gen_text_user = text_user;

    char *prefix_prompt = NULL;
    char *suffix_prompt = NULL;
    char err_buf[256];
    int rc = -1;
    llama_token *prefix_tokens = NULL;
    llama_token *suffix_tokens = NULL;
    int32_t n_prefix = 0;
    int32_t n_suffix = 0;
    llama_pos n_past = s2s->n_past;
    llama_memory_t mem = NULL;
    const struct llama_vocab *vocab = llama_model_get_vocab(s2s->model);

    if (s2s_render_turn(s2s, &prefix_prompt, &suffix_prompt, err_buf, sizeof(err_buf)) != 0) {
        if (err_buf[0] != '\0') {
            fprintf(stderr, "lfm3-audio s2s: voice chat render failed: %s\n", err_buf);
        }
        goto finish;
    }
    if (s2s_tokenize_prompt(vocab, prefix_prompt, &prefix_tokens, &n_prefix) != 0) {
        goto finish;
    }
    if (s2s_tokenize_prompt(vocab, suffix_prompt, &suffix_tokens, &n_suffix) != 0) {
        goto finish;
    }

    /* n_prompt must stay 0: non-zero selects live ASR, not voice-chat turn encoding. */
    struct mtmd_audio_stream_params sparams = mtmd_audio_stream_params_default();
    mtmd_audio_stream *stream = mtmd_audio_stream_init(s2s->mctx, sparams);
    if (stream == NULL) {
        goto finish;
    }

    if (mtmd_audio_stream_push(stream, pcm_in, (size_t)n_samples) != 0) {
        mtmd_audio_stream_free(stream);
        goto finish;
    }
    if (mtmd_audio_stream_flush(stream, true) != 0) {
        mtmd_audio_stream_free(stream);
        goto finish;
    }

    const int64_t n_audio_rows = mtmd_audio_stream_get_n_rows_flushed(stream);
    /* Voice-chat turns only reset seq_start on the first encoder rows of a conversation. */
    (void)mtmd_audio_stream_is_seq_start(stream);

    const uint32_t n_ctx_seq = llama_n_ctx_seq(s2s->ctx);
    const llama_pos gen_reserve = s2s_gen_reserve_positions(cap, s2s->out_rate);
    for (;;) {
        const llama_pos need =
            (llama_pos)n_prefix + (llama_pos)n_audio_rows + (llama_pos)n_suffix + 1 + gen_reserve;
        if (s2s->n_past + need <= (llama_pos)n_ctx_seq) {
            break;
        }
        if (s2s->n_past > 0) {
            fprintf(stderr,
                    "lfm3-audio s2s: context full (n_past=%d need=%lld n_ctx=%u), starting a new "
                    "conversation\n",
                    (int)s2s->n_past, (long long)need, n_ctx_seq);
            s2s_conversation_reset(s2s);
            syllabix_chat_string_free(prefix_prompt);
            syllabix_chat_string_free(suffix_prompt);
            prefix_prompt = NULL;
            suffix_prompt = NULL;
            free(prefix_tokens);
            prefix_tokens = NULL;
            free(suffix_tokens);
            suffix_tokens = NULL;
            n_prefix = 0;
            n_suffix = 0;
            if (s2s_render_turn(s2s, &prefix_prompt, &suffix_prompt, err_buf, sizeof(err_buf)) != 0) {
                mtmd_audio_stream_free(stream);
                goto finish;
            }
            if (s2s_tokenize_prompt(vocab, prefix_prompt, &prefix_tokens, &n_prefix) != 0) {
                mtmd_audio_stream_free(stream);
                goto finish;
            }
            if (s2s_tokenize_prompt(vocab, suffix_prompt, &suffix_tokens, &n_suffix) != 0) {
                mtmd_audio_stream_free(stream);
                goto finish;
            }
            continue;
        }
        fprintf(stderr, "lfm3-audio s2s: prompt exceeds context (need=%lld n_ctx=%u)\n",
                (long long)need, n_ctx_seq);
        mtmd_audio_stream_free(stream);
        goto finish;
    }
    free(suffix_tokens);
    suffix_tokens = NULL;

    const int pre =
        s2s_decode_prompt_tokens(s2s, prefix_tokens, n_prefix, &n_past, abort_cb, abort_user);
    free(prefix_tokens);
    prefix_tokens = NULL;
    syllabix_chat_string_free(prefix_prompt);
    prefix_prompt = NULL;
    if (pre != 0) {
        mtmd_audio_stream_free(stream);
        rc = pre == 1 ? 1 : -1;
        goto finish;
    }

    int64_t out_pcm = 0;

    for (;;) {
        if (s2s_aborted(abort_cb, abort_user)) {
            mtmd_audio_stream_free(stream);
            rc = 1;
            goto finish;
        }
        const int32_t n = mtmd_audio_stream_step(stream, LLAMA_TOKEN_NULL);
        if (n < 0) {
            if (n == -2) {
                mtmd_audio_stream_free(stream);
                goto finish;
            }
            break;
        }
        if (n == 0) {
            continue;
        }
        const float *rows = mtmd_audio_stream_get_output_embd(stream);
        if (rows == NULL) {
            mtmd_audio_stream_free(stream);
            goto finish;
        }
        const int dec = s2s_decode_embd_rows(s2s, rows, n, &n_past, 0, abort_cb, abort_user);
        if (dec != 0) {
            mtmd_audio_stream_free(stream);
            rc = dec == 1 ? 1 : -1;
            goto finish;
        }
    }
    mtmd_audio_stream_free(stream);

    if (n_past <= 0) {
        goto finish;
    }

    mtmd_helper_gen_audio_reset(s2s->gen);

    struct mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.seq_id = 0;
    inp.prompt = suffix_prompt;
    inp.prompt_len = suffix_prompt != NULL ? (int32_t)strlen(suffix_prompt) : 0;
    inp.n_past = (int32_t)n_past;
    inp.out_mode = MTMD_HELPER_GEN_AUDIO_OUT_MIXED;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    inp.stream = true;
    inp.seed = s2s->seed;
    if (mtmd_helper_gen_audio_set_input(s2s->gen, &inp) != 0) {
        syllabix_chat_string_free(suffix_prompt);
        goto finish;
    }
    syllabix_chat_string_free(suffix_prompt);
    suffix_prompt = NULL;

    if (s2s->smpl != NULL) {
        llama_sampler_free(s2s->smpl);
    }
    /* Do not apply mtmd_helper_gen_audio_get_logit_bias: live-ASR bias bans <|im_end|>. */
    s2s->smpl = s2s_make_sampler(s2s->model, vocab, s2s->seed, NULL, 0);
    if (s2s->smpl == NULL) {
        goto finish;
    }

    for (;;) {
        if (s2s_aborted(abort_cb, abort_user)) {
            rc = 1;
            goto finish;
        }
        const int32_t remaining = mtmd_helper_gen_audio_step_prompt(s2s->gen, SYLLABIX_S2S_N_BATCH);
        if (remaining < 0) {
            goto finish;
        }
        if (remaining == 0) {
            break;
        }
    }

    const float *h_state = llama_get_embeddings_ith(s2s->ctx, -1);
    if (h_state == NULL) {
        goto finish;
    }

    s2s->assist_open = true;

    mem = llama_get_memory(s2s->ctx);
    llama_token sampled = LLAMA_TOKEN_NULL;
    bool stop = false;
    bool saw_audio_pcm = false;
    int32_t text_since_last_pcm = 0;
    while (!stop && out_pcm < cap) {
        if (s2s_aborted(abort_cb, abort_user)) {
            rc = 1;
            goto finish;
        }
        if (llama_memory_seq_pos_max(mem, 0) + 2 >= (llama_pos)n_ctx_seq) {
            stop = true;
            break;
        }
        s2s->assist_phase = mtmd_helper_gen_audio_next_is_text(s2s->gen) ? S2S_ASSIST_GEN_TEXT
                                                                         : S2S_ASSIST_GEN_AUDIO;
        if (s2s->assist_phase == S2S_ASSIST_GEN_TEXT) {
            text_since_last_pcm++;
            /* After codec EOS the helper stays in MODE_TEXT; interleaved bursts reset on PCM. */
            if (saw_audio_pcm && text_since_last_pcm > SYLLABIX_S2S_MIXED_MAX_CHUNK_TOKENS) {
                stop = true;
                break;
            }
            sampled = llama_sampler_sample(s2s->smpl, s2s->ctx, -1);
            llama_sampler_accept(s2s->smpl, sampled);
            if (sampled == s2s->tok_im_end) {
                stop = true;
            }
            {
                const int text_rc = s2s_maybe_append_sampled_text(s2s, sampled);
                if (text_rc != 0) {
                    rc = text_rc > 0 ? 1 : -1;
                    goto finish;
                }
            }
        } else {
            sampled = LLAMA_TOKEN_NULL;
        }
        const float *h_next = NULL;
        if (mtmd_helper_gen_audio_step_gen(s2s->gen, sampled, h_state, &h_next, &stop) != 0) {
            goto finish;
        }
        int32_t rate = 0;
        const char *data = NULL;
        size_t data_len = 0;
        int64_t n_out = 0;
        if (mtmd_helper_gen_audio_get_output(s2s->gen, &rate, &data, &data_len, &n_out) != 0) {
            goto finish;
        }
        if (n_out > 0 && data != NULL && data_len >= (size_t)n_out * sizeof(float)) {
            out_pcm += n_out;
            saw_audio_pcm = true;
            text_since_last_pcm = 0;
            if (s2s_emit_pcm(rate, (const float *)data, n_out, 0, pcm_cb, pcm_user) != 0) {
                rc = 1;
                goto finish;
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
        rc = s2s_aborted(abort_cb, abort_user) ? 1 : -1;
        goto finish;
    }
    int32_t rate = 0;
    const char *data = NULL;
    size_t data_len = 0;
    int64_t n_out = 0;
    if (mtmd_helper_gen_audio_get_output(s2s->gen, &rate, &data, &data_len, &n_out) != 0) {
        goto finish;
    }
    if (n_out > 0 && data != NULL && data_len >= (size_t)n_out * sizeof(float)) {
        if (s2s_emit_pcm(rate, (const float *)data, n_out, 1, pcm_cb, pcm_user) != 0) {
            rc = 1;
            goto finish;
        }
    } else {
        if (s2s_emit_pcm(rate, NULL, 0, 1, pcm_cb, pcm_user) != 0) {
            rc = 1;
            goto finish;
        }
    }
    if (s2s_emit_text(text_cb, text_user, "", 1) != 0) {
        rc = 1;
        goto finish;
    }
    rc = 0;

finish:
    s2s->gen_text_cb = NULL;
    s2s->gen_text_user = NULL;
    syllabix_chat_string_free(prefix_prompt);
    syllabix_chat_string_free(suffix_prompt);
    free(prefix_tokens);
    free(suffix_tokens);
    mem = llama_get_memory(s2s->ctx);
    if (rc == 0) {
        s2s->n_past = llama_memory_seq_pos_max(mem, 0) + 1;
        return rc;
    }
    if (s2s->assist_open) {
        if (s2s_close_assistant_on_interrupt(s2s, &n_past) != 0) {
            s2s_conversation_reset(s2s);
            return rc;
        }
        s2s->n_past = llama_memory_seq_pos_max(mem, 0) + 1;
    } else {
        s2s_conversation_reset(s2s);
    }
    return rc;
}

static int s2s_decode_prompt_tokens_logits_on_last(
    struct syllabix_s2s *s2s,
    const llama_token *tokens,
    int32_t n_tokens,
    llama_pos *n_past,
    bool (*abort_cb)(void *user),
    void *abort_user) {
    if (n_tokens <= 0) {
        return 0;
    }
    if (n_tokens > 1) {
        const int pre = s2s_decode_prompt_tokens(s2s, tokens, n_tokens - 1, n_past, abort_cb, abort_user);
        if (pre != 0) {
            return pre;
        }
    }
    if (s2s_aborted(abort_cb, abort_user)) {
        return 1;
    }
    llama_token last = tokens[n_tokens - 1];
    struct llama_batch batch = llama_batch_get_one(&last, 1);
    if (llama_decode(s2s->ctx, batch) != 0) {
        return -1;
    }
    (*n_past)++;
    return 0;
}

int syllabix_s2s_transcribe(
    struct syllabix_s2s *s2s,
    const float *pcm_in,
    int n_samples,
    const char *language,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*text_cb)(const char *text, int is_last, void *user),
    void *text_user) {
    if (s2s == NULL || pcm_in == NULL || n_samples < 1) {
        return -1;
    }
    if (s2s_aborted(abort_cb, abort_user)) {
        return 1;
    }

    s2s_conversation_reset(s2s);
    s2s_last_text_clear(s2s);
    s2s_clear_last_error(s2s);

    char *prefix_prompt = NULL;
    char *suffix_prompt = NULL;
    char err_buf[256];
    err_buf[0] = '\0';
    int rc = -1;
    llama_token *prefix_tokens = NULL;
    llama_token *suffix_tokens = NULL;
    int32_t n_prefix = 0;
    int32_t n_suffix = 0;
    llama_pos n_past = 0;
    struct llama_sampler *local_smpl = NULL;
    const struct llama_vocab *vocab = llama_model_get_vocab(s2s->model);
    const uint32_t n_ctx_seq = llama_n_ctx_seq(s2s->ctx);
    llama_memory_t mem = NULL;

    if (syllabix_s2s_asr_render(s2s->model, language, &prefix_prompt, &suffix_prompt, err_buf,
                                  sizeof(err_buf)) != 0) {
        if (err_buf[0] != '\0') {
            s2s_set_last_error(s2s, err_buf);
        }
        goto finish;
    }
    if (s2s_tokenize_prompt(vocab, prefix_prompt, &prefix_tokens, &n_prefix) != 0) {
        goto finish;
    }
    if (s2s_tokenize_prompt(vocab, suffix_prompt, &suffix_tokens, &n_suffix) != 0) {
        goto finish;
    }
    if (n_suffix <= 0) {
        s2s_set_last_error(s2s, "LFM3-Audio ASR suffix tokenized to no tokens");
        rc = -1;
        goto finish;
    }

    struct mtmd_audio_stream_params sparams = mtmd_audio_stream_params_default();
    mtmd_audio_stream *stream = mtmd_audio_stream_init(s2s->mctx, sparams);
    if (stream == NULL) {
        goto finish;
    }
    if (mtmd_audio_stream_push(stream, pcm_in, (size_t)n_samples) != 0) {
        mtmd_audio_stream_free(stream);
        goto finish;
    }
    if (mtmd_audio_stream_flush(stream, true) != 0) {
        mtmd_audio_stream_free(stream);
        goto finish;
    }

    const int64_t n_audio_rows = mtmd_audio_stream_get_n_rows_flushed(stream);

    const int pre =
        s2s_decode_prompt_tokens(s2s, prefix_tokens, n_prefix, &n_past, abort_cb, abort_user);
    free(prefix_tokens);
    prefix_tokens = NULL;
    syllabix_chat_string_free(prefix_prompt);
    prefix_prompt = NULL;
    if (pre != 0) {
        mtmd_audio_stream_free(stream);
        rc = pre == 1 ? 1 : -1;
        goto finish;
    }

    for (;;) {
        if (s2s_aborted(abort_cb, abort_user)) {
            mtmd_audio_stream_free(stream);
            rc = 1;
            goto finish;
        }
        const int32_t n = mtmd_audio_stream_step(stream, LLAMA_TOKEN_NULL);
        if (n < 0) {
            if (n == -2) {
                mtmd_audio_stream_free(stream);
                goto finish;
            }
            break;
        }
        if (n == 0) {
            continue;
        }
        const float *rows = mtmd_audio_stream_get_output_embd(stream);
        if (rows == NULL) {
            mtmd_audio_stream_free(stream);
            goto finish;
        }
        const int dec = s2s_decode_embd_rows(s2s, rows, n, &n_past, 0, abort_cb, abort_user);
        if (dec != 0) {
            mtmd_audio_stream_free(stream);
            rc = dec == 1 ? 1 : -1;
            goto finish;
        }
    }
    mtmd_audio_stream_free(stream);

    if (n_audio_rows <= 0) {
        if (s2s_emit_text(text_cb, text_user, "", 1) != 0) {
            rc = 1;
        } else {
            rc = 0;
        }
        goto finish;
    }

    const int suf =
        s2s_decode_prompt_tokens_logits_on_last(s2s, suffix_tokens, n_suffix, &n_past, abort_cb,
                                                abort_user);
    free(suffix_tokens);
    suffix_tokens = NULL;
    syllabix_chat_string_free(suffix_prompt);
    suffix_prompt = NULL;
    if (suf != 0) {
        rc = suf == 1 ? 1 : -1;
        goto finish;
    }

    struct llama_logit_bias bias[2];
    size_t n_bias = 0;
    if (s2s->tok_audio_start != LLAMA_TOKEN_NULL) {
        bias[n_bias++] = (struct llama_logit_bias){s2s->tok_audio_start, -INFINITY};
    }
    if (s2s->tok_mixed_start != LLAMA_TOKEN_NULL) {
        bias[n_bias++] = (struct llama_logit_bias){s2s->tok_mixed_start, -INFINITY};
    }
    local_smpl = s2s_make_sampler(s2s->model, vocab, s2s->seed, bias, n_bias);
    if (local_smpl == NULL) {
        goto finish;
    }

    bool stop = false;
    int asr_tokens = 0;
    while (!stop) {
        if (s2s_aborted(abort_cb, abort_user)) {
            rc = 1;
            goto finish;
        }
        if (asr_tokens >= SYLLABIX_S2S_ASR_MAX_TOKENS) {
            stop = true;
            break;
        }
        mem = llama_get_memory(s2s->ctx);
        if (llama_memory_seq_pos_max(mem, 0) + 2 >= (llama_pos)n_ctx_seq) {
            stop = true;
            break;
        }
        const llama_token sampled = llama_sampler_sample(local_smpl, s2s->ctx, -1);
        asr_tokens++;
        llama_sampler_accept(local_smpl, sampled);
        if (sampled == s2s->tok_im_end || sampled == s2s->tok_text_end) {
            stop = true;
            break;
        }
        if (sampled != LLAMA_TOKEN_NULL && sampled != s2s->tok_audio_start &&
            sampled != s2s->tok_mixed_start) {
            char piece[256];
            const int32_t n =
                llama_token_to_piece(vocab, sampled, piece, (int32_t)sizeof(piece), 0, true);
            if (n < 0) {
                goto finish;
            }
            if (n > 0) {
                const int text_rc = s2s_last_text_append_piece_cb(s2s, piece, (size_t)n, text_cb,
                                                                  text_user);
                if (text_rc != 0) {
                    rc = text_rc > 0 ? 1 : -1;
                    goto finish;
                }
            }
        }
        if (s2s_decode_one_token(s2s, sampled, &n_past) != 0) {
            goto finish;
        }
    }

    if (s2s_emit_text(text_cb, text_user, "", 1) != 0) {
        rc = 1;
        goto finish;
    }
    rc = 0;

finish:
    free(prefix_tokens);
    free(suffix_tokens);
    syllabix_chat_string_free(prefix_prompt);
    syllabix_chat_string_free(suffix_prompt);
    if (local_smpl != NULL) {
        llama_sampler_free(local_smpl);
    }
    if (rc != 0 && s2s->last_error[0] == '\0' && err_buf[0] != '\0') {
        s2s_set_last_error(s2s, err_buf);
    }
    if (rc != 0 && s2s->last_error[0] == '\0') {
        s2s_set_last_error(s2s, "LFM3-Audio ASR transcription failed");
    }
    s2s_conversation_reset(s2s);
    return rc;
}

int syllabix_s2s_speak(
    struct syllabix_s2s *s2s,
    const char *text,
    const char *language,
    const char *voice,
    int max_out_samples,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples, int is_last, void *user),
    void *pcm_user) {
    if (s2s == NULL || text == NULL || pcm_cb == NULL) {
        return -1;
    }
    if (s2s_aborted(abort_cb, abort_user)) {
        return 1;
    }

    const int cap = max_out_samples > 0 ? max_out_samples : SYLLABIX_S2S_DEFAULT_MAX_OUT_SAMPLES;

    s2s_conversation_reset(s2s);
    s2s_clear_last_error(s2s);

    char *prompt = NULL;
    char err_buf[256];
    err_buf[0] = '\0';
    int rc = -1;
    int64_t out_pcm = 0;

    if (syllabix_s2s_tts_render(s2s->model, text, language, voice, &prompt, err_buf, sizeof(err_buf)) !=
        0) {
        if (err_buf[0] != '\0') {
            s2s_set_last_error(s2s, err_buf);
        }
        goto finish;
    }

    mtmd_helper_gen_audio_reset(s2s->gen);

    struct mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.seq_id = 0;
    inp.prompt = prompt;
    inp.prompt_len = prompt != NULL ? (int32_t)strlen(prompt) : 0;
    inp.n_past = 0;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    inp.stream = true;
    inp.seed = s2s->seed;
    if (mtmd_helper_gen_audio_set_input(s2s->gen, &inp) != 0) {
        goto finish;
    }
    syllabix_chat_string_free(prompt);
    prompt = NULL;

    for (;;) {
        if (s2s_aborted(abort_cb, abort_user)) {
            rc = 1;
            goto finish;
        }
        const int32_t remaining = mtmd_helper_gen_audio_step_prompt(s2s->gen, SYLLABIX_S2S_N_BATCH);
        if (remaining < 0) {
            goto finish;
        }
        if (remaining == 0) {
            break;
        }
    }

    const float *h_state = llama_get_embeddings_ith(s2s->ctx, -1);
    if (h_state == NULL) {
        goto finish;
    }

    bool stop = false;
    while (!stop && out_pcm < cap) {
        if (s2s_aborted(abort_cb, abort_user)) {
            rc = 1;
            goto finish;
        }
        const float *h_next = NULL;
        if (mtmd_helper_gen_audio_step_gen(s2s->gen, LLAMA_TOKEN_NULL, h_state, &h_next, &stop) != 0) {
            goto finish;
        }
        int32_t rate = 0;
        const char *data = NULL;
        size_t data_len = 0;
        int64_t n_out = 0;
        if (mtmd_helper_gen_audio_get_output(s2s->gen, &rate, &data, &data_len, &n_out) != 0) {
            goto finish;
        }
        if (n_out > 0 && data != NULL && data_len >= (size_t)n_out * sizeof(float)) {
            out_pcm += n_out;
            if (s2s_emit_pcm(rate, (const float *)data, n_out, 0, pcm_cb, pcm_user) != 0) {
                rc = 1;
                goto finish;
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
        rc = s2s_aborted(abort_cb, abort_user) ? 1 : -1;
        goto finish;
    }
    int32_t rate = 0;
    const char *data = NULL;
    size_t data_len = 0;
    int64_t n_out = 0;
    if (mtmd_helper_gen_audio_get_output(s2s->gen, &rate, &data, &data_len, &n_out) != 0) {
        goto finish;
    }
    if (n_out > 0 && data != NULL && data_len >= (size_t)n_out * sizeof(float)) {
        if (s2s_emit_pcm(rate, (const float *)data, n_out, 1, pcm_cb, pcm_user) != 0) {
            rc = 1;
            goto finish;
        }
    } else {
        if (s2s_emit_pcm(rate, NULL, 0, 1, pcm_cb, pcm_user) != 0) {
            rc = 1;
            goto finish;
        }
    }
    rc = 0;

finish:
    syllabix_chat_string_free(prompt);
    if (rc != 0 && s2s->last_error[0] == '\0' && err_buf[0] != '\0') {
        s2s_set_last_error(s2s, err_buf);
    }
    if (rc != 0 && s2s->last_error[0] == '\0') {
        s2s_set_last_error(s2s, "LFM3-Audio TTS speak failed");
    }
    s2s_conversation_reset(s2s);
    return rc;
}
