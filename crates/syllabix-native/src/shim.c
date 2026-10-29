#include "syllabix_native.h"

#include "ggml.h"
#include "llama.h"
#include "whisper.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct syllabix_llama {
    struct llama_model *model;
    struct llama_context *ctx;
};

struct llama_model *syllabix_llama_model(struct syllabix_llama *llm) {
    return llm != NULL ? llm->model : NULL;
}

/* Defined in shim_vulkan.cpp / syllabix_native.h (never throws into C). */

static void silent_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static int aborted(bool (*abort_cb)(void *user), void *abort_user) {
    return abort_cb != NULL && abort_cb(abort_user);
}

static int min_int(int a, int b) {
    return a < b ? a : b;
}

/* Darwin: Metal. Vulkan Linux builds: offload when a device exists.
 * Default Linux/Windows CPU artifacts stay at 0. */
static int syllabix_native_use_gpu(void) {
#if defined(__APPLE__)
    return 1;
#elif defined(GGML_USE_VULKAN)
    return syllabix_vk_device_count_or_zero() > 0 ? 1 : 0;
#else
    return 0;
#endif
}

int syllabix_llama_n_gpu_layers(void) {
    return syllabix_native_use_gpu() ? -1 : 0;
}

int syllabix_whisper_use_gpu(void) {
    return syllabix_native_use_gpu();
}

void syllabix_native_hush_logs(void) {
    whisper_log_set(silent_log, NULL);
    llama_log_set(silent_log, NULL);
}

int syllabix_native_link_anchor(void) {
    /* Volatile so LTO / --gc-sections cannot delete the referred objects.
     * Address-takes only: do not CALL llama/whisper_print_system_info here.
     * They build into shared static std::string buffers (clear + append)
     * that are not thread-safe; racing them against syllabix_llama_system_info
     * from parallel test threads returned wiped strings. See
     * llama_system_info() in src/lib.rs, which memoizes the real call. */
    volatile uintptr_t keep = 0;
    keep ^= (uintptr_t)ggml_new_tensor;
    keep ^= (uintptr_t)whisper_init_from_file_with_params;
    keep ^= (uintptr_t)whisper_full;
    keep ^= (uintptr_t)llama_backend_init;
    keep ^= (uintptr_t)llama_print_system_info;
    keep ^= (uintptr_t)whisper_print_system_info;
    keep ^= (uintptr_t)llama_model_load_from_file;
    keep ^= (uintptr_t)llama_decode;
    keep ^= (uintptr_t)llama_chat_apply_template;
    return keep != 0 ? 1 : 0;
}

const char *syllabix_llama_system_info(void) {
    return llama_print_system_info();
}

void syllabix_llama_backend_init(void) {
    llama_backend_init();
}

void syllabix_llama_backend_free(void) {
    llama_backend_free();
}

struct whisper_context *syllabix_whisper_load(const char *path) {
    struct whisper_context_params params = whisper_context_default_params();
    params.use_gpu = syllabix_whisper_use_gpu() != 0;
    params.flash_attn = false;
    return whisper_init_from_file_with_params(path, params);
}

void syllabix_whisper_free(struct whisper_context *ctx) {
    whisper_free(ctx);
}

int syllabix_whisper_decode(
    struct whisper_context *ctx,
    const float *pcm,
    int n_samples,
    int n_threads,
    const char *language,
    bool (*abort_cb)(void *user),
    void *abort_user,
    char *out,
    int out_cap,
    char *out_lang,
    int out_lang_cap) {
    if (ctx == NULL || pcm == NULL || n_samples <= 0 || out == NULL || out_cap <= 1) {
        return -1;
    }

    struct whisper_state *state = whisper_init_state(ctx);
    if (state == NULL) {
        return -1;
    }

    struct whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.strategy = WHISPER_SAMPLING_GREEDY;
    params.greedy.best_of = 1;
    params.language = language != NULL ? language : "en";
    params.translate = false;
    params.no_context = true;
    params.print_special = false;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.suppress_nst = true;
    params.n_threads = n_threads;
    params.abort_callback = abort_cb;
    params.abort_callback_user_data = abort_user;

    const int rc = whisper_full_with_state(ctx, state, params, pcm, n_samples);
    if (rc != 0) {
        whisper_free_state(state);
        return aborted(abort_cb, abort_user) ? 1 : -1;
    }

    out[0] = '\0';
    int used = 0;
    const int segments = whisper_full_n_segments_from_state(state);
    for (int i = 0; i < segments; i++) {
        const char *piece = whisper_full_get_segment_text_from_state(state, i);
        if (piece == NULL) {
            continue;
        }
        while (*piece == ' ' || *piece == '\t' || *piece == '\n') {
            piece++;
        }
        if (*piece == '\0') {
            continue;
        }
        const size_t piece_len = strlen(piece);
        if (used > 0) {
            if (used + 1 >= out_cap) {
                break;
            }
            out[used++] = ' ';
        }
        if (used + (int)piece_len >= out_cap) {
            const int room = out_cap - used - 1;
            if (room > 0) {
                memcpy(out + used, piece, (size_t)room);
                used += room;
            }
            break;
        }
        memcpy(out + used, piece, piece_len);
        used += (int)piece_len;
    }
    out[used] = '\0';
    if (out_lang != NULL && out_lang_cap > 0) {
        /* Effective language: the requested code, or the one whisper.cpp
         * detected during this run when the caller passed `auto`. */
        const char *code = language != NULL ? language : "en";
        if (strcmp(code, "auto") == 0) {
            const int lang_id = whisper_full_lang_id_from_state(state);
            code = lang_id >= 0 ? whisper_lang_str(lang_id) : "en";
        }
        int lang_used = 0;
        for (; code[lang_used] != '\0' && lang_used < out_lang_cap - 1; lang_used++) {
            out_lang[lang_used] = code[lang_used];
        }
        out_lang[lang_used] = '\0';
    }
    whisper_free_state(state);
    return 0;
}

struct syllabix_llama *syllabix_llama_load(const char *path, int n_threads) {
    if (path == NULL || n_threads < 1) {
        return NULL;
    }

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = syllabix_llama_n_gpu_layers();

    struct llama_model *model = llama_model_load_from_file(path, model_params);
    if (model == NULL) {
        return NULL;
    }

    const int trained = llama_model_n_ctx_train(model);
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_batch = (uint32_t)min_int(trained > 0 ? trained : 512, 512);
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.offload_kqv = syllabix_llama_n_gpu_layers() != 0;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        llama_model_free(model);
        return NULL;
    }

    struct syllabix_llama *llm = (struct syllabix_llama *)malloc(sizeof(struct syllabix_llama));
    if (llm == NULL) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    llm->model = model;
    llm->ctx = ctx;
    return llm;
}

int syllabix_llama_n_ctx(const struct syllabix_llama *llm) {
    if (llm == NULL || llm->ctx == NULL) {
        return 0;
    }
    return (int)llama_n_ctx(llm->ctx);
}

int syllabix_llama_n_ctx_train(const struct syllabix_llama *llm) {
    if (llm == NULL || llm->model == NULL) {
        return 0;
    }
    return llama_model_n_ctx_train(llm->model);
}

int syllabix_llama_count_prompt_tokens(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int thinking) {
    if (llm == NULL || roles == NULL || contents == NULL || n_messages < 1) return -1;
    struct llama_chat_message *chat = calloc((size_t)n_messages, sizeof(*chat));
    if (chat == NULL) return -1;
    for (int i = 0; i < n_messages; ++i) {
        chat[i].role = roles[i];
        chat[i].content = contents[i];
    }
    const char *tmpl = llama_model_chat_template(llm->model, NULL);
    int32_t prompt_len = llama_chat_apply_template(tmpl, chat, (size_t)n_messages, true, NULL, 0);
    if (prompt_len < 1) {
        tmpl = "chatml";
        prompt_len = llama_chat_apply_template(tmpl, chat, (size_t)n_messages, true, NULL, 0);
    }
    if (prompt_len < 1) { free(chat); return -1; }
    char *prompt = malloc((size_t)prompt_len + 1);
    if (prompt == NULL) { free(chat); return -1; }
    if (llama_chat_apply_template(tmpl, chat, (size_t)n_messages, true, prompt, prompt_len + 1) != prompt_len) {
        free(prompt); free(chat); return -1;
    }
    free(chat);
    if (!thinking) {
        static const char suffix[] = "<think>\n</think>\n";
        const size_t suffix_len = sizeof(suffix) - 1;
        char *grown = realloc(prompt, (size_t)prompt_len + suffix_len + 1);
        if (grown == NULL) { free(prompt); return -1; }
        prompt = grown;
        memcpy(prompt + prompt_len, suffix, suffix_len + 1);
        prompt_len += (int32_t)suffix_len;
    }
    const int n_ctx = (int)llama_n_ctx(llm->ctx);
    llama_token *tokens = malloc((size_t)n_ctx * sizeof(*tokens));
    if (tokens == NULL) { free(prompt); return -1; }
    const int count = llama_tokenize(llama_model_get_vocab(llm->model), prompt, prompt_len, tokens, n_ctx, true, true);
    free(tokens);
    free(prompt);
    return count;
}

void syllabix_llama_free(struct syllabix_llama *llm) {
    if (llm == NULL) {
        return;
    }
    if (llm->ctx != NULL) {
        llama_free(llm->ctx);
    }
    if (llm->model != NULL) {
        llama_model_free(llm->model);
    }
    free(llm);
}

static int is_qwen_template(const char *tmpl) {
    return tmpl != NULL && strstr(tmpl, "<tool_call>") != NULL
        && strstr(tmpl, "tool_response") != NULL;
}

static int emit_piece(
    const char *piece,
    int is_last,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user) {
    if (token_cb == NULL) {
        return -1;
    }
    return token_cb(piece, is_last, token_user);
}

/* LiquidAI publishes these settings for LFM2.5 GGUF inference. The model
 * needs sampling enabled; greedy decoding can select EOS in the middle of a
 * response. Keep this model-specific so the launch models remain byte-for-
 * byte on their established greedy path. */
#define SYLLABIX_LFM_TOP_K 50
#define SYLLABIX_LFM_TEMPERATURE 0.1f
#define SYLLABIX_LFM_REPEAT_PENALTY 1.1f
#define SYLLABIX_LFM_REPEAT_LAST_N 64

/* meta-llama/Llama-3.2-1B-Instruct generation_config.json
 * (`do_sample=true, temperature=0.6, top_p=0.9`). */
#define SYLLABIX_LLAMA32_TEMPERATURE 0.6f
#define SYLLABIX_LLAMA32_TOP_P 0.9f

/* Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-2B README presets for text tasks.
 * The voice default is non-thinking; `thinking: true` on qwen3.5-2b uses the
 * thinking preset. `min_p=0.0` is disabled, so no min-p sampler is added.
 * `top_p=1.0` on the non-thinking path keeps every token; the effective
 * shaping there is top-k plus the presence penalty. */
#define SYLLABIX_QWEN_TOP_K 20
#define SYLLABIX_QWEN_TEMPERATURE 1.0f
#define SYLLABIX_QWEN_TOP_P 1.0f
#define SYLLABIX_QWEN_PRESENCE_PENALTY 2.0f
#define SYLLABIX_QWEN_THINK_TOP_P 0.95f
#define SYLLABIX_QWEN_THINK_PRESENCE_PENALTY 1.5f
#define SYLLABIX_QWEN_REPEAT_LAST_N 64

static int is_lfm_template(const char *tmpl) {
    return tmpl != NULL && strstr(tmpl, "tool_call_start") != NULL;
}

static int is_llama32_template(const char *tmpl) {
    return tmpl != NULL && strstr(tmpl, "<|start_header_id|>") != NULL;
}

struct syllabix_lfm_sampler_params {
    int use_lfm;
    int top_k;
    float temp;
    float repeat_penalty;
    int repeat_last_n;
};

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

static void resolve_lfm_sampler_params(const struct llama_model *model,
                                       const char *tmpl,
                                       struct syllabix_lfm_sampler_params *params) {
    memset(params, 0, sizeof(*params));
    params->top_k = SYLLABIX_LFM_TOP_K;
    params->temp = SYLLABIX_LFM_TEMPERATURE;
    params->repeat_penalty = SYLLABIX_LFM_REPEAT_PENALTY;
    params->repeat_last_n = SYLLABIX_LFM_REPEAT_LAST_N;

    float temp = 0.0f;
    float repeat = 0.0f;
    int32_t top_k = 0;
    int32_t last_n = 0;
    const int has_temp = read_meta_float(model, "general.sampling.temp", &temp);
    const int has_top_k = read_meta_int32(model, "general.sampling.top_k", &top_k);
    if (has_temp && has_top_k) {
        params->use_lfm = 1;
        params->temp = temp;
        params->top_k = (int)top_k;
        if (read_meta_float(model, "general.sampling.penalty_repeat", &repeat)) {
            params->repeat_penalty = repeat;
        }
        if (read_meta_int32(model, "general.sampling.penalty_last_n", &last_n) && last_n > 0) {
            params->repeat_last_n = (int)last_n;
        }
        return;
    }
    if (is_lfm_template(tmpl)) {
        params->use_lfm = 1;
    }
}

static void add_lfm_sampler_chain(struct llama_sampler *smpl,
                                  const struct llama_vocab *vocab,
                                  const struct syllabix_lfm_sampler_params *params) {
    llama_sampler_chain_add(
        smpl,
        llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab), params->repeat_last_n, params->repeat_penalty, 0.0f,
            0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(params->top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params->temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

int syllabix_llama_chat_lfm_sampler_preset(const struct syllabix_llama *llm,
                                           float *temp_out,
                                           int *top_k_out,
                                           float *repeat_penalty_out) {
    if (llm == NULL || llm->model == NULL) {
        return 0;
    }
    const char *tmpl = llama_model_chat_template(llm->model, NULL);
    struct syllabix_lfm_sampler_params params;
    resolve_lfm_sampler_params(llm->model, tmpl, &params);
    if (!params.use_lfm) {
        return 0;
    }
    if (temp_out != NULL) {
        *temp_out = params.temp;
    }
    if (top_k_out != NULL) {
        *top_k_out = params.top_k;
    }
    if (repeat_penalty_out != NULL) {
        *repeat_penalty_out = params.repeat_penalty;
    }
    return 1;
}

static struct llama_sampler *make_chat_sampler(const struct syllabix_llama *llm,
                                               const struct llama_vocab *vocab,
                                               int thinking) {
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (smpl == NULL) {
        return NULL;
    }
    const char *tmpl = llama_model_chat_template(llm->model, NULL);
    struct syllabix_lfm_sampler_params lfm;
    resolve_lfm_sampler_params(llm->model, tmpl, &lfm);
    if (lfm.use_lfm) {
        add_lfm_sampler_chain(smpl, vocab, &lfm);
    } else if (is_qwen_template(tmpl)) {
        const float top_p = thinking ? SYLLABIX_QWEN_THINK_TOP_P : SYLLABIX_QWEN_TOP_P;
        const float presence =
            thinking ? SYLLABIX_QWEN_THINK_PRESENCE_PENALTY : SYLLABIX_QWEN_PRESENCE_PENALTY;
        llama_sampler_chain_add(
            smpl,
            llama_sampler_init_penalties(
                llama_vocab_n_tokens(vocab), SYLLABIX_QWEN_REPEAT_LAST_N, 1.0f, 0.0f,
                presence));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(SYLLABIX_QWEN_TOP_K));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(SYLLABIX_QWEN_TEMPERATURE));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else if (is_llama32_template(tmpl)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(SYLLABIX_LLAMA32_TOP_P, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(SYLLABIX_LLAMA32_TEMPERATURE));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }
    return smpl;
}

/* Shared prefill + decode loop over an already-built prompt.
 * Takes `prompt` (caller-allocated, `prompt_len` bytes); frees it before
 * returning. Plain and tools-aware entries share this so only prompt
 * construction differs between the two paths. `thinking` selects the Qwen
 * thinking sampler preset; other templates ignore it. */
static int run_prompt(
    struct syllabix_llama *llm,
    char *prompt,
    int32_t prompt_len,
    int n_threads,
    int thinking,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user,
    const int32_t *render_special,
    int n_render_special) {
    llama_set_n_threads(llm->ctx, n_threads, n_threads);
    const struct llama_vocab *vocab = llama_model_get_vocab(llm->model);
    const int n_ctx = (int)llama_n_ctx(llm->ctx);
    llama_token *tokens = (llama_token *)malloc((size_t)n_ctx * sizeof(llama_token));
    if (tokens == NULL) {
        return -1;
    }
    int32_t n_tokens =
        llama_tokenize(vocab, prompt, prompt_len, tokens, n_ctx, true, true);
    if (n_tokens < 1 || n_tokens >= n_ctx - 1) {
        free(tokens);
        return -1;
    }

    const int n_batch = (int)llama_n_batch(llm->ctx);
    int pos = 0;
    while (pos < n_tokens) {
        if (aborted(abort_cb, abort_user)) {
            free(tokens);
            return 1;
        }
        const int n = min_int(n_batch, n_tokens - pos);
        struct llama_batch batch = llama_batch_get_one(tokens + pos, n);
        const int rc = llama_decode(llm->ctx, batch);
        if (rc != 0) {
            free(tokens);
            return (rc == 2 || aborted(abort_cb, abort_user)) ? 1 : -1;
        }
        pos += n;
    }
    free(tokens);

    struct llama_sampler *smpl = make_chat_sampler(llm, vocab, thinking);
    if (smpl == NULL) {
        return -1;
    }

    char pending[257];
    int pending_len = 0;
    int emitted = 0;
    int status = 0;

    int generated = 0;
    for (;;) {
        if (aborted(abort_cb, abort_user)) {
            status = 1;
            break;
        }
        if (n_tokens + generated >= n_ctx - 1) {
            break;
        }
        const llama_token id = llama_sampler_sample(smpl, llm->ctx, -1);
        if (id == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, id)) {
            break;
        }
        char piece[257];
        /* Template-preserved tokens (tool-call markers such as LFM's
         * <|tool_call_start|>) must reach the chat parser verbatim, matching
         * llama-server; every other control token stays hidden. */
        bool special = false;
        for (int k = 0; k < n_render_special; k++) {
            if (render_special[k] == id) {
                special = true;
                break;
            }
        }
        const int n = llama_token_to_piece(vocab, id, piece, 256, 0, special);
        if (n < 0 || n > 256) {
            status = -1;
            break;
        }
        piece[n] = '\0';
        if (pending_len > 0) {
            const int cb = emit_piece(pending, 0, token_cb, token_user);
            if (cb != 0) {
                status = cb;
                break;
            }
            emitted += 1;
        }
        memcpy(pending, piece, (size_t)n + 1);
        pending_len = n;

        struct llama_batch batch = llama_batch_get_one((llama_token *)&id, 1);
        const int rc = llama_decode(llm->ctx, batch);
        if (rc != 0) {
            status = (rc == 2 || aborted(abort_cb, abort_user)) ? 1 : -1;
            break;
        }
        generated += 1;
    }

    if (status == 0) {
        if (pending_len > 0) {
            status = emit_piece(pending, 1, token_cb, token_user);
            if (status == 0) {
                emitted += 1;
            }
        } else {
            status = emit_piece("", 1, token_cb, token_user);
        }
    } else if (status == 1 && pending_len > 0 && emitted == 0) {
        /* Cancelled before any callback: do not emit a last token. */
    }

    llama_sampler_free(smpl);
    llama_set_abort_callback(llm->ctx, NULL, NULL);
    if (aborted(abort_cb, abort_user) && status != -1) {
        return 1;
    }
    return status;
}

/* Run generation on a pre-rendered prompt (shared sampler loop). */
int syllabix_llama_run_prompt(
    struct syllabix_llama *llm,
    const char *prompt,
    int32_t prompt_len,
    int thinking,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user,
    const int32_t *render_special,
    int n_render_special) {
    if (llm == NULL || llm->ctx == NULL || llm->model == NULL || prompt == NULL || prompt_len < 1
        || token_cb == NULL) {
        return -1;
    }
    if (aborted(abort_cb, abort_user)) {
        return 1;
    }
    llama_set_n_threads(llm->ctx, n_threads, n_threads);
    llama_set_abort_callback(llm->ctx, abort_cb, abort_user);
    llama_memory_clear(llama_get_memory(llm->ctx), true);
    char *owned = (char *)malloc((size_t)prompt_len + 1);
    if (owned == NULL) {
        return -1;
    }
    memcpy(owned, prompt, (size_t)prompt_len);
    owned[prompt_len] = '\0';
    int status = run_prompt(
        llm, owned, prompt_len, n_threads, thinking, abort_cb, abort_user, token_cb, token_user,
        render_special, n_render_special);
    free(owned);
    return status;
}

int syllabix_llama_generate(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int thinking,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user) {
    if (llm == NULL || llm->ctx == NULL || llm->model == NULL || roles == NULL || contents == NULL
        || n_messages < 1 || token_cb == NULL) {
        return -1;
    }
    if (aborted(abort_cb, abort_user)) {
        return 1;
    }

    llama_set_n_threads(llm->ctx, n_threads, n_threads);
    llama_set_abort_callback(llm->ctx, abort_cb, abort_user);
    llama_memory_clear(llama_get_memory(llm->ctx), true);

    struct llama_chat_message *chat =
        (struct llama_chat_message *)calloc((size_t)n_messages, sizeof(struct llama_chat_message));
    if (chat == NULL) {
        return -1;
    }
    for (int i = 0; i < n_messages; i++) {
        if (roles[i] == NULL || contents[i] == NULL) {
            free(chat);
            return -1;
        }
        chat[i].role = roles[i];
        chat[i].content = contents[i];
    }

    const char *tmpl = llama_model_chat_template(llm->model, NULL);
    int32_t prompt_len = llama_chat_apply_template(tmpl, chat, (size_t)n_messages, true, NULL, 0);
    if (prompt_len < 1) {
        tmpl = "chatml";
        prompt_len = llama_chat_apply_template(tmpl, chat, (size_t)n_messages, true, NULL, 0);
    }
    if (prompt_len < 1) {
        free(chat);
        return -1;
    }
    char *prompt = (char *)malloc((size_t)prompt_len + 1);
    if (prompt == NULL) {
        free(chat);
        return -1;
    }
    if (llama_chat_apply_template(tmpl, chat, (size_t)n_messages, true, prompt, prompt_len + 1)
        != prompt_len) {
        free(prompt);
        free(chat);
        return -1;
    }
    prompt[prompt_len] = '\0';
    free(chat);

    if (!thinking) {
        static const char suffix[] = "<think>\n</think>\n";
        const size_t suffix_len = sizeof(suffix) - 1;
        char *grown = (char *)realloc(prompt, (size_t)prompt_len + suffix_len + 1);
        if (grown == NULL) {
            free(prompt);
            return -1;
        }
        prompt = grown;
        memcpy(prompt + prompt_len, suffix, suffix_len + 1);
        prompt_len += (int32_t)suffix_len;
    }

    prompt[prompt_len] = '\0';
    int status = run_prompt(
        llm, prompt, prompt_len, n_threads, thinking, abort_cb, abort_user, token_cb,
        token_user, NULL, 0);
    free(prompt);
    return status;
}

