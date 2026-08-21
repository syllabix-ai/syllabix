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

void syllabix_native_hush_logs(void) {
    whisper_log_set(silent_log, NULL);
    llama_log_set(silent_log, NULL);
}

int syllabix_native_link_anchor(void) {
    /* Volatile so LTO / --gc-sections cannot delete the referred objects. */
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
    const char *llama_info = llama_print_system_info();
    const char *whisper_info = whisper_print_system_info();
    return (llama_info != NULL) && (whisper_info != NULL) && (keep != 0);
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
    params.use_gpu = false;
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
    int out_cap) {
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
    whisper_free_state(state);
    return 0;
}

struct syllabix_llama *syllabix_llama_load(const char *path, int n_ctx, int n_threads) {
    if (path == NULL || n_ctx < 64 || n_threads < 1) {
        return NULL;
    }

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    struct llama_model *model = llama_model_load_from_file(path, model_params);
    if (model == NULL) {
        return NULL;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t)n_ctx;
    ctx_params.n_batch = (uint32_t)min_int(n_ctx, 512);
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.offload_kqv = false;

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

int syllabix_llama_generate(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int n_predict,
    int thinking,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user) {
    if (llm == NULL || llm->ctx == NULL || llm->model == NULL || roles == NULL || contents == NULL
        || n_messages < 1 || n_predict < 1 || token_cb == NULL) {
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

    const struct llama_vocab *vocab = llama_model_get_vocab(llm->model);
    const int n_ctx = (int)llama_n_ctx(llm->ctx);
    llama_token *tokens = (llama_token *)malloc((size_t)n_ctx * sizeof(llama_token));
    if (tokens == NULL) {
        free(prompt);
        return -1;
    }
    int32_t n_tokens =
        llama_tokenize(vocab, prompt, prompt_len, tokens, n_ctx, true, true);
    free(prompt);
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

    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (smpl == NULL) {
        return -1;
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    char pending[257];
    int pending_len = 0;
    int emitted = 0;
    int status = 0;

    for (int i = 0; i < n_predict; i++) {
        if (aborted(abort_cb, abort_user)) {
            status = 1;
            break;
        }
        const llama_token id = llama_sampler_sample(smpl, llm->ctx, -1);
        if (id == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, id)) {
            break;
        }
        char piece[257];
        const int n = llama_token_to_piece(vocab, id, piece, 256, 0, false);
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
