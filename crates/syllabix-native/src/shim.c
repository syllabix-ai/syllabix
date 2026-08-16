#include "syllabix_native.h"

#include "ggml.h"
#include "llama.h"
#include "whisper.h"

#include <stdint.h>
#include <string.h>

static void silent_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
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
    params.language = "en";
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
        return (abort_cb != NULL && abort_cb(abort_user)) ? 1 : -1;
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
