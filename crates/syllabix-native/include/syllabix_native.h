#ifndef SYLLABIX_NATIVE_H
#define SYLLABIX_NATIVE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_context;
struct syllabix_llama;

void syllabix_native_hush_logs(void);

/* Keep both frontends in the linked binary. */
int syllabix_native_link_anchor(void);

const char *syllabix_llama_system_info(void);
void syllabix_llama_backend_init(void);
void syllabix_llama_backend_free(void);

struct whisper_context *syllabix_whisper_load(const char *path);
void syllabix_whisper_free(struct whisper_context *ctx);

/* 0 = ok, 1 = cancelled, -1 = error. `out` is a UTF-8 buffer of `out_cap` bytes. */
int syllabix_whisper_decode(
    struct whisper_context *ctx,
    const float *pcm,
    int n_samples,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    char *out,
    int out_cap);

struct syllabix_llama *syllabix_llama_load(const char *path, int n_ctx, int n_threads);
void syllabix_llama_free(struct syllabix_llama *llm);

/*
 * 0 = ok, 1 = cancelled, -1 = error.
 * `roles`/`contents` are parallel arrays (`system` / `user` / `assistant`).
 * `token_cb` is invoked in order; the last call has `is_last != 0`.
 * `token_cb` returns 0 to continue, 1 to cancel, -1 on error.
 */
int syllabix_llama_generate(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int n_predict,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user);

#ifdef __cplusplus
}
#endif

#endif
