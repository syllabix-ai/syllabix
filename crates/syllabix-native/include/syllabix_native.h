#ifndef SYLLABIX_NATIVE_H
#define SYLLABIX_NATIVE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_context;

void syllabix_native_hush_logs(void);

/* Keep both frontends in the linked binary. No model load. */
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

#ifdef __cplusplus
}
#endif

#endif
