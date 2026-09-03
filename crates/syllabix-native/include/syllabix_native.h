#ifndef SYLLABIX_NATIVE_H
#define SYLLABIX_NATIVE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_context;
struct syllabix_llama;
struct syllabix_qwen_tts;

void syllabix_native_hush_logs(void);

/* Keep both frontends in the linked binary. */
int syllabix_native_link_anchor(void);

const char *syllabix_llama_system_info(void);
void syllabix_llama_backend_init(void);
void syllabix_llama_backend_free(void);

/* Darwin Metal: -1 (all layers). Linux/Windows CPU: 0. Does not load weights. */
int syllabix_llama_n_gpu_layers(void);
/* Darwin Metal: true. Linux/Windows: false. Does not load weights. */
int syllabix_whisper_use_gpu(void);

struct whisper_context *syllabix_whisper_load(const char *path);
void syllabix_whisper_free(struct whisper_context *ctx);

/* 0 = ok, 1 = cancelled, -1 = error. `out` is a UTF-8 buffer of `out_cap` bytes.
 * `language` is a whisper.cpp language id (`en`) or `auto`. NULL means `en`.
 * `out_lang` receives the effective language id (`en`, or the code detected by
 * whisper.cpp when `language` was `auto`). May be NULL to skip the copy. */
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
    int out_lang_cap);

/* Context length comes from the GGUF (`n_ctx_train`), not a caller budget. */
struct syllabix_llama *syllabix_llama_load(const char *path, int n_threads);
void syllabix_llama_free(struct syllabix_llama *llm);
int syllabix_llama_n_ctx(const struct syllabix_llama *llm);
int syllabix_llama_n_ctx_train(const struct syllabix_llama *llm);

/*
 * 0 = ok, 1 = cancelled, -1 = error.
 * `roles`/`contents` are parallel arrays (`system` / `user` / `assistant`).
 * `thinking` is 0 to append an empty `<think></think>` closer for models
 * whose chat format supports it; callers use 1 for every other model.
 * Generate until EOS, cancel, or the model's context window is full.
 * `token_cb` is invoked in order; the last call has `is_last != 0`.
 * `token_cb` returns 0 to continue, 1 to cancel, -1 on error.
 */
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
    void *token_user);

/* Issue 87: tools-aware entry for Qwen3 / Qwen3.5. `tools_json` is the JSON
 * array of tool definitions (the same contract the online adapter sends).
 * NULL/empty — or a non-Qwen model template — delegates to the plain path
 * byte-for-byte, so tool-free generation is unchanged. `tool`-role messages
 * render as native `<tool_response>` turns; see `syllabix_llama_render_qwen`. */
int syllabix_llama_generate_with_tools(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int thinking,
    const char *tools_json,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user);

/* Pure Qwen prompt renderer (no model handle): returns the prompt length, or
 * -1 on error. With `out == NULL`, measures only. Unit-test seam for the
 * tools preamble and `tool`-role grouping without loading weights. */
int syllabix_llama_render_qwen(
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    const char *tools_json,
    int thinking,
    char *out,
    int out_cap);

int syllabix_llama_count_prompt_tokens_with_tools(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int thinking,
    const char *tools_json);

/* Phase 4 LFM spike: tools-aware entry for LiquidAI LFM2.5. `tools_json` is
 * the JSON array of tool definitions (the same shared contract the online
 * adapter sends). NULL/empty — or a non-LFM model template — delegates to
 * the plain path byte-for-byte, so tool-free generation is unchanged.
 * `tool`-role messages render as native `<|im_start|>tool` turns; see
 * `syllabix_llama_render_lfm`. Separate entry from the Qwen path (second
 * tool-dialect parser exception); neither path touches the other. */
int syllabix_llama_generate_with_lfm_tools(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int thinking,
    const char *tools_json,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user);

/* Pure LFM prompt renderer (no model handle): returns the prompt length, or
 * -1 on error. With `out == NULL`, measures only. Unit-test seam for the
 * tools preamble and `tool`-role turns without loading weights. */
int syllabix_llama_render_lfm(
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    const char *tools_json,
    int thinking,
    char *out,
    int out_cap);

int syllabix_llama_count_prompt_tokens_with_lfm_tools(
    struct syllabix_llama *llm,
    const char *const *roles,
    const char *const *contents,
    int n_messages,
    int thinking,
    const char *tools_json);

/* Qwen3-TTS (row 31; row 32 voice anchor). Backbone GGUF + mmproj through
 * the shared ggml. Sampling mirrors upstream tools/tts defaults; `seed`
 * pins the fixture. At load the engine synthesizes one short clip from a
 * fixed seed and reuses it as its own speaker reference, so every sentence
 * of every run speaks with the same voice. */
struct syllabix_qwen_tts *syllabix_qwen_tts_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed);
void syllabix_qwen_tts_free(struct syllabix_qwen_tts *tts);

/* 1 when the self-voice reference engaged, 0 when generation fell back to
 * unconditioned sampling (anchor failure). Diagnostics and tests. */
int syllabix_qwen_tts_has_voice(const struct syllabix_qwen_tts *tts);

/*
 * 0 = ok, 1 = cancelled, -1 = error.
 * Synthesizes one sentence. `out_pcm` receives malloc'd mono i16
 * (clipped like the upstream WAV writer); free with
 * syllabix_qwen_tts_pcm_free. `lang` may be NULL or empty for `en`.
 */
int syllabix_qwen_tts_synthesize(
    struct syllabix_qwen_tts *tts,
    const char *text,
    const char *lang,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int32_t *out_sample_rate,
    int16_t **out_pcm,
    int64_t *out_n_samples);

/* Incremental variant. pcm_cb receives native-rate f32 PCM after each
 * vocoder window; its final invocation has is_last != 0 and may be empty.
 * Return 0 to continue, 1 to cancel, or -1 for an error. */
int syllabix_qwen_tts_synthesize_streaming(
    struct syllabix_qwen_tts *tts,
    const char *text,
    const char *lang,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples,
                  int is_last, void *user),
    void *pcm_user);
void syllabix_qwen_tts_pcm_free(int16_t *pcm);

#ifdef __cplusplus
}
#endif

#endif
