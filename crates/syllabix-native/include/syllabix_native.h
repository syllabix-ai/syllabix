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
struct syllabix_qwen_asr;
struct syllabix_mio;
struct llama_model;

void syllabix_native_hush_logs(void);

/* Keep both frontends in the linked binary. */
int syllabix_native_link_anchor(void);

const char *syllabix_llama_system_info(void);
void syllabix_llama_backend_init(void);
void syllabix_llama_backend_free(void);

/* Darwin Metal: -1 (all layers). Linux/Windows CPU: 0. Does not load weights.
 * Vulkan Linux builds: -1 when a device exists, else 0. */
int syllabix_llama_n_gpu_layers(void);
/* Darwin Metal: true. Vulkan Linux with a device: true. Else false. */
int syllabix_whisper_use_gpu(void);

/* Vulkan device 0 probe helpers (exception-safe). Return 0 when Vulkan is off,
 * unavailable, or the call fails; never throw into C/Rust. */
int syllabix_vk_device_count_or_zero(void);
/* 1 + NUL-terminated description in `out` on success; 0 on failure. */
int syllabix_vk_device0_description(char *out, size_t out_cap);
/* 1 + *total_bytes set to device-local heap size on success; 0 on failure. */
int syllabix_vk_device0_vram_bytes(uint64_t *total_bytes);

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

/* 1 when chat decoding uses the LFM sampler (LFM template or GGUF
 * general.sampling.*). Writes preset fields when non-NULL. */
int syllabix_llama_chat_lfm_sampler_preset(
    const struct syllabix_llama *llm,
    float *temp_out,
    int *top_k_out,
    float *repeat_penalty_out);

struct llama_model *syllabix_llama_model(struct syllabix_llama *llm);

int syllabix_llama_run_prompt(
    struct syllabix_llama *llm,
    const char *prompt,
    int prompt_len,
    int thinking,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user,
    const int32_t *render_special,
    int n_render_special);

struct syllabix_chat_session;

void syllabix_chat_string_free(char *s);

struct syllabix_chat_session *syllabix_chat_session_new(
    struct syllabix_llama *llm,
    char *err_out,
    size_t err_cap);
struct syllabix_chat_session *syllabix_chat_session_from_template(
    const char *template_src,
    char *err_out,
    size_t err_cap);
void syllabix_chat_session_free(struct syllabix_chat_session *session);
int syllabix_chat_tools_supported(const struct syllabix_chat_session *session);
const char *syllabix_chat_last_format_name(const struct syllabix_chat_session *session);
int syllabix_chat_apply(
    struct syllabix_chat_session *session,
    const char *messages_json,
    const char *tools_json,
    int enable_thinking,
    char **prompt_out,
    char *err_out,
    size_t err_cap);
int syllabix_chat_parse(
    struct syllabix_chat_session *session,
    const char *text,
    int is_partial,
    char **result_out,
    char *err_out,
    size_t err_cap);
int syllabix_chat_generate(
    struct syllabix_chat_session *session,
    const char *messages_json,
    const char *tools_json,
    int enable_thinking,
    int thinking,
    int n_threads,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*token_cb)(const char *piece, int is_last, void *user),
    void *token_user,
    char *err_out,
    size_t err_cap);

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

/* Qwen3-TTS. Backbone GGUF + mmproj through
 * the shared ggml. Sampling mirrors upstream tools/tts defaults; `seed`
 * pins the fixture. At load the engine synthesizes one short clip from a
 * fixed seed and reuses it as its own speaker reference, so every sentence
 * of every run speaks with the same voice. */
/* Qwen compute placement. Distinct ids so Vulkan is not reported as Metal. */
#define SYLLABIX_QWEN_BACKEND_CPU 0
#define SYLLABIX_QWEN_BACKEND_METAL 1
#define SYLLABIX_QWEN_BACKEND_VULKAN 2

struct syllabix_qwen_tts *syllabix_qwen_tts_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed,
    int backend);
void syllabix_qwen_tts_free(struct syllabix_qwen_tts *tts);

/* Selected compute placement: SYLLABIX_QWEN_BACKEND_*. */
int syllabix_qwen_tts_backend(const struct syllabix_qwen_tts *tts);

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

/* LFM3-Audio turn-based S2S. */
struct syllabix_s2s;
struct syllabix_s2s *syllabix_s2s_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed,
    int n_ctx,
    const char *voice_chat_system_prompt);
void syllabix_s2s_free(struct syllabix_s2s *s2s);
int syllabix_s2s_in_rate(const struct syllabix_s2s *s2s);
int syllabix_s2s_out_rate(const struct syllabix_s2s *s2s);
int syllabix_s2s_n_gpu_layers(const struct syllabix_s2s *s2s);
const char *syllabix_s2s_last_text(const struct syllabix_s2s *s2s);
/* Clear KV and conversation position (n_past = 0). */
void syllabix_s2s_reset(struct syllabix_s2s *s2s);
int syllabix_s2s_n_past(const struct syllabix_s2s *s2s);
/*
 * Voice chat turn prefix/suffix split on <|audio|>. Frees *prefix_out and *suffix_out with
 * syllabix_chat_string_free on success.
 */
int syllabix_s2s_voice_chat_render(struct llama_model *model, const char *system_prompt,
                                   char **prefix_out, char **suffix_out, char *err_out,
                                   size_t err_cap);
/* Continuation turn: prefix/suffix after the prior assistant close + next user block. */
int syllabix_s2s_voice_chat_render_next(struct llama_model *model, const char *system_prompt,
                                        char **prefix_out, char **suffix_out, char *err_out,
                                        size_t err_cap);
/*
 * 0=ok, 1=cancelled, -1=error.
 * pcm_in: f32 mono at in_rate. max_out_samples caps native PCM (0 = default).
 * pcm_cb: native-rate f32; last call is_last!=0. Return 1 to cancel.
 */
int syllabix_s2s_generate(
    struct syllabix_s2s *s2s,
    const float *pcm_in,
    int n_samples,
    int max_out_samples,
    bool (*abort_cb)(void *user),
    void *abort_user,
    int (*pcm_cb)(int32_t sample_rate, const float *pcm, int64_t n_samples, int is_last, void *user),
    void *pcm_user);

/* Qwen3-ASR (mtmd qwen3a encoder + Qwen3 decoder GGUF). Same shared ggml
 * and SYLLABIX_QWEN_BACKEND_* placement as TTS. `language` is an English
 * display name (`English`) or `auto`/NULL/empty for open detection. */
struct syllabix_qwen_asr *syllabix_qwen_asr_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    int backend);
void syllabix_qwen_asr_free(struct syllabix_qwen_asr *asr);

/* Selected compute placement: SYLLABIX_QWEN_BACKEND_*. */
int syllabix_qwen_asr_backend(const struct syllabix_qwen_asr *asr);

/*
 * 0 = ok, 1 = cancelled, -1 = error.
 * `pcm` is 16 kHz mono f32. `out` is a UTF-8 buffer of `out_cap` bytes.
 */
int syllabix_qwen_asr_decode(
    struct syllabix_qwen_asr *asr,
    const float *pcm,
    int n_samples,
    const char *language,
    bool (*abort_cb)(void *user),
    void *abort_user,
    char *out,
    int out_cap);

struct syllabix_parakeet;
struct syllabix_parakeet *syllabix_parakeet_load(const char *gguf_path, int n_threads);
void syllabix_parakeet_free(struct syllabix_parakeet *pk);
void syllabix_parakeet_set_threads(int n_threads); /* process-wide ggml compute threads */
int syllabix_parakeet_model_loaded(const struct syllabix_parakeet *pk);
/* 0 = ok, -1 = error (message in out). pcm: 16 kHz mono f32. */
int syllabix_parakeet_transcribe(struct syllabix_parakeet *pk, const float *pcm, int n_samples,
                                 char *out, int out_cap);

/* Indic-Mio Qwen3 GGUF. Samples speech-token codes (ids 151669–164468
 * minus 151669). Prompt, stop ids, and top-p/temp 0.9 match the model
 * card. `n_gpu_layers` is 0 on CPU and -1 to offload (Metal/Vulkan). */
struct syllabix_mio *syllabix_mio_load(
    const char *path,
    int n_threads,
    int n_gpu_layers);
void syllabix_mio_free(struct syllabix_mio *mio);

/*
 * 0 = ok, 1 = cancelled, -1 = error.
 * Tokenize with parse_special and no BOS. `code_cb` receives each speech
 * code. Stop ids 151645 / 151643 are suppressed until the first speech
 * token. `max_tokens` caps the generation loop (about 750 ≈ 30 s).
 */
int syllabix_mio_generate(
    struct syllabix_mio *mio,
    const char *text,
    unsigned int seed,
    int max_tokens,
    bool (*abort_cb)(void *user),
    void *abort_user,
    void (*code_cb)(int32_t code, void *user),
    void *code_user);

#ifdef __cplusplus
}
#endif

#endif
