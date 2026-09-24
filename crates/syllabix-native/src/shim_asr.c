/* Qwen3-ASR native path. Drives vendored libmtmd audio encode plus a
 * Qwen3 decoder GGUF against the same shared ggml as whisper/llama/TTS.
 *
 * Flow: PCM bitmap -> mtmd_tokenize (media marker) -> mtmd_helper_eval_chunks
 * -> greedy decode until EOG. Prompt wrapping is the Qwen3-ASR chatml
 * surface (`language <Name><asr_text>` plus audio), not the GGUF jinja. */

#include "syllabix_native.h"

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SYLLABIX_QWEN_ASR_N_BATCH 512
#define SYLLABIX_QWEN_ASR_MAX_TOKENS 512
#define SYLLABIX_QWEN_ASR_PROMPT_CAP 1024

struct syllabix_qwen_asr {
    struct llama_model *model;
    struct llama_context *ctx;
    mtmd_context *mctx;
    struct llama_sampler *smpl;
    int backend; /* SYLLABIX_QWEN_BACKEND_* */
};

static int asr_aborted(bool (*abort_cb)(void *user), void *abort_user) {
    return abort_cb != NULL && abort_cb(abort_user);
}

static void asr_silent_log(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

static int asr_lang_is_open(const char *lang) {
    return lang == NULL || lang[0] == '\0' || strcmp(lang, "auto") == 0;
}

static int asr_build_prompt(char *buf, size_t cap, const char *lang, const char *marker) {
    int n;
    if (asr_lang_is_open(lang)) {
        n = snprintf(
            buf,
            cap,
            "<|im_start|>user\n<asr_text>%s<|im_end|>\n<|im_start|>assistant\n",
            marker);
    } else {
        n = snprintf(
            buf,
            cap,
            "<|im_start|>user\nlanguage %s<asr_text>%s<|im_end|>\n"
            "<|im_start|>assistant\nlanguage %s<asr_text>",
            lang,
            marker,
            lang);
    }
    if (n < 0 || (size_t)n >= cap) {
        return -1;
    }
    return 0;
}

/* Public entry is syllabix_qwen_asr_load in shim_vulkan.cpp, which catches
 * ggml backend C++ exceptions thrown through this file (built -fexceptions). */
struct syllabix_qwen_asr *syllabix_qwen_asr_load_unguarded(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    int backend) {
    if (model_path == NULL || mmproj_path == NULL || n_threads < 1) {
        return NULL;
    }
    if (backend != SYLLABIX_QWEN_BACKEND_CPU
        && backend != SYLLABIX_QWEN_BACKEND_METAL
        && backend != SYLLABIX_QWEN_BACKEND_VULKAN) {
        return NULL;
    }

    mtmd_helper_log_set(asr_silent_log, NULL);
    llama_backend_init();

    /* Auto placement is orchestrated by Rust: preferred GPU first, then CPU
     * reload on failure. Explicit cpu/metal/vulkan use this single-attempt entry. */
    const int gpu_layers = backend == SYLLABIX_QWEN_BACKEND_CPU ? 0 : -1;

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (model == NULL) {
        return NULL;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_batch = SYLLABIX_QWEN_ASR_N_BATCH;
    ctx_params.n_ubatch = SYLLABIX_QWEN_ASR_N_BATCH;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.offload_kqv = gpu_layers != 0;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        llama_model_free(model);
        return NULL;
    }

    struct mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = gpu_layers != 0;
    mtmd_params.print_timings = false;
    mtmd_params.n_threads = n_threads;
    mtmd_context *mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (mctx == NULL) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    if (!mtmd_support_audio(mctx) || mtmd_get_audio_sample_rate(mctx) != 16000) {
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (smpl == NULL) {
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    struct syllabix_qwen_asr *asr =
        (struct syllabix_qwen_asr *)malloc(sizeof(struct syllabix_qwen_asr));
    if (asr == NULL) {
        llama_sampler_free(smpl);
        mtmd_free(mctx);
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    asr->model = model;
    asr->ctx = ctx;
    asr->mctx = mctx;
    asr->smpl = smpl;
    asr->backend = backend;
    return asr;
}

int syllabix_qwen_asr_backend(const struct syllabix_qwen_asr *asr) {
    return asr != NULL ? asr->backend : SYLLABIX_QWEN_BACKEND_CPU;
}

void syllabix_qwen_asr_free(struct syllabix_qwen_asr *asr) {
    if (asr == NULL) {
        return;
    }
    if (asr->smpl != NULL) {
        llama_sampler_free(asr->smpl);
    }
    if (asr->mctx != NULL) {
        mtmd_free(asr->mctx);
    }
    if (asr->ctx != NULL) {
        llama_free(asr->ctx);
    }
    if (asr->model != NULL) {
        llama_model_free(asr->model);
    }
    free(asr);
}

int syllabix_qwen_asr_decode(
    struct syllabix_qwen_asr *asr,
    const float *pcm,
    int n_samples,
    const char *language,
    bool (*abort_cb)(void *user),
    void *abort_user,
    char *out,
    int out_cap) {
    if (asr == NULL || pcm == NULL || n_samples < 1 || out == NULL || out_cap < 2) {
        return -1;
    }
    if (asr_aborted(abort_cb, abort_user)) {
        return 1;
    }

    llama_memory_clear(llama_get_memory(asr->ctx), true);
    llama_sampler_reset(asr->smpl);

    const char *marker = mtmd_default_marker();
    char prompt[SYLLABIX_QWEN_ASR_PROMPT_CAP];
    if (asr_build_prompt(prompt, sizeof(prompt), language, marker) != 0) {
        return -1;
    }

    mtmd_bitmap *bitmap = mtmd_bitmap_init_from_audio((size_t)n_samples, pcm);
    if (bitmap == NULL) {
        return -1;
    }

    mtmd_input_text text;
    text.text = prompt;
    text.text_len = strlen(prompt);
    text.add_special = false;
    text.parse_special = true;

    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    if (chunks == NULL) {
        mtmd_bitmap_free(bitmap);
        return -1;
    }
    const mtmd_bitmap *bitmaps[1] = {bitmap};
    int32_t tok = mtmd_tokenize(asr->mctx, chunks, &text, bitmaps, 1);
    mtmd_bitmap_free(bitmap);
    if (tok != 0) {
        mtmd_input_chunks_free(chunks);
        return -1;
    }
    if (asr_aborted(abort_cb, abort_user)) {
        mtmd_input_chunks_free(chunks);
        return 1;
    }

    llama_pos n_past = 0;
    const int32_t n_batch = (int32_t)llama_n_batch(asr->ctx);
    int32_t eval = mtmd_helper_eval_chunks(
        asr->mctx,
        asr->ctx,
        chunks,
        0,
        0,
        n_batch,
        true,
        &n_past);
    mtmd_input_chunks_free(chunks);
    if (eval != 0) {
        return asr_aborted(abort_cb, abort_user) ? 1 : -1;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(asr->model);
    int out_len = 0;
    out[0] = '\0';
    for (int i = 0; i < SYLLABIX_QWEN_ASR_MAX_TOKENS; i++) {
        if (asr_aborted(abort_cb, abort_user)) {
            return 1;
        }
        const llama_token id = llama_sampler_sample(asr->smpl, asr->ctx, -1);
        llama_sampler_accept(asr->smpl, id);
        if (id == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, id)) {
            break;
        }
        char piece[256];
        const int n = llama_token_to_piece(vocab, id, piece, (int32_t)sizeof(piece) - 1, 0, false);
        if (n < 0 || n >= (int)sizeof(piece)) {
            return -1;
        }
        if (out_len + n >= out_cap) {
            break;
        }
        memcpy(out + out_len, piece, (size_t)n);
        out_len += n;
        out[out_len] = '\0';

        llama_batch batch = llama_batch_get_one((llama_token *)&id, 1);
        const int rc = llama_decode(asr->ctx, batch);
        if (rc != 0) {
            return (rc == 2 || asr_aborted(abort_cb, abort_user)) ? 1 : -1;
        }
    }
    return 0;
}
