/* Indic-Mio speech-token LM. Loads a Qwen3 GGUF and samples speech
 * codes; MioCodec (ONNX + Rust iSTFT) turns those into audio.
 *
 * Prompt, stop ids, and sampling match the Indic-Mio model card /
 * LiteRT path. This is not the chat sampler: chat returns text and
 * applies a presence penalty.
 */

#include "syllabix_native.h"

#include "ggml.h"
#include "llama.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SYLLABIX_MIO_TOP_P 0.9f
#define SYLLABIX_MIO_TEMP 0.9f
#define SYLLABIX_MIO_N_CTX 2048
#define SYLLABIX_MIO_N_BATCH 512
#define SYLLABIX_MIO_SPEECH_FIRST 151669
#define SYLLABIX_MIO_SPEECH_LAST 164468
#define SYLLABIX_MIO_STOP_IM_END 151645
#define SYLLABIX_MIO_STOP_EOT 151643

struct syllabix_mio {
    struct llama_model *model;
    struct llama_context *ctx;
};

static int mio_aborted(bool (*abort_cb)(void *user), void *abort_user) {
    return abort_cb != NULL && abort_cb(abort_user);
}

static int mio_is_speech(llama_token token) {
    return token >= SYLLABIX_MIO_SPEECH_FIRST && token <= SYLLABIX_MIO_SPEECH_LAST;
}

static int mio_is_stop(llama_token token) {
    return token == SYLLABIX_MIO_STOP_IM_END || token == SYLLABIX_MIO_STOP_EOT;
}

static void mio_mask_logits(struct llama_context *ctx, int n_vocab, int seen_speech) {
    float *logits = llama_get_logits_ith(ctx, -1);
    if (logits == NULL) {
        return;
    }
    for (int i = 0; i < n_vocab; i++) {
        const int allowed = mio_is_speech(i) || (seen_speech && mio_is_stop(i));
        if (!allowed) {
            logits[i] = -INFINITY;
        }
    }
}

static struct llama_sampler *mio_make_sampler(unsigned int seed) {
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (smpl == NULL) {
        return NULL;
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(SYLLABIX_MIO_TOP_P, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(SYLLABIX_MIO_TEMP));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    return smpl;
}

/* Public entry is syllabix_mio_load in shim_vulkan.cpp. */
struct syllabix_mio *syllabix_mio_load_unguarded(
    const char *path,
    int n_threads,
    int n_gpu_layers) {
    if (path == NULL || n_threads < 1) {
        return NULL;
    }

    llama_backend_init();

    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(path, model_params);
    if (model == NULL) {
        return NULL;
    }

    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = SYLLABIX_MIO_N_CTX;
    ctx_params.n_batch = SYLLABIX_MIO_N_BATCH;
    ctx_params.n_ubatch = SYLLABIX_MIO_N_BATCH;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.offload_kqv = n_gpu_layers != 0;

    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        llama_model_free(model);
        return NULL;
    }

    struct syllabix_mio *mio = (struct syllabix_mio *)malloc(sizeof(struct syllabix_mio));
    if (mio == NULL) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }
    mio->model = model;
    mio->ctx = ctx;
    return mio;
}

void syllabix_mio_free(struct syllabix_mio *mio) {
    if (mio == NULL) {
        return;
    }
    if (mio->ctx != NULL) {
        llama_free(mio->ctx);
    }
    if (mio->model != NULL) {
        llama_model_free(mio->model);
    }
    free(mio);
}

int syllabix_mio_generate(
    struct syllabix_mio *mio,
    const char *text,
    unsigned int seed,
    int max_tokens,
    bool (*abort_cb)(void *user),
    void *abort_user,
    void (*code_cb)(int32_t code, void *user),
    void *code_user) {
    if (mio == NULL || mio->ctx == NULL || mio->model == NULL || text == NULL || code_cb == NULL
        || max_tokens < 1) {
        return -1;
    }
    if (mio_aborted(abort_cb, abort_user)) {
        return 1;
    }

    static const char prefix[] = "<|im_start|>user\n";
    static const char mid[] = "<|im_end|>\n<|im_start|>assistant\n";
    const size_t text_len = strlen(text);
    const size_t prompt_len = (sizeof(prefix) - 1) + text_len + (sizeof(mid) - 1);
    char *prompt = (char *)malloc(prompt_len + 1);
    if (prompt == NULL) {
        return -1;
    }
    memcpy(prompt, prefix, sizeof(prefix) - 1);
    memcpy(prompt + (sizeof(prefix) - 1), text, text_len);
    memcpy(prompt + (sizeof(prefix) - 1) + text_len, mid, sizeof(mid) - 1);
    prompt[prompt_len] = '\0';

    llama_memory_clear(llama_get_memory(mio->ctx), true);

    const struct llama_vocab *vocab = llama_model_get_vocab(mio->model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_ctx = (int)llama_n_ctx(mio->ctx);
    llama_token *tokens = (llama_token *)malloc((size_t)n_ctx * sizeof(llama_token));
    if (tokens == NULL) {
        free(prompt);
        return -1;
    }
    /* add_special = false (no BOS); parse_special = true. */
    const int n_prompt = llama_tokenize(
        vocab, prompt, (int32_t)prompt_len, tokens, n_ctx, false, true);
    free(prompt);
    if (n_prompt <= 0 || n_prompt >= n_ctx) {
        free(tokens);
        return -1;
    }

    int remaining = n_prompt;
    int offset = 0;
    while (remaining > 0) {
        if (mio_aborted(abort_cb, abort_user)) {
            free(tokens);
            return 1;
        }
        const int n = remaining > SYLLABIX_MIO_N_BATCH ? SYLLABIX_MIO_N_BATCH : remaining;
        if (llama_decode(mio->ctx, llama_batch_get_one(tokens + offset, n)) != 0) {
            free(tokens);
            return -1;
        }
        offset += n;
        remaining -= n;
    }
    free(tokens);

    struct llama_sampler *smpl = mio_make_sampler(seed);
    if (smpl == NULL) {
        return -1;
    }

    int seen_speech = 0;
    int rc = 0;
    for (int step = 0; step < max_tokens; step++) {
        if (mio_aborted(abort_cb, abort_user)) {
            rc = 1;
            break;
        }
        mio_mask_logits(mio->ctx, n_vocab, seen_speech);
        llama_token token = llama_sampler_sample(smpl, mio->ctx, -1);
        if (mio_is_stop(token)) {
            if (seen_speech) {
                break;
            }
        } else if (mio_is_speech(token)) {
            seen_speech = 1;
            code_cb(token - SYLLABIX_MIO_SPEECH_FIRST, code_user);
        }
        if (llama_decode(mio->ctx, llama_batch_get_one(&token, 1)) != 0) {
            rc = -1;
            break;
        }
    }

    llama_sampler_free(smpl);
    return rc;
}
