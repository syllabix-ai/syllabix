#include "syllabix_native.h"

#include "ggml_graph.hpp"
#include "parakeet_capi.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>

struct syllabix_parakeet {
    parakeet_ctx *ctx = nullptr;
    std::string load_error;
};

static void copy_cstr(const char *src, char *out, int out_cap) {
    if (out == nullptr || out_cap <= 0) {
        return;
    }
    if (src == nullptr) {
        out[0] = '\0';
        return;
    }
    const int n = static_cast<int>(std::min(std::strlen(src), static_cast<size_t>(out_cap - 1)));
    std::memcpy(out, src, static_cast<size_t>(n));
    out[n] = '\0';
}

extern "C" {

struct syllabix_parakeet *syllabix_parakeet_load(const char *gguf_path, int n_threads) {
    try {
        if (n_threads > 0) {
            pk::set_num_threads(n_threads);
        }
        auto *handle = new syllabix_parakeet();
        if (gguf_path == nullptr) {
            handle->load_error = "gguf path is null";
            return handle;
        }
        handle->ctx = parakeet_capi_load(gguf_path);
        if (handle->ctx == nullptr) {
            const char *err = parakeet_capi_load_error();
            handle->load_error =
                (err != nullptr && err[0] != '\0') ? err : "parakeet load failed";
            return handle;
        }
        return handle;
    } catch (...) {
        return nullptr;
    }
}

void syllabix_parakeet_free(struct syllabix_parakeet *pk) {
    if (pk == nullptr) {
        return;
    }
    if (pk->ctx != nullptr) {
        parakeet_capi_free(pk->ctx);
    }
    delete pk;
}

void syllabix_parakeet_set_threads(int n_threads) {
    try {
        pk::set_num_threads(n_threads);
    } catch (...) {
    }
}

int syllabix_parakeet_transcribe(struct syllabix_parakeet *pk, const float *pcm, int n_samples,
                                 char *out, int out_cap) {
    try {
        if (pk == nullptr || pcm == nullptr || n_samples < 1 || out == nullptr || out_cap < 2) {
            copy_cstr("invalid parakeet transcribe arguments", out, out_cap);
            return -1;
        }
        if (pk->ctx == nullptr) {
            copy_cstr(
                pk->load_error.empty() ? "parakeet context is not loaded" : pk->load_error.c_str(),
                out, out_cap);
            return -1;
        }
        char *text =
            parakeet_capi_transcribe_pcm(pk->ctx, pcm, n_samples, 16000, 2 /* tdt/rnnt */);
        if (text == nullptr) {
            const char *err = parakeet_capi_last_error(pk->ctx);
            copy_cstr((err != nullptr && err[0] != '\0') ? err : "parakeet transcribe failed", out,
                      out_cap);
            return -1;
        }
        copy_cstr(text, out, out_cap);
        parakeet_capi_free_string(text);
        return 0;
    } catch (const std::exception &e) {
        copy_cstr(e.what(), out, out_cap);
        return -1;
    } catch (...) {
        copy_cstr("parakeet transcribe failed", out, out_cap);
        return -1;
    }
}

int syllabix_parakeet_model_loaded(const struct syllabix_parakeet *pk) {
    return pk != nullptr && pk->ctx != nullptr;
}

} // extern "C"
