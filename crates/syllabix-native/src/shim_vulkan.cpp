/* Exception boundary between ggml's C++ backends and the C shim / Rust.
 *
 * ggml-Vulkan throws (vk::SystemError, std::runtime_error) on instance init
 * failures (no ICD, API < 1.2) and on device errors. An exception reaching
 * Rust aborts the process, so every entry here converts a throw into the
 * caller's failure value instead. */

#include "syllabix_native.h"

#include "ggml.h"

#if defined(GGML_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

extern "C" struct syllabix_qwen_tts *syllabix_qwen_tts_load_unguarded(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed,
    int backend);

extern "C" struct syllabix_s2s *syllabix_s2s_load_unguarded(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed,
    const char *voice_chat_system_prompt);

extern "C" struct syllabix_qwen_asr *syllabix_qwen_asr_load_unguarded(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    int backend);

extern "C" struct syllabix_mio *syllabix_mio_load_unguarded(
    const char *path,
    int n_threads,
    int n_gpu_layers);

extern "C" int syllabix_vk_device_count_or_zero(void) {
#if defined(GGML_USE_VULKAN)
    try {
        const int count = ggml_backend_vk_get_device_count();
        return count > 0 ? count : 0;
    } catch (...) {
        return 0;
    }
#else
    return 0;
#endif
}

extern "C" int syllabix_vk_device0_description(char *out, size_t out_cap) {
    if (out == nullptr || out_cap == 0) {
        return 0;
    }
    out[0] = '\0';
#if defined(GGML_USE_VULKAN)
    try {
        if (ggml_backend_vk_get_device_count() <= 0) {
            return 0;
        }
        ggml_backend_vk_get_device_description(0, out, out_cap);
        return out[0] != '\0' ? 1 : 0;
    } catch (...) {
        out[0] = '\0';
        return 0;
    }
#else
    (void)out_cap;
    return 0;
#endif
}

extern "C" int syllabix_vk_device0_vram_bytes(uint64_t *total_bytes) {
    if (total_bytes == nullptr) {
        return 0;
    }
    *total_bytes = 0;
#if defined(GGML_USE_VULKAN)
    try {
        if (ggml_backend_vk_get_device_count() <= 0) {
            return 0;
        }
        size_t free_bytes = 0;
        size_t total = 0;
        ggml_backend_vk_get_device_memory(0, &free_bytes, &total);
        if (total == 0) {
            return 0;
        }
        *total_bytes = static_cast<uint64_t>(total);
        return 1;
    } catch (...) {
        *total_bytes = 0;
        return 0;
    }
#else
    return 0;
#endif
}

extern "C" struct syllabix_qwen_tts *syllabix_qwen_tts_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed,
    int backend) {
    try {
        return syllabix_qwen_tts_load_unguarded(model_path, mmproj_path, n_threads, seed, backend);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct syllabix_s2s *syllabix_s2s_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    unsigned int seed,
    const char *voice_chat_system_prompt) {
    try {
        return syllabix_s2s_load_unguarded(model_path, mmproj_path, n_threads, seed,
                                           voice_chat_system_prompt);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct syllabix_qwen_asr *syllabix_qwen_asr_load(
    const char *model_path,
    const char *mmproj_path,
    int n_threads,
    int backend) {
    try {
        return syllabix_qwen_asr_load_unguarded(model_path, mmproj_path, n_threads, backend);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct syllabix_mio *syllabix_mio_load(
    const char *path,
    int n_threads,
    int n_gpu_layers) {
    try {
        return syllabix_mio_load_unguarded(path, n_threads, n_gpu_layers);
    } catch (...) {
        return nullptr;
    }
}
