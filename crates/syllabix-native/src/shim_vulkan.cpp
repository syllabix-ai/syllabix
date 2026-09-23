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
