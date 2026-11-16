cmake_minimum_required(VERSION 3.14)
project(syllabix-native C CXX)

# One static ggml. Both frontends link it. No shared libs in the product binary.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "static ggml / whisper / llama" FORCE)

# Sequence 26: Darwin uses OS-shipped Metal + Accelerate. Linux/Windows stay
# portable CPU. CUDA/HIP/OpenMP/OpenBLAS stay off everywhere. Vulkan sources
# are vendored; default remains OFF. build.rs may pass -DGGML_VULKAN=ON when
# SYLLABIX_GGML_VULKAN=1 on Linux — do not FORCE here or that -D is clobbered.
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "ggml: use Vulkan")
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(GGML_CANN OFF CACHE BOOL "" FORCE)
set(GGML_MUSA OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
set(GGML_ZDNN OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

if (APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    set(GGML_BLAS ON CACHE BOOL "" FORCE)
    set(GGML_BLAS_VENDOR "Apple" CACHE STRING "" FORCE)
    set(GGML_ACCELERATE ON CACHE BOOL "" FORCE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(GGML_NATIVE ON CACHE BOOL "" FORCE)
        # KleidiAI SME needs ___arm_tpidr2_save (compiler-rt). That does not
        # link from rustc on Darwin. Metal + Accelerate stay on.
        set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
    else()
        set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
        set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
    endif()
else()
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
set(LLAMA_LLGUIDANCE OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_SYSTEM_GGML OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/llama.cpp")
set(WHISPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/whisper.cpp")

add_subdirectory("${LLAMA_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp")

if (NOT TARGET ggml)
    message(FATAL_ERROR "llama.cpp did not create the ggml target")
endif()

add_library(whisper STATIC "${WHISPER_DIR}/src/whisper.cpp")
target_include_directories(whisper PUBLIC
    "${WHISPER_DIR}/include"
    "${WHISPER_DIR}/src"
)
target_link_libraries(whisper PUBLIC ggml)
target_compile_features(whisper PUBLIC cxx_std_17)
target_compile_definitions(whisper PRIVATE WHISPER_VERSION="1.9.2")
find_package(Threads REQUIRED)
target_link_libraries(whisper PUBLIC Threads::Threads)

# Qwen3-TTS audio path. Upstream tools/mtmd from the same llama.cpp pin as above
# (see vendor/README.md); uses the same ggml as whisper/llama. Video support and
# its ffmpeg/subprocess dependencies stay off; audio generation only.
set(MTMD_VIDEO OFF CACHE BOOL "mtmd video support" FORCE)
add_subdirectory("${LLAMA_DIR}/tools/mtmd" "${CMAKE_BINARY_DIR}/mtmd")

option(SYLLABIX_PARAKEET "Fluid Speech Pro via parakeet.cpp" OFF)
set(SYLLABIX_NATIVE_SRC src/shim.c src/shim_chat.cpp src/shim_tts.c src/shim_asr.c src/shim_mio.c src/shim_vulkan.cpp)
if (SYLLABIX_PARAKEET)
    set(PARAKEET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/parakeet.cpp")
    set(PARAKEET_SRC
        ${PARAKEET_DIR}/src/parakeet.cpp
        ${PARAKEET_DIR}/src/model.cpp
        ${PARAKEET_DIR}/src/vad_head.cpp
        ${PARAKEET_DIR}/src/vad_segmenter.cpp
        ${PARAKEET_DIR}/src/vad_json.cpp
        ${PARAKEET_DIR}/src/bundle.cpp
        ${PARAKEET_DIR}/src/bundle_map.cpp
        ${PARAKEET_DIR}/src/parakeet_capi.cpp
        ${PARAKEET_DIR}/src/common.cpp
        ${PARAKEET_DIR}/src/audio_io.cpp
        ${PARAKEET_DIR}/src/transcription_json.cpp
        ${PARAKEET_DIR}/src/model_loader.cpp
        ${PARAKEET_DIR}/src/backend.cpp
        ${PARAKEET_DIR}/src/ggml_graph.cpp
        ${PARAKEET_DIR}/src/backend_pool.cpp
        ${PARAKEET_DIR}/src/fft.cpp
        ${PARAKEET_DIR}/src/mel.cpp
        ${PARAKEET_DIR}/src/mel_gpu.cpp
        ${PARAKEET_DIR}/src/subsampling.cpp
        ${PARAKEET_DIR}/src/relpos_attention.cpp
        ${PARAKEET_DIR}/src/conformer.cpp
        ${PARAKEET_DIR}/src/pos_enc.cpp
        ${PARAKEET_DIR}/src/encoder.cpp
        ${PARAKEET_DIR}/src/streaming_encoder.cpp
        ${PARAKEET_DIR}/src/streaming.cpp
        ${PARAKEET_DIR}/src/buffered_streaming.cpp
        ${PARAKEET_DIR}/src/ctc_decoder.cpp
        ${PARAKEET_DIR}/src/prediction.cpp
        ${PARAKEET_DIR}/src/joint.cpp
        ${PARAKEET_DIR}/src/prompt_kernel.cpp
        ${PARAKEET_DIR}/src/ternary.cpp
        ${PARAKEET_DIR}/src/exact_matvec.cpp
        ${PARAKEET_DIR}/src/ternary_kernels_x86.cpp
        ${PARAKEET_DIR}/src/ternary_kernels_neon.cpp
        ${PARAKEET_DIR}/src/tdt.cpp
        ${PARAKEET_DIR}/src/rnnt.cpp
        ${PARAKEET_DIR}/src/transducer_batch.cpp
        ${PARAKEET_DIR}/src/tokenizer.cpp
        ${PARAKEET_DIR}/src/search.cpp
        ${PARAKEET_DIR}/src/transcription.cpp
        ${PARAKEET_DIR}/src/diarization.cpp
        ${PARAKEET_DIR}/src/diarization_encoder.cpp
        ${PARAKEET_DIR}/src/diarization_head.cpp
        ${PARAKEET_DIR}/src/sas_merge.cpp
        ${PARAKEET_DIR}/src/asr_committer.cpp
        ${PARAKEET_DIR}/src/diar_pcm_stream.cpp
        ${PARAKEET_DIR}/src/scene_stream.cpp
        ${PARAKEET_DIR}/src/diarization_streaming.cpp
        ${PARAKEET_DIR}/src/ced_tagger.cpp
        ${PARAKEET_DIR}/src/sound_stream.cpp
        ${PARAKEET_DIR}/src/speaker_registry.cpp
        ${PARAKEET_DIR}/src/speaker_identifier.cpp
        ${PARAKEET_DIR}/src/speaker_encoder.cpp
        ${PARAKEET_DIR}/src/scene_render.cpp
        ${PARAKEET_DIR}/src/silero_vad.cpp
        ${PARAKEET_DIR}/src/dr_wav_impl.cpp)
    add_library(parakeet STATIC ${PARAKEET_SRC})
    target_include_directories(parakeet PUBLIC
        "${PARAKEET_DIR}/include"
        PRIVATE
        "${PARAKEET_DIR}/src"
        "${PARAKEET_DIR}/third_party")
    target_compile_definitions(parakeet PRIVATE PARAKEET_VERSION="2eac2e7")
    if (NOT MSVC)
        set_source_files_properties(
            ${PARAKEET_DIR}/src/ternary.cpp
            ${PARAKEET_DIR}/src/ternary_kernels_x86.cpp
            ${PARAKEET_DIR}/src/ternary_kernels_neon.cpp
            PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    endif()
    target_link_libraries(parakeet PUBLIC ggml)
    target_compile_features(parakeet PUBLIC cxx_std_17)
    list(APPEND SYLLABIX_NATIVE_SRC src/shim_parakeet.cpp)
endif()

add_library(syllabix_native STATIC ${SYLLABIX_NATIVE_SRC})
# shim_vulkan.cpp catches ggml C++ exceptions raised under shim_tts.c / shim_asr.c frames;
# GCC/Clang only unwind C frames compiled with -fexceptions.
if(NOT MSVC)
    set_source_files_properties(
        src/shim_tts.c src/shim_asr.c src/shim_mio.c
        PROPERTIES COMPILE_OPTIONS "-fexceptions")
endif()
target_include_directories(syllabix_native PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${WHISPER_DIR}/include"
    "${LLAMA_DIR}/include"
    "${LLAMA_DIR}/tools/mtmd"
)
target_link_libraries(syllabix_native PUBLIC whisper llama llama-common mtmd)
if (SYLLABIX_PARAKEET)
    target_link_libraries(syllabix_native PUBLIC parakeet)
    target_include_directories(syllabix_native PRIVATE
        "${PARAKEET_DIR}/include"
        "${PARAKEET_DIR}/src")
endif()
target_compile_features(syllabix_native PUBLIC cxx_std_17)

include(GNUInstallDirs)
set(SYLLABIX_INSTALL_TARGETS syllabix_native whisper llama mtmd)
if (SYLLABIX_PARAKEET)
    list(APPEND SYLLABIX_INSTALL_TARGETS parakeet)
endif()
install(TARGETS ${SYLLABIX_INSTALL_TARGETS}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# ggml / ggml-base / ggml-cpu install themselves from the llama.cpp tree.

# crates.io: KleidiAI remains OFF above. Turning it on downloads kernel
# sources at configure time, which a packaged tarball cannot do.
