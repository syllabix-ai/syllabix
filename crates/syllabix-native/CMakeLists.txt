cmake_minimum_required(VERSION 3.14)
project(syllabix-native C CXX)

# One static ggml. Both frontends link it. No shared libs in the product binary.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "static ggml / whisper / llama" FORCE)

# Sequence 26: Darwin uses OS-shipped Metal + Accelerate. Linux/Windows stay
# portable CPU. CUDA/HIP/Vulkan/OpenMP/OpenBLAS stay off everywhere.
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
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

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_SYSTEM_GGML OFF CACHE BOOL "" FORCE)
set(LLAMA_ALL_WARNINGS OFF CACHE BOOL "" FORCE)
set(LLAMA_FATAL_WARNINGS OFF CACHE BOOL "" FORCE)

set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/llama.cpp")
set(WHISPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/whisper.cpp")

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

add_library(syllabix_native STATIC src/shim.c)
target_include_directories(syllabix_native PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${WHISPER_DIR}/include"
    "${LLAMA_DIR}/include"
)
target_link_libraries(syllabix_native PUBLIC whisper llama)

include(GNUInstallDirs)
install(TARGETS syllabix_native whisper llama
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# ggml / ggml-base / ggml-cpu install themselves from the llama.cpp tree.
