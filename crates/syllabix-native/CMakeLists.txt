cmake_minimum_required(VERSION 3.14)
project(syllabix-native C CXX)

# One static ggml. Both frontends link it. No shared libs in the product binary.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "static ggml / whisper / llama" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_HIP OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_SYCL OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
set(GGML_HEXAGON OFF CACHE BOOL "" FORCE)
set(GGML_CANN OFF CACHE BOOL "" FORCE)
set(GGML_MUSA OFF CACHE BOOL "" FORCE)
set(GGML_WEBGPU OFF CACHE BOOL "" FORCE)
set(GGML_ZDNN OFF CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)

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
