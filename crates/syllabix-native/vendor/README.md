# Vendored ggml frontends

PR 10 compiles **one** `ggml` and links both speech frontends to it.

| Tree | Pin | Role |
|---|---|---|
| `llama.cpp` | `c4261a74a59c5a719e8f73038e28633b08f2bea4` (tag v0.9.0) | `ggml` engine + llama.cpp frontend + vendored `common/` (Jinja chat + tool-call parse) |
| `whisper.cpp` | `1fe009caeda75f69bc864d6370b10674e45a92bd` | Whisper STT frontend only |
| `parakeet.cpp` | `2eac2e77d9c7ef8899f7a86a46e693ac5ac1a5cd` | Fluid Speech Pro / Parakeet TDT STT frontend only; built against the llama.cpp `ggml` |

`ggml/include/ggml.h` is **not** byte-identical between the llama.cpp and whisper.cpp pins; whisper still links the llama vendored `ggml`. Syllabix **does not compile** whisper.cpp's `ggml/` (that directory is not vendored).

Only `src/`, `include/`, `third_party/dr_wav.h`, and `LICENSE` from parakeet.cpp are vendored — not its bundled `third_party/ggml`, examples, tests, or upstream `CMakeLists.txt`. The upstream `ggml-patches/0001-ggml-cpu-fold-broadcast-iterations-in-llamafile_sgem.patch` CPU speed tweak is intentionally **not** applied on Syllabix's single shared ggml (it may cost a little CPU time).

Sequence 26 vendors `ggml/src/ggml-metal` and `ggml/src/ggml-blas` from the same llama.cpp pin. Sequence 34 also vendors `ggml/src/ggml-vulkan` from that pin (shaders + shader-gen included). CMake only compiles a backend when `GGML_<NAME>` is ON:

- **Darwin:** Metal (`GGML_METAL_EMBED_LIBRARY`), Apple Accelerate BLAS, `GGML_ACCELERATE`. Darwin-arm64 also sets `GGML_NATIVE`. KleidiAI is **off**: SME objects need `___arm_tpidr2_save` and do not link from rustc.
- **Linux / Windows product default:** portable CPU. CUDA, HIP, Vulkan, SYCL, OpenCL, OpenMP, OpenBLAS stay off. Published Linux x64 must not gain `DT_NEEDED` on `libvulkan.so.1`.
- **Linux Vulkan opt-in:** set `SYLLABIX_GGML_VULKAN=1` so `build.rs` passes `-DGGML_VULKAN=ON` (needs `libvulkan-dev`, `glslc`/`shaderc`, and `spirv-headers`). Darwin and Windows ignore the env and stay off.

Do not add a second `ggml` copy. Do not use `--allow-multiple-definition` or post-build `objcopy` symbol renaming.

**Local patch (sequence 27), Darwin only:** `ggml/src/ggml-metal/ggml-metal-device.cpp` and `ggml-metal.cpp` leak their process-global Metal device caches on purpose (`static ... = new ...`, destructor never runs). Freeing them from C++ static destructors during `__cxa_finalize` aborted every ggml-Metal process at exit — `native_inference` passed all tests, then SIGABRT in `ggml_metal_rsets_free`. No runtime behavior change; keep the patch when bumping the pin.

**Local patch (adaptive Qwen placement), `tools/mtmd/clip.cpp`:** propagate compute-buffer reservation failures through `reserve_compute_meta`. Qwen's Apple Silicon auto mode relies on this recoverable error to destroy a failed Metal context and reload on CPU before the first turn. Keep the patch until upstream checks the scheduler return value.

**Local patch (Vulkan Qwen placement), `tools/mtmd/models/qwen3tts-gen.cpp`:**
the code predictor and code2wav `GET_ROWS` index views into `out_code_cache`
and `inp_codes` are wrapped in `ggml_cont`. Those views sit at 4-byte element
offsets, below the device `minStorageBufferOffsetAlignment`, and ggml-Vulkan can
fail on misaligned `GET_ROWS` indices during the first Qwen step. CPU and Metal
results are unchanged. Keep the patch until upstream ggml-Vulkan accepts
misaligned `GET_ROWS` indices.

**Local patch (common chat-only), `common/CMakeLists.txt`, `vendor/CMakeLists.txt`:** Syllabix links `llama-common` for Jinja chat and tool parsing only. Single `llama-common` static lib (no `llama-common-base`); no CLI/download sources or `http.h`; `vendor/cpp-httplib` not vendored; keep `hash`, `miniaudio`, `nlohmann`, `sheredom`, `stb`. No runtime behavior change.

CPU vs Metal tok/s for the three v0 GGUFs: [`llama-bench.md`](llama-bench.md).
