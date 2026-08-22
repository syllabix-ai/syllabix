# Vendored ggml frontends

PR 10 compiles **one** `ggml` and links both speech frontends to it.

| Tree | Pin | Role |
|---|---|---|
| `llama.cpp` | `ece963f41b0b02d7a0d61436ae365762c073a4c8` | `ggml` engine + llama.cpp frontend |
| `whisper.cpp` | `1fe009caeda75f69bc864d6370b10674e45a92bd` | Whisper STT frontend only |

`ggml/include/ggml.h` is byte-identical at these two commits. Syllabix **does not compile** whisper.cpp's `ggml/` (that directory is not vendored).

Sequence 26 vendors `ggml/src/ggml-metal` and `ggml/src/ggml-blas` from the same llama.cpp pin. CMake only compiles a backend when `GGML_<NAME>` is ON:

- **Darwin:** Metal (`GGML_METAL_EMBED_LIBRARY`), Apple Accelerate BLAS, `GGML_ACCELERATE`. Darwin-arm64 also sets `GGML_NATIVE`. KleidiAI is **off**: SME objects need `___arm_tpidr2_save` and do not link from rustc.
- **Linux / Windows:** portable CPU. CUDA, HIP, Vulkan, SYCL, OpenCL, OpenMP, OpenBLAS stay off.

Do not add a second `ggml` copy. Do not use `--allow-multiple-definition` or post-build `objcopy` symbol renaming.

**Local patch (sequence 27), Darwin only:** `ggml/src/ggml-metal/ggml-metal-device.cpp` and `ggml-metal.cpp` leak their process-global Metal device caches on purpose (`static ... = new ...`, destructor never runs). Freeing them from C++ static destructors during `__cxa_finalize` aborted every ggml-Metal process at exit — `native_inference` passed all 16 tests, then SIGABRT in `ggml_metal_rsets_free`. No runtime behavior change; keep the patch when bumping the pin.

**Local patch (sequence 32), `tools/mtmd/clip.cpp`:** `PROJECTOR_TYPE_QWEN3TTS_GEN` loads `a.gen.code.proj_in.{weight,bias}` as optional (`get_tensor(..., /*required=*/false)`). The gen graph already null-guards both tensors (`code_gen::project_in` passes through untouched when absent); the row-32 0.6B backbone conversion omits them (talker hidden size == predictor hidden size there), and the required lookup rejected every 0.6B mmproj at load. No effect on the 1.7B files, which carry both tensors; keep the patch when bumping the pin, drop once upstream makes these optional.

CPU vs Metal tok/s for the three v0 GGUFs: [`llama-bench.md`](llama-bench.md).
