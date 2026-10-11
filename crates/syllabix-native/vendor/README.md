# Vendored ggml frontends

PR 10 compiles **one** `ggml` and links both speech frontends to it.

| Tree | Pin | Role |
|---|---|---|
| `llama.cpp` | `ece963f41b0b02d7a0d61436ae365762c073a4c8` | `ggml` engine + llama.cpp frontend |
| `whisper.cpp` | `1fe009caeda75f69bc864d6370b10674e45a92bd` | Whisper STT frontend only |

`ggml/include/ggml.h` is byte-identical at these two commits. Syllabix **does not compile** whisper.cpp's `ggml/` (that directory is not vendored).

Sequence 26 vendors `ggml/src/ggml-metal` and `ggml/src/ggml-blas` from the same llama.cpp pin. Sequence 34 also vendors `ggml/src/ggml-vulkan` from that pin (shaders + shader-gen included). CMake only compiles a backend when `GGML_<NAME>` is ON:

- **Darwin:** Metal (`GGML_METAL_EMBED_LIBRARY`), Apple Accelerate BLAS, `GGML_ACCELERATE`. Darwin-arm64 also sets `GGML_NATIVE`. KleidiAI is **off**: SME objects need `___arm_tpidr2_save` and do not link from rustc.
- **Linux / Windows product default:** portable CPU. CUDA, HIP, Vulkan, SYCL, OpenCL, OpenMP, OpenBLAS stay off. Published Linux x64 must not gain `DT_NEEDED` on `libvulkan.so.1`.
- **Linux Vulkan opt-in:** set `SYLLABIX_GGML_VULKAN=1` so `build.rs` passes `-DGGML_VULKAN=ON` (needs `libvulkan-dev`, `glslc`/`shaderc`, and `spirv-headers`). Darwin and Windows ignore the env and stay off.

Do not add a second `ggml` copy. Do not use `--allow-multiple-definition` or post-build `objcopy` symbol renaming.

**Local patch (sequence 27), Darwin only:** `ggml/src/ggml-metal/ggml-metal-device.cpp` and `ggml-metal.cpp` leak their process-global Metal device caches on purpose (`static ... = new ...`, destructor never runs). Freeing them from C++ static destructors during `__cxa_finalize` aborted every ggml-Metal process at exit — `native_inference` passed all 16 tests, then SIGABRT in `ggml_metal_rsets_free`. No runtime behavior change; keep the patch when bumping the pin.

**Local patch (sequence 32), `tools/mtmd/clip.cpp`:** `PROJECTOR_TYPE_QWEN3TTS_GEN` loads `a.gen.code.proj_in.{weight,bias}` as optional (`get_tensor(..., /*required=*/false)`). The gen graph already null-guards both tensors (`code_gen::project_in` passes through untouched when absent); the row-32 0.6B backbone conversion omits them (talker hidden size == predictor hidden size there), and the required lookup rejected every 0.6B mmproj at load. No effect on the 1.7B files, which carry both tensors; keep the patch when bumping the pin, drop once upstream makes these optional.

**Local patch (sequence 33), `tools/mtmd/mtmd-helper-gen.{h,cpp}`:** expose `mtmd_helper_gen_audio_take_output`, which returns only PCM newly produced by an already-completed vocoder window without flushing an incomplete window. The Qwen shim forwards those windows to Rust as generation continues, rather than waiting for `get_output()` to flush and return the entire utterance. Decoder state and the existing window sizes remain upstream-owned. Contribute this narrow helper API upstream; retain the patch only until it merges.

**Local patch (adaptive Qwen placement), `tools/mtmd/clip.cpp`:** propagate
compute-buffer reservation and graph-allocation failures through
`clip_encode`. Qwen's Apple Silicon auto mode relies on this recoverable error
to destroy a failed Metal context and reload on CPU before the first turn.
Keep the patch until upstream checks both scheduler return values.

**Local patch (Vulkan Qwen placement), `tools/mtmd/models/qwen3tts-gen.cpp`:**
the code predictor and code2wav `GET_ROWS` index views into `out_code_cache`
and `inp_codes` are wrapped in `ggml_cont`. Those views sit at 4-byte element
offsets, below the device `minStorageBufferOffsetAlignment` (16 on NVIDIA L4),
and ggml-Vulkan aborts on
`GGML_ASSERT(dst->op != GGML_OP_GET_ROWS || (a_offset == 0 && b_offset == 0 && d_offset == 0))`
during the first Qwen step. The copy is one I32 row per lookup; CPU and Metal
results are unchanged. Keep the patch until upstream ggml-Vulkan accepts
misaligned `GET_ROWS` indices.

CPU vs Metal tok/s for the three v0 GGUFs: [`llama-bench.md`](llama-bench.md).
