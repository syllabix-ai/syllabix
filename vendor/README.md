# Vendored ggml frontends

PR 10 compiles **one** `ggml` and links both speech frontends to it.

| Tree | Pin | Role |
|---|---|---|
| `llama.cpp` | `ece963f41b0b02d7a0d61436ae365762c073a4c8` | `ggml` engine + llama.cpp frontend |
| `whisper.cpp` | `1fe009caeda75f69bc864d6370b10674e45a92bd` | Whisper STT frontend only |

`ggml/include/ggml.h` is byte-identical at these two commits. Syllabix **does not compile** whisper.cpp's `ggml/` (that directory is not vendored). GPU/accelerator backend trees were omitted from llama.cpp's `ggml/src` because v0 is CPU-only; CMake only adds a backend when `GGML_<NAME>` is ON.

Do not add a second `ggml` copy. Do not use `--allow-multiple-definition` or post-build `objcopy` symbol renaming.
