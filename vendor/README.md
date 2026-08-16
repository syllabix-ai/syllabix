# Vendored native crates

`llama-cpp-sys-2` is vendored from crates.io 0.1.154 with one change: after
CMake builds the static archives, `build.rs` extracts every object and
renames ggml/gguf/quantize symbols (`llama_iso_*`), including undefined
relocations. whisper.cpp ships its own ggml. One `syllabix` binary needs
both, and `--allow-multiple-definition` makes llama.cpp call whisper's
older ggml and abort. Bindgen FFI names are retargeted with `#[link_name]`.

Bump the crate by replacing this directory and re-applying the isolate step
in `build.rs`.
