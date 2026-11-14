# syllabix-native

Shared Syllabix types (errors, audio frames, model cache, ONNX helpers) used by
`syllabix-core` and the upcoming speech crates. The C++ engines (ggml,
whisper.cpp, llama.cpp) sit behind the `cpp` feature. This crate is not a
public API: depend on `syllabix-core`.

Engine source lives in `vendor/` so a crates.io package can build without
downloading code. KleidiAI stays off because it fetches extra sources at
configure time.

Licenses: Apache-2.0 for Syllabix (`LICENSE`), MIT for llama.cpp and
whisper.cpp (`vendor/llama.cpp/LICENSE`, `vendor/whisper.cpp/LICENSE`).
See `NOTICE`.
