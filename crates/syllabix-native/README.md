# syllabix-native

Shared ggml plus the whisper.cpp and llama.cpp frontends used by
`syllabix-core`. This crate is not a public API: depend on `syllabix-core`.

Engine source lives in `vendor/` so a crates.io package can build without
downloading code. KleidiAI stays off because it fetches extra sources at
configure time.

Licenses: Apache-2.0 for Syllabix (`LICENSE`), MIT for llama.cpp and
whisper.cpp (`vendor/llama.cpp/LICENSE`, `vendor/whisper.cpp/LICENSE`).
See `NOTICE`.
