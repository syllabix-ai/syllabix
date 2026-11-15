# syllabix-stt

Speech-to-text for Rust: one `Stt` trait over local engines and an
OpenAI-compatible remote client. It is the speech-to-text part of
[Syllabix](https://github.com/syllabix-ai/syllabix), without the voice agent.

Pick engines with Cargo features. Defaults are `moonshine` and `remote` (ONNX
plus HTTP). `whisper`, `qwen3-asr`, `parakeet`, and `hima` are opt-in. With only `remote`
the crate builds without CMake, a C++ compiler, or ONNX Runtime.

Default ONNX engines use the `ort` crate, which may download a prebuilt ONNX
Runtime at build time; some targets have no binary. For HTTP only, use
`default-features = false` with the `remote` feature.

| Feature | Engine | Needs C++? | Model ids |
| --- | --- | --- | --- |
| `moonshine` | Moonshine streaming (ONNX) | No (ONNX Runtime) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
| `remote` | OpenAI-compatible `/audio/transcriptions` | No (HTTP only) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
| `whisper` | whisper.cpp | Yes (CMake) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
| `qwen3-asr` | Qwen3-ASR 0.6B | Yes (CMake) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
| `parakeet` | Fluid Speech Pro GGUF | Yes (CMake) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
| `hima` | Hima-ASR (ONNX) | No (ONNX Runtime) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |

C++ engines need the host packages in
[embed — Host compile](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#host-compile).

## Cache

By default, weights land under `models/v1` at `$SYLLABIX_CACHE_DIR`, else
`$XDG_CACHE_HOME/syllabix`, else `~/.cache/syllabix` (on Windows,
`%LOCALAPPDATA%\syllabix\cache` when those are unset). The first use of a
model id downloads only that asset over HTTPS, checks SHA-256, then reuses the
file; sharing the same root with the Syllabix CLI is fine. Point a custom root
with `ModelCache::new` and `Manifest` on this crate. Model ids and licenses:
[engines](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md).
This crate is Apache-2.0; third-party notices are in [NOTICE](NOTICE).

The remote engine reads its key from `SYLLABIX_STT_API_KEY` only.

Examples: [Moonshine local WAV](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-stt/examples/transcribe-wav-local.rs)
and [remote HTTP](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-stt/examples/transcribe-wav.rs).
API docs: [docs.rs/syllabix-stt](https://docs.rs/syllabix-stt).
For the full voice agent, use
[syllabix-core](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md).
