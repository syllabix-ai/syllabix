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
Local engines download their weights on first use
([embed — Cache](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#cache)).
The remote engine reads its key from `SYLLABIX_STT_API_KEY` only.

Example: [transcribe a WAV file](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-stt/examples/transcribe-wav.rs)
with the remote engine. API docs: [docs.rs/syllabix-stt](https://docs.rs/syllabix-stt).
For the full voice agent, use
[syllabix-core](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md).
