# syllabix-tts

Text-to-speech for Rust: one `Tts` trait over local engines and an
OpenAI-compatible remote client. It is the text-to-speech part of
[Syllabix](https://github.com/syllabix-ai/syllabix), without the voice agent.

Pick engines with Cargo features. Defaults are `pocket`, `kokoro`, and `remote`
(ONNX plus HTTP). `qwen3` is opt-in. With only `remote` the crate builds without
CMake, a C++ compiler, or ONNX Runtime.

Default ONNX engines use the `ort` crate, which may download a prebuilt ONNX
Runtime at build time; some targets have no binary. For HTTP only, use
`default-features = false` with the `remote` feature.

| Feature | Engine | Needs C++? | Model ids |
| --- | --- | --- | --- |
| `pocket` | Pocket TTS (ONNX) | No (ONNX Runtime) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
| `kokoro` | Kokoro-82M (ONNX) | No (ONNX Runtime) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
| `remote` | OpenAI-compatible `/audio/speech` | No (HTTP only) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
| `qwen3` | Qwen3-TTS; Indic-Mio currently needs `qwen3` + `kokoro` (`ort`) | Yes (CMake) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |

C++ engines need the host packages in
[embed — Host compile](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#host-compile).
Local engines download their weights on first use
([embed — Cache](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#cache)).
The remote engine reads its key from `SYLLABIX_TTS_API_KEY` only.

Example: [speak text into a WAV file](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-tts/examples/speak-wav.rs)
with the remote engine. API docs: [docs.rs/syllabix-tts](https://docs.rs/syllabix-tts).
For the full voice agent, use
[syllabix-core](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md).
