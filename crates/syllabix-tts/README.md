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

## Cache

By default, weights land under `models/v1` at `$SYLLABIX_CACHE_DIR`, else
`$XDG_CACHE_HOME/syllabix`, else `~/.cache/syllabix` (on Windows,
`%LOCALAPPDATA%\syllabix\cache` when those are unset). The first use of a
model id downloads only that asset over HTTPS, checks SHA-256, then reuses the
file; sharing the same root with the Syllabix CLI is fine. Point a custom root
with `ModelCache::new` and `Manifest` on this crate. Model ids and licenses:
[engines](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md).
This crate is Apache-2.0; third-party notices are in [NOTICE](NOTICE).

The remote engine reads its key from `SYLLABIX_TTS_API_KEY` only.

Examples: [Pocket local WAV](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-tts/examples/speak-wav-local.rs)
and [remote HTTP](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-tts/examples/speak-wav.rs).
API docs: [docs.rs/syllabix-tts](https://docs.rs/syllabix-tts).
For the full voice agent, use
[syllabix-core](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md).
