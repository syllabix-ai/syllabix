# syllabix-core

Rust **voice agent** library: VAD, Whisper STT, LLM, and TTS in one
**speech-to-speech** conversation loop (shared types, provider contracts, and
the in-memory runtime).

This crate is not on crates.io yet (`publish = false`). Other repos take a git
dependency until a later publish. Pin a tag: the SDK (`syllabix-core`) and
the CLI (`syllabix`) share one version and ship on the same `vX.Y.Z` tag.
Do not track `main`.

```toml
syllabix-core = { git = "https://github.com/syllabix-ai/syllabix.git", tag = "v0.1.0" }
```

See [docs/embed.md](../../docs/embed.md) and [CHANGELOG.md](../../CHANGELOG.md).
See also the [repository README](https://github.com/syllabix-ai/syllabix#readme).
