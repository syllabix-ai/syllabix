# syllabix-core

Rust **voice agent** library: VAD, Whisper STT, LLM, and TTS in one
**speech-to-speech** conversation loop (shared types, provider contracts, and
the in-memory runtime).

This crate is not on crates.io yet (`publish = false`). Other repos take a git
dependency until a later publish. Pin a tag: the SDK (`syllabix-core`) and
the CLI (`syllabix`) share one version and ship on the same `vX.Y.Z` tag.
Do not track `main`.

Most apps use the defaults. They give you everything `syllabix run` uses
(local speech engines, mic and speakers, and a cloud LLM option). Building
them needs CMake and a C++ compiler.

```toml
syllabix-core = { git = "https://github.com/syllabix-ai/syllabix.git", tag = "v0.1.0" }
```

If your app brings its own speech-to-text, LLM, and text-to-speech, turn the
defaults off. Then you don't need CMake:

```toml
syllabix-core = { git = "https://github.com/syllabix-ai/syllabix.git", tag = "v0.1.0", default-features = false }
```

The other options are listed under
[Cargo features](../../docs/embed.md#cargo-features).

See [docs/embed.md](../../docs/embed.md). Release notes are on the git tag
and its GitHub Release.
See also the [repository README](https://github.com/syllabix-ai/syllabix#readme).
