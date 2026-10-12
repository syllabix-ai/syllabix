# syllabix-core

Rust **voice agent** library: VAD, Whisper STT, LLM, and TTS in one
**speech-to-speech** conversation loop (shared types, provider contracts, and
the in-memory runtime).

```bash
cargo add syllabix-core
```

Or in `Cargo.toml`:

```toml
syllabix-core = "0.1"
```

API docs: [docs.rs/syllabix-core](https://docs.rs/syllabix-core).

Most apps use the defaults. They give you everything `syllabix run` uses
(local speech engines, mic and speakers, and a cloud LLM option). The first
build compiles the bundled whisper.cpp / llama.cpp with CMake and a C++
compiler. Nothing is downloaded at build time. Host packages:
[embed — Host compile](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#host-compile).

If your app brings its own speech-to-text, LLM, and text-to-speech, turn the
defaults off. Then you don't need CMake:

```toml
syllabix-core = { version = "0.1", default-features = false }
```

The other options are listed under
[Cargo features](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#cargo-features).

For unreleased changes on `main`, you can still depend on git:

```toml
[dependencies]
syllabix-core = { git = "https://github.com/syllabix-ai/syllabix.git" }
```

To use a released git tag instead of crates.io:

```toml
[dependencies]
syllabix-core = { git = "https://github.com/syllabix-ai/syllabix.git", tag = "v0.1.1" }
```

The `syllabix` CLI is not on crates.io. Install it from
[GitHub Releases](https://github.com/syllabix-ai/syllabix/releases) or
`cargo install --git`.

See [docs/embed.md](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md)
for lanes and the supported API. Release notes live on
[GitHub Releases](https://github.com/syllabix-ai/syllabix/releases).
See also the [repository README](https://github.com/syllabix-ai/syllabix#readme).
