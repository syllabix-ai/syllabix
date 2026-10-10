# syllabix

Local-first **voice agent** CLI for on-device **speech-to-speech** conversation
(Whisper STT, VAD, TTS). Download one binary and talk; no Python or API key
for the default setup.

This crate is not on crates.io yet (`publish = false`). Prefer a
[GitHub Release](https://github.com/syllabix-ai/syllabix/releases) binary. To
compile the CLI from a git tag:

```bash
cargo install --git https://github.com/syllabix-ai/syllabix.git --tag v0.1.0 syllabix
```

The CLI and the SDK (`syllabix-core`) share that version. Do not track `main`.
Host compile packages: [install](../../docs/install.md#compile).

See the [repository README](https://github.com/syllabix-ai/syllabix#readme).
