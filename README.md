# Syllabix

Local voice agent. One native binary. Apache-2.0.

v0 target: a stranger downloads `syllabix`, runs it, and is in a voice conversation on their laptop in under 3 minutes — no Python, pip, or API key.

This repository is the product. Founder docs live in [`syllabix-ai/syllabix_founder_documents`](https://github.com/syllabix-ai/syllabix_founder_documents). The launch contract is `V0_LAUNCH.md`.

```bash
cargo run -p syllabix -- --help
cargo run -p syllabix -- init     # optional syllabix.yaml
cargo run -p syllabix -- run      # zero-config mic + speakers with full-duplex AEC
cargo run -p syllabix -- run --turn-debug   # optional per-turn WAVs + sidecar
cargo run -p syllabix -- run --barge-in     # interrupt TTS when the user speaks
```

## Intended 3-minute path (Release publication is sequence 23)

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
chmod +x syllabix
./syllabix run
```

GitHub Releases are not published yet (sequence 23). Sequence 22 produces the installable files:

| Target | Artifact |
| --- | --- |
| Linux x64 | `syllabix-Linux-x86_64` |
| macOS Apple Silicon | `syllabix-Darwin-arm64` |
| macOS Intel | `syllabix-Darwin-x86_64` |
| Windows x64 | `syllabix-Windows-x86_64.exe` |

```bash
./scripts/package-release.sh                         # host triple
./scripts/package-release.sh x86_64-unknown-linux-gnu
./scripts/check-clean-artifact.sh dist/syllabix-Linux-x86_64
./scripts/check-repro.sh                             # two isolated dist builds
```

`cargo build -p syllabix --profile dist` is thin-LTO, one codegen unit, debuginfo stripped. Default `release` is unchanged for `ci-local.sh`. The executable does **not** pack Silero / Whisper / Llama / Kokoro weights; first `run` fetches them into `$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1` (`%LOCALAPPDATA%\syllabix\cache\models\v1` on Windows), verifies SHA-256, and reuses the cache offline. `--help` / `init` do not need the cache.

From this checkout, `cargo run -p syllabix -- run` talks on a machine with a microphone and speakers. WebRTC AEC3 echo control is on by default and calibrates automatically from the samples sent to the speakers while the microphone stays open. If the terminal reports a lost speaker reference or the laptop still self-interrupts, use headphones and include the device names from the startup line in a bug report.

## CLI

| Command | Now | Launch |
| --- | --- | --- |
| `syllabix run` | Zero-config local mic/speaker conversation, full-duplex AEC, and TUI timings | Same |
| `syllabix run --turn-debug [dir]` | Opt-in. Writes `capture.wav` / `clean.wav` / `utterance.wav` / `tts.wav` and `turn.json` per turn under `dir`, `$SYLLABIX_TURN_DEBUG_DIR`, or `target/turn-debug`. Default `run` writes nothing. | Same |
| `syllabix run --barge-in` | Opt-in. VAD keeps running during TTS; user SpeechStart stops playback, flushes queued audio, and cancels LLM/TTS. Off by default. Combine with `--turn-debug` to dump interrupted turns. Whisper utterances always include 200 ms of post-AEC preroll. | Same |
| `syllabix init [dir]` | Optional `syllabix.yaml` scaffold | Same |

There is no `serve`, `bench`, cloud provider, or API key in v0. `run` does not require yaml. If `syllabix.yaml` is present, it must name the v0 on-device stack and `language: en`.

## Develop

Requires Rust 1.91+, CMake, and a C++ compiler. whisper.cpp and llama.cpp share one CPU `ggml` compiled into the binary. Linux contributors also need ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/ci-local.sh   # Linux stand-in for GitHub Actions (includes dist package + clean artifact)
./scripts/check-repro.sh  # two dist builds; matching *.repro.json including sha256
# `cargo llvm-cov --workspace --fail-under-lines 85` skips native inference (`cfg(coverage)`). Run `cargo test` for Whisper/Llama/Kokoro.
```

Model weights are not in git and are not packed into the `dist` executable. A versioned manifest lists Silero, whisper.cpp `small`, llama-3.2-1b, and Kokoro. The cache writes into `$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1`, verifies SHA-256, and reuses files offline. `syllabix run` fills that cache on first launch.

CI records and plays a WAV fixture (no microphone). That path also soaks 30 minutes of *audio time* through bounded queues faster than real time. A six-turn native loop test runs Silero → whisper.cpp → llama.cpp → Kokoro through fixture capture/playback. The first STT/LLM/TTS/VAD test run fetches weights into the model cache.

On a laptop with a mic and speakers:

```bash
cargo run -p syllabix -- run
cargo test -p syllabix-core --test audio_io hardware_record_and_play_if_devices_exist -- --ignored --nocapture
```

PR 18's echo gate needs a quiet laptop with its built-in microphone and speakers selected. Do not wear headphones or speak during this command. It allows 10 seconds for automatic calibration, then plays the versioned speech fixture continuously for 1 minute and requires Silero to detect zero false user turns:

```bash
cargo test -p syllabix-core --test audio_io hardware_aec_1_minute_playback_has_zero_false_turns -- --ignored --exact --nocapture
```

The microphone remains open throughout the test. Muting capture during playback does not pass this gate. The test writes `render.wav`, `capture.wav`, `clean.wav`, and `sidecar.json` to `$SYLLABIX_AEC_DEBUG_DIR` or `target/aec-debug`.

To dump a live conversation for diagnosis (listen to `utterance.wav` against STT text and `tts.wav` against the LLM reply):

```bash
cargo run -p syllabix --release -- run --turn-debug target/turn-debug
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
