# Syllabix

Local voice agent. One native binary. Apache-2.0.

v0 target: a stranger downloads `syllabix`, runs it, and is in a voice conversation on their laptop in under 3 minutes — no Python, pip, or API key.

This repository is the product. Founder docs live in [`syllabix-ai/syllabix_founder_documents`](https://github.com/syllabix-ai/syllabix_founder_documents). The launch contract is `V0_LAUNCH.md`.

```bash
cargo run -p syllabix -- --help
cargo run -p syllabix -- init     # optional syllabix.yaml
cargo run -p syllabix -- run      # zero-config mic + speakers with full-duplex AEC
```

## Intended 3-minute path (Release binary not shipping yet)

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
chmod +x syllabix
./syllabix run
```

GitHub Releases are not published yet. From this checkout, `cargo run -p syllabix -- run` talks on a machine with a microphone and speakers. First run fetches model weights into `$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1`. WebRTC AEC3 echo control is on by default and calibrates automatically from the samples sent to the speakers while the microphone stays open. If the terminal reports a lost speaker reference or the laptop still self-interrupts, use headphones and include the device names from the startup line in a bug report.

## CLI

| Command | Now | Launch |
| --- | --- | --- |
| `syllabix run` | Zero-config local mic/speaker conversation, full-duplex AEC, and TUI timings | Same; barge-in comes next |
| `syllabix init [dir]` | Optional `syllabix.yaml` scaffold | Same |

There is no `serve`, `bench`, cloud provider, or API key in v0. `run` does not require yaml. If `syllabix.yaml` is present, it must name the v0 on-device stack and `language: en`.

## Develop

Requires Rust 1.91+, CMake, and a C++ compiler. whisper.cpp and llama.cpp share one CPU `ggml` compiled into the binary. Linux contributors also need ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/ci-local.sh   # Linux stand-in for GitHub Actions
# `cargo llvm-cov --workspace --fail-under-lines 85` skips native inference (`cfg(coverage)`). Run `cargo test` for Whisper/Llama/Kokoro.
```

Model weights are not in git. A versioned manifest lists Silero, whisper.cpp `small`, llama-3.2-1b, and Kokoro. The cache writes into `$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1`, verifies SHA-256, and reuses files offline. `syllabix run` fills that cache on first launch.

CI records and plays a WAV fixture (no microphone). That path also soaks 30 minutes of *audio time* through bounded queues faster than real time. A six-turn native loop test runs Silero → whisper.cpp → llama.cpp → Kokoro through fixture capture/playback. The first STT/LLM/TTS/VAD test run fetches weights into the model cache.

On a laptop with a mic and speakers:

```bash
cargo run -p syllabix -- run
cargo test -p syllabix-core --test audio_io hardware_record_and_play_if_devices_exist -- --ignored --nocapture
```

PR 17's echo gate needs a quiet laptop with its built-in microphone and speakers selected. Do not wear headphones or speak during this command. It allows 10 seconds for automatic calibration, then plays the versioned speech fixture continuously for 1 minute and requires Silero to detect zero false user turns:

```bash
cargo test -p syllabix-core --test audio_io hardware_aec_1_minute_playback_has_zero_false_turns -- --ignored --exact --nocapture
```

The microphone remains open throughout the test. Muting capture during playback does not pass this gate.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
