# Contributing

Product code lives in this repository. Agent process, launch sequence, and learnings live in [`syllabix-ai/syllabix_founder_documents`](https://github.com/syllabix-ai/syllabix_founder_documents).

## Build

Requires Rust 1.91+, CMake, and a C++ compiler. whisper.cpp and llama.cpp share one `ggml` compiled into the binary (Darwin Metal + Accelerate; Linux/Windows portable CPU). Linux also needs ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace   # launch-stack native inference (small / llama-3.2-1b / kokoro)
./scripts/ci-local.sh    # Linux stand-in for GitHub Actions (fmt through tests, then Linux dist + smoke)
./scripts/check-repro.sh # two dist builds; run when the dist profile or packaging script changes
```

`cargo llvm-cov --workspace --fail-under-lines 85` skips native inference (`cfg(coverage)`).

Exclusive extra native suites (not the default `cargo test --workspace` bar):

```bash
SYLLABIX_NATIVE_MODELS=qwen3-0.6 cargo test -p syllabix-core --test native_inference
```

Model weights are not in git and are not packed into the `dist` executable. A versioned manifest lists Silero, the whisper.cpp STT menu, Llama 3.2 1B (default), Qwen3.5-0.8B, Qwen3.5-2B, Kokoro, and the Qwen3-TTS backbones. Zero-config `run` fetches only the selected ids. The cache writes into `$SYLLABIX_CACHE_DIR/models/v1` when the env var is set, otherwise `~/.cache/syllabix/models/v1`, verifies SHA-256, and reuses files offline.

CPU vs Metal tok/s for the three default GGUFs: [`vendor/llama-bench.md`](../vendor/llama-bench.md). ggml vendor pins and local patches: [`vendor/README.md`](../vendor/README.md). Measurement protocols: [`reference-profiles.md`](reference-profiles.md).

## Live devices

From this checkout, `cargo run -p syllabix -- run` talks on a machine with a microphone and speakers.

```bash
cargo run -p syllabix -- run
cargo test -p syllabix-core --test audio_io hardware_record_and_play_if_devices_exist -- --ignored --nocapture
```

The echo gate needs a quiet laptop with its built-in microphone and speakers selected. Do not wear headphones or speak during this command. It allows 10 seconds for automatic calibration, then plays the versioned speech fixture continuously for 1 minute and requires Silero to detect zero false user turns:

```bash
cargo test -p syllabix-core --test audio_io hardware_aec_1_minute_playback_has_zero_false_turns -- --ignored --exact --nocapture
```

The microphone remains open throughout. Muting capture during playback does not pass. Debug dumps: `$SYLLABIX_AEC_DEBUG_DIR` or `target/aec-debug`.

To dump a live conversation, enable diagnostics in `syllabix.yaml` and run a release binary — see [diagnostics.md](diagnostics.md):

```yaml
diagnostics:
  timestamps: true
  audio: true
  directory: target/turn-debug
```

```bash
cargo run -p syllabix --release -- run
python3 scripts/summarize-timelines.py target/turn-debug
```

## Packaging (maintainers)

GitHub Release files are the four artifact names plus `SHA256SUMS`. `.github/workflows/release.yml` publishes them on `v*` tags. Pull requests do not package dist binaries. If Actions cannot run, build each target with `scripts/package-release.sh` and attach with `scripts/publish-release.sh v0.1.0`.

```bash
./scripts/package-release.sh                         # host triple
./scripts/package-release.sh x86_64-unknown-linux-gnu
./scripts/check-clean-artifact.sh dist/syllabix-Linux-x86_64
./scripts/smoke-offline-setup.sh dist/syllabix-Linux-x86_64
./scripts/check-repro.sh
./scripts/publish-release.sh v0.1.0
```

`cargo build -p syllabix --profile dist` is thin-LTO, one codegen unit, debuginfo stripped. Default `release` is unchanged for `ci-local.sh`.

After a Release is published, validate the documented download path: `SMOKE_RELEASE_URL=https://github.com/syllabix-ai/syllabix/releases/download/<tag> scripts/smoke-setup.sh syllabix-Linux-x86_64` (repeat per OS you can touch).

Linux contributors who hit a missing ALSA link need `libasound2-dev`. Users of the Release binary never compile anything.

## CI

Per-PR CI is fmt, clippy, and llvm-cov ≥85% on the self-hosted macOS runner (no native weight load). Weekly CI runs `cargo test --workspace` on Linux / Windows / macOS, including launch-stack native inference.
