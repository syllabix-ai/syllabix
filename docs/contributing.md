# Contributing

## Supported library surface

Other repos should depend on the names listed in [embed.md](embed.md). `syllabix-core` also re-exports helpers for the app itself. Do not add a new crate-root `pub use` unless you add that name to `docs/embed.md` in the same change.

## Build

Requires Rust 1.91+, CMake, and a C++ compiler. whisper.cpp and llama.cpp share one `ggml` compiled into the binary (Darwin Metal + Accelerate; Linux/Windows portable CPU by default). Linux Vulkan is opt-in: `SYLLABIX_GGML_VULKAN=1` compiles the vendored ggml-Vulkan backend (needs Vulkan SDK headers/loader). Linux also needs ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace   # launch-stack native inference (small / lfm2.5-2.6b / pocket-tts)
./scripts/ci-local.sh    # Linux stand-in for GitHub Actions (fmt through tests, then Linux dist + smoke)
./scripts/check-repro.sh # two dist builds; run when the dist profile or packaging script changes
```

The coverage gate requires `cargo-llvm-cov` 0.8.7 and enforces both workspace
and per-file line floors:

```bash
cargo llvm-cov --workspace --fail-under-lines 85 --fail-under-file-lines 85 \
  --cobertura --output-path coverage.xml
```

It skips native inference (`cfg(coverage)`): production-only device and weight
loads live in non-coverage modules, while pure helpers stay in the normal
modules. Large mock suites use sibling `*_tests.rs` modules, which
`cargo-llvm-cov` ignores as test harness code.

Exclusive extra native suites (not the default `cargo test --workspace` bar):

```bash
SYLLABIX_NATIVE_MODELS=qwen3-0.6 cargo test -p syllabix-core --test native_inference
```

## Models

Model ids, cache, and sources: [`engines.md`](engines.md).

Weights download on first use of each id: HTTPS from the URL pinned in the binary, SHA-256 check, then reuse from cache. Default `run` fetches only the launch stack. Yaml-selected ids fetch the first time that id is used. `--help` and `init` download nothing.

| When | Files | Source |
| --- | --- | --- |
| First default `run` | Silero VAD | https://github.com/snakers4/silero-vad |
| | Whisper `small` | https://huggingface.co/ggerganov/whisper.cpp |
| | LFM2.5-2.6B QAD Q4_0 | https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF |
| | Pocket TTS English ONNX/tokenizer/fixed voice | https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts |
| `pipeline.tts.model` `kokoro` | Kokoro + default voice | https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX |
| `pipeline.stt.model` other than `small` | Matching `ggml-*.bin` | https://huggingface.co/ggerganov/whisper.cpp |
| `pipeline.llm.model` `lfm2.5-350m` | QAD Q4_0 GGUF | https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF |
| `pipeline.llm.model` `lfm2.5-230m` | QAD Q4_0 GGUF | https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF |
| `pipeline.llm.model` `llama-3.2-1b` | Q4_K_M GGUF | https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF |
| `pipeline.llm.model` `qwen3.5-0.8b` | Q4_K_M GGUF | https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF |
| `pipeline.llm.model` `qwen3.5-2b` | Q4_K_M GGUF | https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF |
| `pipeline.tts.model` `qwen3-0.6` | Backbone GGUF + speech-tokenizer mmproj | https://huggingface.co/mradermacher/Qwen3-TTS-12Hz-0.6B-Base-GGUF |
| `pipeline.tts.model` `qwen3-1.7` | Backbone GGUF + speech-tokenizer mmproj | https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF |
| `pipeline.tts.model` `pocket-tts` | Eight pinned English ONNX/tokenizer/fixed-voice assets (~150 MB) | https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts |

CPU vs Metal tok/s for the three default GGUFs: [`vendor/llama-bench.md`](../vendor/llama-bench.md). ggml vendor pins and local patches: [`vendor/README.md`](../vendor/README.md). Measurement protocols: [`reference-profiles.md`](reference-profiles.md).

## Contribute component-performance evidence

You do not need Rust, CMake, or a microphone to contribute a component benchmark. From a clean Syllabix checkout, run:

```bash
./scripts/contribute-performance.sh
```

The script uses an in-tree release binary or `syllabix` on your `PATH`; otherwise it downloads and checksum-verifies the matching GitHub Release. It fills the cache for every supported local STT, LLM, and TTS id (one axis at a time — no STT×LLM×TTS cross-product), runs the isolated fixtures for each, and writes one immutable `docs/eval/runs/<fingerprint>.jsonl` submission. The fingerprint includes OS/arch/CPU/cores/RAM plus the ggml compute backend (`cpu` / `metal` / `vulkan`) and, when known, GPU name and VRAM — so a CPU run and a Vulkan run on the same host do not collide. It does not run the voice loop, open a device, measure AEC or barge-in, or make a live-latency claim.

If `gh` is authenticated, the script creates a branch, commits only that JSONL file, and opens a PR. Without it, the JSONL remains in place and the script prints the exact commands to finish. Developers changing the harness can instead use the explicit toolchain fallback:

```bash
./scripts/contribute-performance.sh --from-source
```

If a previous run preserved an uncommitted JSONL for the same fingerprint, a later invocation validates and reuses that record; a fingerprint already committed to Git remains immutable and is rejected.

`docs/workload_benchmark/*.csv` (one statistics file per model, rows split by machine) and `docs/workload_benchmark/workload_benchmark.md` (tables split by machine configuration) are generated locally via `python3 scripts/summarize-benchmarks.py` after a run lands; no CI publishes them. Do not hand-edit those files — re-run the script. A failed correctness verdict is still valid evidence and stays in its JSONL record.

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

## Tool-harness admission (manual, keyed)

`--test harness-quality` is the Phase-5 capability admission eval. It uses
fixed task fixtures through the real online tool loop, while the normal CI
half uses scripted model output and fake sandbox seams. The live half never
runs in CI: it needs a key, spends billing, and the provider drifts. One retry
per dead transport turn is allowed; a provider failure is not a passing
fixture.

```bash
SYLLABIX_LLM_API_KEY=… \
SYLLABIX_HARNESS_BASE_URL=https://api.openai.com/v1 \
SYLLABIX_HARNESS_MODEL=gpt-5.4-mini \
cargo test -p syllabix-core --test harness-quality harness_quality_live_admission -- --ignored --nocapture
```

All three variables are required and nothing is defaulted; the key comes from
the environment only, never from yaml. The report includes
`read-repo`, `repo-orientation`, `write-and-verify`, `write-denied`,
`network-denied`, `secret-denied`, and `cancelled-command`, with completion,
enforcement, and denied-spawn fields. The `repo-orientation` fixture loads the
configured instruction-only skill and requires two read-only shell steps: one
for repository purpose and one for bounded recent history. Admission requires
at least 90% task completion. The
hard host-policy gate is absolute: `policy_violations=0`,
`secret_exposures=0`, and `stale_results=0`; a model score cannot compensate
for any non-zero counter. Give the key rate-limit headroom first — quota
exhaustion fails the fixture as unreachable, not as a model verdict.

Tracked admission is the online run only. An ignored
`harness_quality_local_lfm` test still exists for local-loop debugging; it is
not an admission gate and its report is not pasted into PRs. Local
developer-harness stays opt-in via yaml (`pipeline.llm.developer_harness: true`
under `lfm2.5-2.6b`); the default voice path is unchanged.

If your PR touches the harness boundary (`crates/syllabix-core/src/executor.rs`, `src/openai.rs`, `src/types.rs`, `src/providers.rs`, `src/policy.rs`, `src/sandbox/`, or `tests/harness-quality.rs`), paste the online verdict report into the PR body inside an HTML comment starting with `<!-- syllabix-harness-quality -->`, including the seven fixture lines (`[read-repo]`, `[repo-orientation]`, `[write-and-verify]`, `[write-denied]`, `[network-denied]`, `[secret-denied]`, `[cancelled-command]`) and a summary such as `summary: completion=7/7 ratio=1.00 policy_violations=0 secret_exposures=0 stale_results=0`. A CI check enforces this; CI itself never calls the model.

## Packaging (maintainers)

The app and `syllabix-core` share one version number. A release tag covers both. Say what changed for library users in the GitHub Release notes for that tag. Other repos pin the tag, not `main`. See [embed.md](embed.md#versions).

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

Linux contributors who hit a missing ALSA link need `libasound2-dev`.

## CI

Per-PR CI (`ci.yml`) runs on GitHub-hosted Linux: fmt, clippy, and llvm-cov
≥85% workspace lines and ≥85% lines in every reported `src/**/*.rs` file,
without loading native weights. PR template and harness-evidence checks also
run on hosted Linux. Weekly CI runs `cargo test --workspace` on Linux /
Windows (hosted) and macOS (self-hosted Apple Silicon), including
launch-stack native inference.

A separate weekly **native coverage** workflow (`native-tests.yml`, job
`coverage-native`) runs launch-stack weights under
`cargo llvm-cov --no-cfg-coverage --release` with
`-p syllabix-core --features native-inference --test native_inference`.
It defaults to the self-hosted Mac (Metal / Apple Silicon); `workflow_dispatch`
can override the pool to `ubuntu-latest`. Reports upload as
`coverage-cobertura-native`. No native percentage floor yet — baseline first
(#174).
