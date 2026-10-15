# Contributing

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

Exclusive extra native suites (not the default `cargo test --workspace` bar).
`--features native-inference` turns on `ggml` and the tests that download
and run the real models (GGUF and ONNX). ONNX (`ort`) is always built in.

```bash
SYLLABIX_NATIVE_MODELS=qwen3-0.6 cargo test -p syllabix-core --features native-inference --test native_inference
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

CPU vs Metal tok/s for the three default GGUFs: [`crates/syllabix-native/vendor/llama-bench.md`](../crates/syllabix-native/vendor/llama-bench.md). ggml vendor pins and local patches: [`crates/syllabix-native/vendor/README.md`](../crates/syllabix-native/vendor/README.md). Measurement protocols: [`reference-profiles.md`](reference-profiles.md).

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

If your PR touches the harness boundary (`crates/syllabix-core/src/executor.rs`, `src/openai.rs`, `src/types.rs`, `src/providers.rs`, `src/policy.rs`, `src/sandbox/`, or `tests/harness-quality.rs`), paste the online verdict report into the PR body inside an HTML comment starting with `<!-- syllabix-harness-quality -->`, including the seven fixture lines (`[read-repo]`, `[repo-orientation]`, `[write-and-verify]`, `[write-denied]`, `[network-denied]`, `[secret-denied]`, `[cancelled-command]`) and a summary such as `summary: completion=7/7 ratio=1.00 policy_violations=0 secret_exposures=0 stale_results=0`. A CI check enforces this; CI itself never calls the model. Import-only edits on those files (moved `use` lines, comments, or type paths with no behavior change) do not need a pasted report; the evidence workflow skips them.

## Packaging (maintainers)

Three packages share one version: the library other apps use
(`syllabix-core`), the on-device speech build it compiles (`syllabix-native`),
and the `syllabix` command. Push a tag named `vX.Y.Z`. The version comes
from the tag; don't bump Cargo.toml by hand. That tag is the GitHub Release
and the crates.io publish of `syllabix-core` / `syllabix-native`. App
binaries from the GitHub Release report the same version (`./syllabix --version`).
Other apps normally depend with `syllabix-core = "0.1"`; a git pin (`tag = "vX.Y.Z"`,
or unreleased `main`) is optional. Release notes are on the tag and the
GitHub Release. If you are embedding Syllabix in another app, see
[`embed.md`](embed.md).

Pushing the tag also publishes crates to crates.io. A pull request never
uploads anything. The publish job writes the tag version (`0.1.0` for
`v0.1.0`) into Cargo.toml for that run only; it is not committed. Publish
in this order:

1. `syllabix-native` first. It contains the whisper.cpp / llama.cpp source
   the library needs to compile.
2. Wait until that version is listed on crates.io.
3. Then `syllabix-core`, which depends on that same version of
   `syllabix-native`.

The `syllabix` command is not on crates.io. People download the app from
GitHub Releases. The crates.io job is `.github/workflows/crates-io.yml`.
Before pushing a tag, you can rehearse it from the Actions tab (crates.io
workflow, Run workflow, enter the tag); nothing is uploaded.

Create a GitHub Release by pushing the tag only. Do not click "Create a
new release" on the GitHub Releases page. If you publish that empty
release, GitHub locks the name and this workflow cannot attach the
binaries. A leftover draft is fine: the workflow will upload onto it.
This repo uses immutable releases. Once a version's tag name has been
used for a published release, GitHub never lets that name come back, even
if you delete the release.

If that already happened, recover from the Actions tab: open the Release
workflow, Run workflow, pick the existing tag as the branch/tag, and fill
`release_name` with a name that does not start with `v` (for example
`app-v0.1.1`). The job builds from that tag and publishes the binaries
under the new name at the same commit. Leave `release_name` empty to
package artifacts without publishing.

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

After a Release is published, validate the documented download path: `SMOKE_RELEASE_URL=https://github.com/syllabix-ai/syllabix/releases/download/<tag> scripts/smoke-setup.sh syllabix-Linux-x86_64` (repeat per OS you can touch). The release workflow also opens a PR syncing the `vX.Y.Z` pins in the install docs (`scripts/sync-release-pins.sh`); merge it after the release.

Linux contributors who hit a missing ALSA link need `libasound2-dev`.

## Supported library API

`syllabix-core` crate-root re-exports are the Lane 2 and Lane 3 lists in
[`embed.md`](embed.md). Other public items live in named modules and can
change without notice.

Do not add a new crate-root `pub use` unless you also document it in
`docs/embed.md` (either as supported, or as explicitly not frozen). The same
rule sits above the re-exports in `crates/syllabix-core/src/lib.rs`.
`crates/syllabix-core/tests/supported_api.rs` fails if a freeze-list name
leaves the crate root, and if a crate-root re-export is missing from those
lists. The test honors `#[cfg(feature)]` so names that need `full` are
only required when those features are on.

## CI

Per-PR CI (`ci.yml`) runs on GitHub-hosted Linux: fmt, clippy, and llvm-cov
≥85% workspace lines and ≥85% lines in every reported `src/**/*.rs` file,
without loading native weights. A **native inference** check
(`native-tests.yml`) runs the real `whisper-small`, `lfm2.5-230m`,
and `pocket-tts` weights on pull requests and pushes to `main` on
`ubuntu-latest`. It runs when crates, lockfiles, setup scripts, or that
workflow change; docs-only PRs skip the heavy steps so the check still
reports. Model weights are cached with `actions/cache`, keyed on those
ids plus a hash of `manifest.rs`.
A **native serve** job in the same workflow (Linux only) starts
[`syllabix serve tts`](cli.md#serve) then `serve stt`, waits for each
`GET /health`, and runs the six-turn real loop test against those
servers (`SYLLABIX_STT_BASE_URL` / `SYLLABIX_TTS_BASE_URL`, keys
`SYLLABIX_STT_API_KEY` / `SYLLABIX_TTS_API_KEY`). Same path gate and
model cache; VAD and the LLM stay in-process.
A **sdk no-native** job
`cargo check`s `syllabix-core` with `--no-default-features` on a runner
without CMake (`setup-linux.sh` is skipped). A Linux-only **crates.io
package** job runs `cargo package -p syllabix-native` with verify (builds
from the tarball) and lists `syllabix-core` without verify (`--verify` would
download that version of `syllabix-native` from crates.io, which fails when
that version is not published yet). It fails if the native `.crate` is over
10 MB. PR template and harness-evidence checks also run on hosted Linux.

The **git-consumer** workflow (`git-consumer.yml`) pretends to be another repo.
It depends on `syllabix-core` from git: a `v*` tag when one exists, otherwise
the commit under test. It then compiles
`crates/syllabix-core/examples/embed-loop.rs` and `custom-llm.rs`. Locally, run
`./scripts/ci-git-consumer.sh`.

Weekly CI runs `cargo test --workspace` on Linux / Windows (hosted) and macOS
(self-hosted Apple Silicon), including launch-stack native inference.

The native inference check runs those weights under
`cargo llvm-cov --no-cfg-coverage --release` with
`-p syllabix-core --features native-inference --test native_inference`
(`native-inference` turns on `ggml`; ONNX stays always-on). Reports upload as
`coverage-cobertura-native`. No native percentage floor yet — baseline first
(#174).
