# Embed

Other repos can use Syllabix without forking `pipeline.rs`. The default product is still `syllabix run`.

Depend on the library from [crates.io](https://crates.io/crates/syllabix-core). API docs: [docs.rs/syllabix-core](https://docs.rs/syllabix-core).

Most apps use the defaults (same as `syllabix run`). The first build compiles the bundled whisper.cpp / llama.cpp with CMake and a C++ compiler ([Host compile](#host-compile)). Nothing is downloaded at build time. Models still download on first run ([Cache](#cache)).

```bash
cargo add syllabix-core
```

```toml
[dependencies]
syllabix-core = "0.1"
```

If your app brings its own speech-to-text, LLM, and text-to-speech, turn the defaults off. Then you don't need CMake:

```toml
[dependencies]
syllabix-core = { version = "0.1", default-features = false }
```

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

The `syllabix` CLI is not on crates.io. Install it from a [GitHub Release](https://github.com/syllabix-ai/syllabix/releases) or `cargo install --git` (Lane 1).

## Tags and versions

The SDK (`syllabix-core`) and the CLI (`syllabix`) share one version. A git tag `vX.Y.Z` is that version for both crates, the GitHub Release binaries, and the crates.io publish. `syllabix-native` is not a public API; depend on `syllabix-core`.

What each release contains is on the git tag and its [GitHub Release](https://github.com/syllabix-ai/syllabix/releases). Maintainers cut tags as described in [contributing — Packaging](contributing.md#packaging-maintainers). Hosts: [Compatibility](#compatibility).

## Compatibility

Depend on `syllabix-core` from crates.io, and read the [GitHub Release](https://github.com/syllabix-ai/syllabix/releases) notes before upgrading.

The supported API is only the freeze lists on this page: [Lane 2](#supported-types-lane-2), [Lane 2 with `full`](#supported-types-lane-2-needs-full), [Lane 3](#supported-types-lane-3-additional), and [Lane 3 with `full`](#supported-types-lane-3-needs-full). These are names that are always at the crate root, plus the ones that exist only with `full`.

CI checks this list.

Items in named modules (`testing`, `g2p`, `sandbox`, and the rest) and the `*_ASSET` constants are **not** covered unless a freeze list names them.

## Three lanes

| Lane | Status | What you do | When it fits |
| --- | --- | --- | --- |
| **1 — Sidecar binary** | Ready | Run a GitHub Release binary (usual) or `cargo install --git`. Put `syllabix.yaml` in the working directory. | Electron / Tauri / Python / CI. No custom STT/LLM. A Release needs no compile; `cargo install --git` does. |
| **2 — Embed the live loop** | Ready | `run_live` / `run_live_with_controls` + `LoopEvent` | In-process Rust host that wants the shipped conversation loop. |
| **3 — Compose stages** | Preview | Implement `Vad` / `Stt` / `Llm` / `Tts` / `AudioCapture` / `AudioSink`, swap them on a `Session`, and run | Bring your own engine (often only the LLM). |

Ready = used by the CLI. Preview = less proven. Depend from crates.io ([Compatibility](#compatibility)).

## Cargo features

`syllabix-core` defaults to the same build as `syllabix run`: local speech engines, microphone and speakers, and the online LLM adapter. With `default-features = false` you get `run_loop`, the provider traits, `Session`, and the in-memory stages under `syllabix_core::testing` if you supply every stage yourself. That path does not run CMake and does not compile whisper.cpp / llama.cpp.

| Feature | On by default? | Who should enable it | What it turns on |
| --- | --- | --- | --- |
| `full` | yes (`default`) | Lane 2, and Lane 3 hosts that still use shipped engines or the real mic/speakers | `ggml` + `device-audio` + `cloud-llm`. `run_live`, `load_real_providers`, and `Session` defaults. |
| `ggml` | via `full` | Lane 3 that still loads this crate’s Whisper / local LLM / Qwen engines | Optional `syllabix-native` (CMake + C++). |
| `device-audio` | via `full` | Lane 3 that still uses the real microphone and speakers | Device I/O and echo cancellation (`NativeCapture` / `NativePlayback`). |
| `cloud-llm` | via `full` | Hosts whose yaml uses `pipeline.llm.provider: online` with `load_real_providers` | `LiveLlm::Cloud` in the real loader. `openai::OpenAiLlm` always compiles. |
| `native-inference` | no | Contributors running the slow model tests (`native_inference`) | Turns on `ggml` plus the tests that download and run the real models (GGUF and ONNX). ONNX (`ort`) is always built in; this feature does not switch it on. |

**Lane 3 that implements every stage** (see `examples/custom-llm.rs`): `default-features = false`.  
**Lane 3 that swaps only the LLM and keeps the other defaults:** keep `full` (or at least `ggml` + `device-audio`). That still compiles whisper.cpp.  
**Lane 2 `run_live`:** `full` (or `ggml` + `device-audio`).

Do not enable `ggml` unless you have the [Host compile](#host-compile) toolchain. Hosts should not enable `native-inference`.

## Lane 1 — Sidecar binary

The usual Lane 1 install is a [GitHub Release](https://github.com/syllabix-ai/syllabix/releases) binary ([install](install.md)). You download a file and run it. You do not compile.

If this machine already has a C++ compiler and CMake, you can instead install from a git tag:

```bash
cargo install --git https://github.com/syllabix-ai/syllabix.git --tag v0.1.1 syllabix
```

Always pass `--tag vX.Y.Z`. Do not install from `main`. This builds the on-device speech libraries, so you need the packages under [Host compile](#host-compile) (same list: [install — Compile](install.md#compile)). More notes: [install — Install from git](install.md#install-from-git).

**Contract**

| Piece | What hosts may rely on |
| --- | --- |
| CLI | `syllabix run`, `syllabix run --barge-in`, `syllabix init [DIR]`, `syllabix --help`. Full list: [cli](cli.md). |
| Config | Optional `syllabix.yaml` in the process working directory. Missing file uses built-ins. Keys: [syllabix.yaml](syllabix-yaml.md). |
| Environment | `SYLLABIX_LLM_API_KEY` for `pipeline.llm.provider: online` (never yaml). `SYLLABIX_CACHE_DIR` to share or relocate the model cache. |
| Exit codes | `0` success. `1` I/O, config, device, cache, cancel, provider, or pipeline failure. `2` command not implemented. |

`--help` and `init` download nothing. First `run` of the default stack fetches about 2.2 GB into the cache (below).

## Lane 2 — Embed the live loop

The host compiles the C++ speech libraries ([Host compile](#host-compile)), shares the model cache ([Cache](#cache)), and must carry Apache-2.0 plus Sonora BSD in its NOTICE.

```rust
use syllabix_core::{run_live, AgentConfig, Cancel};

fn main() -> syllabix_core::Result<()> {
    let config = AgentConfig::resolve_for_run(&std::env::current_dir()?)?;
    run_live(&config, Cancel::new(), None, false)?;
    Ok(())
}
```

The minimal call above is covered by `crates/syllabix-core/examples/embed-loop.rs`: a CI-checked fuller copy that also subscribes to `LoopEvent` and reports `LoopReport` (`cargo clippy --workspace --all-targets` compiles examples, so the supported API cannot bitrot).

`AgentConfig::v0()` is the same zero-config stack as a missing yaml file. Pass `true` as the last argument of `run_live` for barge-in. Subscribe to `LoopEvent` with `std::sync::mpsc::channel` when you need a UI. For a live handle, call `run_live_with_controls(config, cancel, Some(tx), controls)`.

`run_live` fails fast on a missing `SYLLABIX_LLM_API_KEY` when yaml selects an online LLM, before devices or weights load.

### Supported types (Lane 2)

Treat only this list as the supported crate-root surface. Other public items live in named modules (`syllabix_core::testing`, `syllabix_core::g2p`, `syllabix_core::sandbox`, and the rest) and are not a stability promise. New crate-root re-exports still need a line on this page before they land ([contributing](contributing.md#supported-library-api)). `crates/syllabix-core/tests/supported_api.rs` fails if a crate-root `pub use` is missing from these tables, or if a freeze-list name leaves the crate root.

| Type / function | Role |
| --- | --- |
| `AgentConfig` | Validated yaml / built-in stack (`v0`, `resolve_for_run`, `load_path`, `parse_yaml`). |
| `Cancel` | Shutdown and in-flight generation cancel. |
| `RuntimeControls` | Barge-in and mic mute while the loop is running. |
| `LoopEvent` | Ready, user/partial/assistant text, thinking, AEC, tools, timings, playback. |
| `LoopReport` | Completed turns, queue occupancy, cancelled/skipped. |
| `Error` / `Result` | Recoverable failures; `Error::exit_code` matches the CLI. |
| `ModelCache` / `cache_root` | First-run weight cache (`ModelCache::v0()` is the product cache). |

### Supported types (Lane 2, needs `full`)

These crate-root names exist only when the `full` feature is on (`ggml` + `device-audio`). They are still the supported Lane 2 surface.

| Type / function | Role |
| --- | --- |
| `run_live` | Load weights, open default mic/speakers, run until shutdown or capture ends (`config`, `Cancel`, `Option<Sender<LoopEvent>>`, barge-in `bool`). |
| `run_live_with_controls` | `run_live_with_controls(config, cancel, Some(tx), controls)`. |

## Lane 3 — Compose stages

Build a `Session` from yaml (`Session::from_config`) or `Session::v0()`.
Stages you do not set load like `run_live`: Silero, the configured
speech-to-text / language model / text-to-speech from the cache, native
mic and speakers. Swap only the stages you own.

```rust
use syllabix_core::{AgentConfig, Cancel, Session};

let config = AgentConfig::resolve_for_run(&std::env::current_dir()?)?;
Session::from_config(&config)
    .with_llm(EchoLlm)
    .run(Cancel::new())?;
```

You do not wire `load_real_providers`, `HttpFetcher`, or `NativeCapture`
by hand. Those still exist if you want the lower-level seam; see
[`real.rs`](../crates/syllabix-core/src/real.rs).

```rust
use syllabix_core::{Cancel, Llm, Result, TokenChunk, Transcript};

struct EchoLlm;

impl Llm for EchoLlm {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn generate(
        &mut self,
        _history: &[syllabix_core::HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(syllabix_core::Error::Cancelled);
        }
        on_token(TokenChunk {
            turn: user.turn,
            generation: cancel.generation(),
            index: 0,
            text: user.text.clone(),
            is_last: true,
        })
    }
}
```

The trait impl above is covered by `crates/syllabix-core/examples/custom-llm.rs`. That file uses in-memory fixtures so CI can run without a microphone (`cargo clippy --workspace --all-targets` compiles examples, so the supported API cannot bitrot). A host that swaps only the LLM and keeps the other `Session` defaults still needs the `ggml` and `device-audio` features (on by default). Skipping CMake is only for hosts that supply every stage themselves.

A yaml `pipeline.llm.provider: online` still needs `SYLLABIX_LLM_API_KEY` when the LLM slot is the default. A custom `Llm` does not. Native mic/speaker types live under `syllabix_core::audio` and are **not** in the Lane 2 freeze list.

### Supported types (Lane 3, additional)

| Type / function | Role |
| --- | --- |
| `Session` | Builder: defaults for every stage; `with_vad` / `with_stt` / `with_llm` / `with_tts` / `with_capture` / `with_sink` / `with_frames` then `run`. |
| `Vad`, `Stt`, `Llm`, `Tts` | Pipeline stages. |
| `AudioCapture`, `AudioSink` | Mic (or fixture) and speakers (or collector). |
| `run_loop` / `run_loop_captured` | In-memory cascade; blocks until workers join. |
| `PipelineStages`, `LoopConfig` | Arguments to `run_loop`. |
| `AudioFrame` | One PCM frame into VAD or capture. |
| `VadEvent` / `Utterance` / `TurnId` | VAD events and the turn they belong to. |
| `Transcript` / `HistoryTurn` / `TokenChunk` | STT text, chat history, and LLM stream chunks. |
| `SynthesizedAudio` | TTS audio for the sink. |

Do not fork `pipeline.rs`. Swap a trait impl instead.

### Supported types (Lane 3, needs `full`)

`load_real_providers` is crate-root only with the `ggml` feature (on via `full`).

| Type / function | Role |
| --- | --- |
| `load_real_providers` | Silero + configured STT/LLM/TTS from the cache. |

## Host compile

If you embed Syllabix in your own Rust app (Lane 2 or Lane 3), your machine compiles the C++ speech libraries that ship with Syllabix (whisper.cpp / llama.cpp). The source is bundled in the crate; nothing is downloaded at build time. You need CMake and a C++ compiler. The first compile takes several minutes. Lane 3 hosts that set `default-features = false` and supply every stage themselves skip that compile. If you only run a downloaded Syllabix binary (Lane 1), skip this section. A Lane 1 `cargo install --git` build compiles the CLI, so it needs the same packages ([install — Install from git](install.md#install-from-git)).

Same packages as [install — Compile](install.md#compile). This repo’s Linux setup script also installs extra tools used only in our tests; those are called out below.

| You need | Why |
| --- | --- |
| Rust 1.91 or newer | Pinned in this repo. |
| CMake and a C++ compiler | Builds the speech libraries. First compile takes several minutes. |
| Linux: `pkg-config` and ALSA headers (`libasound2-dev`) | Mic and speakers on Linux. |
| macOS: Xcode Command Line Tools | Apple’s C++ compiler. GPU libraries link on their own. |
| Windows: Visual C++ build tools, Windows SDK, and CMake | Open an x64 Native Tools prompt so `cl.exe` is on `PATH`. |

**Linux (Debian / Ubuntu)**

```bash
sudo apt-get install -y build-essential cmake pkg-config libasound2-dev
```

In a Syllabix checkout you can run `./scripts/setup-linux.sh` instead. That script also installs `git` and `bubblewrap`. You do not need those two just to depend on `syllabix-core` from another repo.

**macOS** — Install Xcode Command Line Tools, CMake (Homebrew `cmake` if needed), and rustup. Put rustup’s `cargo` ahead of Homebrew’s. In a Syllabix checkout you can run `./scripts/setup-macos.sh`.

**Windows** — Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **Desktop development with C++**, [CMake](https://cmake.org/download/) on `PATH`, and rustup. Open an x64 Native Tools prompt, then build.

Optional: Linux Vulkan is `SYLLABIX_GGML_VULKAN=1` (needs Vulkan headers). Default Linux and Windows builds use CPU.

## Cache

The first run of the default models downloads about **2.2 GB** (Silero, Whisper `small`, LFM2.5-2.6B, Pocket TTS). Files are fetched over HTTPS, checked, then reused. `--help` and `init` download nothing. Other yaml model ids download the first time you use that id. Full list: [engines](engines.md).

Files always go in `<cache-root>/models/v1`.

| When | Cache root | Files |
| --- | --- | --- |
| `SYLLABIX_CACHE_DIR` is set | that folder | `$SYLLABIX_CACHE_DIR/models/v1` |
| else `$XDG_CACHE_HOME` is set | `$XDG_CACHE_HOME/syllabix` | `$XDG_CACHE_HOME/syllabix/models/v1` |
| else on Windows | `%LOCALAPPDATA%\syllabix\cache` | `%LOCALAPPDATA%\syllabix\cache\models\v1` |
| else `$HOME` is set | `~/.cache/syllabix` | `~/.cache/syllabix/models/v1` |
| else | `.syllabix-cache` in the current directory | `.syllabix-cache/models/v1` |

Set `SYLLABIX_CACHE_DIR` to the **cache root**, not to `models/v1`. Two apps share downloads when they use the same root: leave the variable unset (same user default) or set the same path in every process.

## Licenses

Hosts that embed `syllabix-core` ship Syllabix code plus vendored audio/ML and fetched weights. Reproduce the notices your distribution actually includes.

| Layer | License | What to do |
| --- | --- | --- |
| Syllabix (`syllabix`, `syllabix-core`) | Apache-2.0 | Keep [LICENSE](../LICENSE). |
| Sonora (WebRTC AEC3) | BSD 3-Clause | Copy the Sonora / WebRTC notice from [NOTICE](../NOTICE) into your NOTICE. |
| llama.cpp, whisper.cpp, ggml | MIT | Vendored; see `crates/syllabix-native/vendor/*/LICENSE`. |
| ONNX Runtime (`ort`) | MIT | Transitive. |
| Default VAD (Silero) | MIT | Fetched into the cache. |
| Default STT (Whisper `small`) | MIT (OpenAI Whisper) | Fetched into the cache. |
| Default LLM (LFM2.5-2.6B) | LiquidAI LFM 1.0 | **Not** Apache-2.0. Governs the default local weights. |
| Default TTS (Pocket TTS) | CC BY 4.0 | Fetched into the cache; attribution in NOTICE. |
| Optional yaml models | Llama 3.2 Community, Qwen Apache-2.0, Kokoro Apache-2.0, Moonshine MIT, … | Only if you select that id. |

Full third-party list and model URLs: [NOTICE](../NOTICE). Nothing in the default binary requires a GPL component at runtime or as a static link.
