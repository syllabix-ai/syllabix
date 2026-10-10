# Embed

Other repos can use Syllabix without forking `pipeline.rs`. The default product is still `syllabix run`.

```toml
[dependencies]
syllabix-core = { git = "https://github.com/syllabix-ai/syllabix.git", tag = "v0.1.0" }
```

```bash
cargo add syllabix-core --git https://github.com/syllabix-ai/syllabix.git --tag v0.1.0
```

## Tags and versions

The SDK (`syllabix-core`) and the CLI (`syllabix`) share one version: `[workspace.package] version` in the root `Cargo.toml`. A git tag `vX.Y.Z` is that version for both crates and for the GitHub Release binaries. `syllabix-native` is not a public API; it ships as a workspace member of the same tag.

Pin `tag = "vX.Y.Z"`. Do not track `main` or a branch. `main` can change without a version bump.

What each tag contains: [CHANGELOG.md](../CHANGELOG.md). Maintainers cut tags as described in [contributing — Packaging](contributing.md#packaging-maintainers).

## Three lanes

| Lane | Status | What you do | When it fits |
| --- | --- | --- | --- |
| **1 — Sidecar binary** | Ready | Spawn a Release binary (or `cargo install --git`) with `syllabix.yaml` in the working directory | Electron / Tauri / Python / CI. No CMake in your repo. No custom STT/LLM. |
| **2 — Embed the live loop** | Ready | `run_live` / `run_live_with_controls` + `LoopEvent` | In-process Rust host that wants the shipped conversation loop. |
| **3 — Compose stages** | Preview | Implement `Vad` / `Stt` / `Llm` / `Tts` / `AudioCapture` / `AudioSink`, swap them on a `Session`, and run | Bring your own engine (often only the LLM). |

Ready = used by the CLI, bugfixes only. Preview = on `main`, may churn.

## Lane 1 — Sidecar binary

The usual Lane 1 install is a [GitHub Release](https://github.com/syllabix-ai/syllabix/releases) binary ([install](install.md)). That path needs no CMake in your repo.

From a machine that already has the compile toolchain, you can instead install the CLI from a git tag:

```bash
cargo install --git https://github.com/syllabix-ai/syllabix.git --tag v0.1.0 syllabix
```

`syllabix` is the CLI package. Pin `--tag vX.Y.Z`. Do not track `main`. This compiles ggml, so it needs the same packages as [Host compile](#host-compile) — listed once there and in [install — Compile](install.md#compile). Full command notes: [install — Install from git](install.md#install-from-git).

**Contract**

| Piece | What hosts may rely on |
| --- | --- |
| CLI | `syllabix run`, `syllabix run --barge-in`, `syllabix init [DIR]`, `syllabix --help`. Full list: [cli](cli.md). |
| Config | Optional `syllabix.yaml` in the process working directory. Missing file uses built-ins. Keys: [syllabix.yaml](syllabix-yaml.md). |
| Environment | `SYLLABIX_LLM_API_KEY` for `pipeline.llm.provider: online` (never yaml). `SYLLABIX_CACHE_DIR` to share or relocate the model cache. |
| Exit codes | `0` success. `1` I/O, config, device, cache, cancel, provider, or pipeline failure. `2` command not implemented. |

`--help` and `init` download nothing. First `run` of the default stack fetches about 2.2 GB into the cache (below).

## Lane 2 — Embed the live loop

The host compiles ggml ([Host compile](#host-compile)), shares the model cache ([Cache](#cache)), and must carry Apache-2.0 plus Sonora BSD in its NOTICE.

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

Treat only this list as the supported crate-root surface. Other `pub use` items in `syllabix-core` are not a stability promise. New crate-root re-exports still need a line on this page before they land ([contributing](contributing.md#supported-library-api)). The automated check that every crate-root name is mentioned here comes back in Phase 2, after fakes, `*_ASSET` constants, and the sandbox/G2P helpers leave the crate root.

| Type / function | Role |
| --- | --- |
| `AgentConfig` | Validated yaml / built-in stack (`v0`, `resolve_for_run`, `load_path`, `parse_yaml`). |
| `run_live` | Load weights, open default mic/speakers, run until shutdown or capture ends (`config`, `Cancel`, `Option<Sender<LoopEvent>>`, barge-in `bool`). |
| `run_live_with_controls` | `run_live_with_controls(config, cancel, Some(tx), controls)`. |
| `Cancel` | Shutdown and in-flight generation cancel. |
| `RuntimeControls` | Barge-in and mic mute while the loop is running. |
| `LoopEvent` | Ready, user/partial/assistant text, thinking, AEC, tools, timings, playback. |
| `LoopReport` | Completed turns, queue occupancy, cancelled/skipped. |
| `Error` / `Result` | Recoverable failures; `Error::exit_code` matches the CLI. |
| `ModelCache` / `cache_root` | First-run weight cache (`ModelCache::v0()` is the product cache). |

## Lane 3 — Compose stages

Build a [`Session`](../crates/syllabix-core/src/session.rs) from yaml (or `Session::v0()`). Unset stages load like `run_live`: Silero, the configured STT/LLM/TTS from the cache, native mic and speakers. Swap only the stages you own. You do not wire `load_real_providers`, `HttpFetcher`, or `NativeCapture` by hand.

`run_loop` (iterator of `AudioFrame`) / `run_loop_captured` and `load_real_providers` still exist if you want the lower-level seam. Arguments for the loader ([`real.rs`](../crates/syllabix-core/src/real.rs)):

```text
load_real_providers(
    cache: &ModelCache,                 // ModelCache::v0()
    fetcher: &dyn Fetcher,              // &HttpFetcher
    progress: &mut dyn Progress,        // &mut StderrProgress
    cancel: &Cancel,
    config: &AgentConfig,
    llm_api_key: Option<&zeroize::Zeroizing<String>>,
) -> Result<(SileroVad, LiveStt, LiveLlm, LiveTts)>
```

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

The trait impl above is covered by `crates/syllabix-core/examples/custom-llm.rs`: a CI-checked copy that puts `EchoLlm` on a `Session` with fixture VAD/STT/TTS/sink and in-memory frames (`cargo clippy --workspace --all-targets` compiles examples, so the supported API cannot bitrot). Production hosts typically swap only the LLM and keep the other defaults.

A yaml `pipeline.llm.provider: online` still needs `SYLLABIX_LLM_API_KEY` when the LLM slot is the default. A custom `Llm` does not. Native mic/speaker types live under `syllabix_core::audio` and are **not** in the Lane 2 freeze list.

### Supported types (Lane 3, additional)

| Type / function | Role |
| --- | --- |
| `Session` | Builder: defaults for every stage; `with_vad` / `with_stt` / `with_llm` / `with_tts` / `with_capture` / `with_sink` / `with_frames` then `run`. |
| `Vad`, `Stt`, `Llm`, `Tts` | Pipeline stages. |
| `AudioCapture`, `AudioSink` | Mic (or fixture) and speakers (or collector). |
| `run_loop` / `run_loop_captured` | In-memory cascade; blocks until workers join. |
| `PipelineStages`, `LoopConfig` | Arguments to `run_loop`. |
| `load_real_providers` | Silero + configured STT/LLM/TTS from the cache. |

Do not fork `pipeline.rs`. Swap a trait impl instead.

## Host compile

Lane 2 and Lane 3 compile `syllabix-native` (vendored llama.cpp / whisper.cpp, one ggml via CMake in `build.rs`). Lane 1 hosts that only spawn a Release binary do not need this toolchain. A Lane 1 `cargo install --git` build does compile the CLI, so it needs the same packages ([install — Install from git](install.md#install-from-git)).

Same OS packages as [install — Compile](install.md#compile), except tools this repo uses only for its own tests (`bubblewrap` in `scripts/setup-linux.sh` is for sandbox CI, not for linking `syllabix-core`).

| Need | Why |
| --- | --- |
| Rust 1.91+ | `rust-toolchain.toml` and workspace `rust-version`. |
| CMake and a C++ compiler | Static ggml. Cold compile takes several minutes. |
| Linux: `pkg-config` + ALSA headers (`libasound2-dev`) | `cpal` uses ALSA. |
| macOS: Xcode Command Line Tools | Apple Clang; Metal + Accelerate link automatically. |
| Windows: MSVC + Windows SDK + CMake | `x86_64-pc-windows-msvc`; open an x64 Native Tools prompt so `cl.exe` is on `PATH`. |

**Linux (Debian / Ubuntu)** — what a host repo actually needs to compile:

```bash
sudo apt-get install -y build-essential cmake pkg-config libasound2-dev
```

A Syllabix checkout can run `./scripts/setup-linux.sh` instead (CI Linux jobs do). That script also installs `git` and `bubblewrap`; neither is required just to compile an out-of-tree crate that depends on `syllabix-core`. Optional Vulkan: `./scripts/setup-linux.sh --with-vulkan`, then `SYLLABIX_GGML_VULKAN=1`. Default Linux / Windows ggml is portable CPU.

**macOS** — Xcode Command Line Tools, CMake (Homebrew `cmake` if missing), rustup. Prefer rustup’s `cargo` over Homebrew’s. A Syllabix checkout can run `./scripts/setup-macos.sh` (CI macOS jobs do).

**Windows** — [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with **Desktop development with C++**, [CMake](https://cmake.org/download/) on `PATH`, rustup for `x86_64-pc-windows-msvc`. Weekly CI compiles on GitHub `windows-latest` with that toolchain and no extra setup script.

## Cache

Weights download on first use of each id (HTTPS URL pinned in the binary, SHA-256, then reuse). `--help` and `init` download nothing. Other yaml ids fetch the first time that id is used.

The default stack (Silero, Whisper `small`, LFM2.5-2.6B, Pocket TTS) is about **2.2 GB**. Pinned `size_bytes` in [`manifest.rs`](../crates/syllabix-core/src/models/manifest.rs) sum to 2,234,270,682 bytes. Catalogue: [engines](engines.md).

`cache_root()` (then `ModelCache::v0()`) resolves in this order. Assets always land in `<root>/models/v1` (`Manifest` version 1):

| When | Cache root | Asset directory |
| --- | --- | --- |
| `SYLLABIX_CACHE_DIR` set | that directory (as-is) | `$SYLLABIX_CACHE_DIR/models/v1` |
| else `$XDG_CACHE_HOME` set | `$XDG_CACHE_HOME/syllabix` | `$XDG_CACHE_HOME/syllabix/models/v1` |
| else Windows | `%LOCALAPPDATA%\syllabix\cache` | `%LOCALAPPDATA%\syllabix\cache\models\v1` |
| else `$HOME` set | `~/.cache/syllabix` | `~/.cache/syllabix/models/v1` |
| else | `.syllabix-cache` (cwd) | `.syllabix-cache/models/v1` |

Set `SYLLABIX_CACHE_DIR` to the **cache root**, not to `models/v1`. Two apps share downloads when they use the same root: leave the variable unset (same user default) or export the same path in every process (CLI, embed hosts, CI). `ModelCache` takes an advisory lock per asset so parallel first-run resolves do not race.

## Licenses

Hosts that embed `syllabix-core` ship Syllabix code plus vendored audio/ML and fetched weights. Reproduce the notices your distribution actually includes.

| Layer | License | What to do |
| --- | --- | --- |
| Syllabix (`syllabix`, `syllabix-core`) | Apache-2.0 | Keep [LICENSE](../LICENSE). |
| Sonora (WebRTC AEC3) | BSD 3-Clause | Copy the Sonora / WebRTC notice from [NOTICE](../NOTICE) into your NOTICE. |
| llama.cpp, whisper.cpp, ggml | MIT | Vendored; see `vendor/*/LICENSE`. |
| ONNX Runtime (`ort`) | MIT | Transitive. |
| Default VAD (Silero) | MIT | Fetched into the cache. |
| Default STT (Whisper `small`) | MIT (OpenAI Whisper) | Fetched into the cache. |
| Default LLM (LFM2.5-2.6B) | LiquidAI LFM 1.0 | **Not** Apache-2.0. Governs the default local weights. |
| Default TTS (Pocket TTS) | CC BY 4.0 | Fetched into the cache; attribution in NOTICE. |
| Optional yaml models | Llama 3.2 Community, Qwen Apache-2.0, Kokoro Apache-2.0, Moonshine MIT, … | Only if you select that id. |

Full third-party list and model URLs: [NOTICE](../NOTICE). Nothing in the default binary requires a GPL component at runtime or as a static link.
