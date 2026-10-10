# Embed

Other repos can use Syllabix without forking `pipeline.rs`. The default product is still `syllabix run`.

```toml
[dependencies]
syllabix-core = { git = "https://github.com/syllabix-ai/syllabix.git", tag = "v0.1.0" }
```

```bash
cargo add syllabix-core --git https://github.com/syllabix-ai/syllabix.git --tag v0.1.0
```

## Three lanes

| Lane | Status | What you do | When it fits |
| --- | --- | --- | --- |
| **1 — Sidecar binary** | Ready | Spawn a Release binary (or `cargo install --git`) with `syllabix.yaml` in the working directory | Electron / Tauri / Python / CI. No CMake in your repo. No custom STT/LLM. |
| **2 — Embed the live loop** | Ready | `run_live` / `run_live_with_controls` + `LoopEvent` | In-process Rust host that wants the shipped conversation loop. |
| **3 — Compose stages** | Preview | Implement `Vad` / `Stt` / `Llm` / `Tts` / `AudioCapture` / `AudioSink` and call `run_loop` | Bring your own engine (often only the LLM). |

Ready = used by the CLI, bugfixes only. Preview = on `main`, may churn.

## Lane 1 — Sidecar binary

Download a [GitHub Release](https://github.com/syllabix-ai/syllabix/releases) artifact ([install](install.md)) or, from a machine that already has the compile toolchain, `cargo install --git https://github.com/syllabix-ai/syllabix.git --tag v0.1.0 syllabix`.

**Contract**

| Piece | What hosts may rely on |
| --- | --- |
| CLI | `syllabix run`, `syllabix run --barge-in`, `syllabix init [DIR]`, `syllabix --help`. Full list: [cli](cli.md). |
| Config | Optional `syllabix.yaml` in the process working directory. Missing file uses built-ins. Keys: [syllabix.yaml](syllabix-yaml.md). |
| Environment | `SYLLABIX_LLM_API_KEY` for `pipeline.llm.provider: online` (never yaml). `SYLLABIX_CACHE_DIR` to share or relocate the model cache. |
| Exit codes | `0` success. `1` I/O, config, device, cache, cancel, provider, or pipeline failure. `2` command not implemented. |

`--help` and `init` download nothing. First `run` of the default stack fetches about 2.2 GB into the cache (below).

## Lane 2 — Embed the live loop

The host compiles ggml (CMake, C++, ALSA on Linux), shares the model cache, and must carry Apache-2.0 plus Sonora BSD in its NOTICE.

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

Implement the stage you want to replace. Call `load_real_providers` for the stages you keep, then `run_loop` (iterator of `AudioFrame`) or `run_loop_captured` (an `AudioCapture`). Arguments ([`real.rs`](../crates/syllabix-core/src/real.rs)):

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

The trait impl above is covered by `crates/syllabix-core/examples/custom-llm.rs`: a CI-checked copy that wires `EchoLlm` into `PipelineStages` with fixture VAD/STT/TTS/sink and `run_loop` (`cargo clippy --workspace --all-targets` compiles examples, so the supported API cannot bitrot). Production hosts typically keep the stages they do not replace from `load_real_providers`.

Wire `EchoLlm` into `PipelineStages` with the VAD/STT/TTS you loaded (or your own impls) and a sink. Native mic/speaker types live under `syllabix_core::audio` and are **not** in the Lane 2 freeze list.

### Supported types (Lane 3, additional)

| Type / function | Role |
| --- | --- |
| `Vad`, `Stt`, `Llm`, `Tts` | Pipeline stages. |
| `AudioCapture`, `AudioSink` | Mic (or fixture) and speakers (or collector). |
| `run_loop` / `run_loop_captured` | In-memory cascade; blocks until workers join. |
| `PipelineStages`, `LoopConfig` | Arguments to `run_loop`. |
| `load_real_providers` | Silero + configured STT/LLM/TTS from the cache. |

Do not fork `pipeline.rs`. Swap a trait impl instead.

## Host compile

Lane 2 and Lane 3 compile `syllabix-native` (vendored llama.cpp / whisper.cpp, one ggml). Same bar as [install — Compile](install.md#compile):

- Rust 1.91+ (`rust-toolchain.toml`)
- CMake and a C++ compiler
- Linux: ALSA headers (`libasound2-dev`) and `pkg-config` (`./scripts/setup-linux.sh`)
- Darwin: Metal + Accelerate for STT/LLM
- Linux / Windows: portable CPU by default; Linux Vulkan is `SYLLABIX_GGML_VULKAN=1`

Lane 1 hosts do not need CMake.

## Cache

Weights download on first use of each id (HTTPS URL pinned in the binary, SHA-256, then reuse). Default stack is about **2.2 GB** (Silero, Whisper `small`, LFM2.5-2.6B, Pocket TTS). Catalogue: [engines](engines.md).

| Variable / path | Effect |
| --- | --- |
| `SYLLABIX_CACHE_DIR` | Cache root. Assets land in `$SYLLABIX_CACHE_DIR/models/v1`. |
| unset, Unix | `~/.cache/syllabix/models/v1` (or `$XDG_CACHE_HOME/syllabix/models/v1`) |
| unset, Windows | `%LOCALAPPDATA%\syllabix\cache\models\v1` |

Point every app on the machine at the same `SYLLABIX_CACHE_DIR` to share downloads. `ModelCache` uses an advisory lock so parallel resolves of one asset do not race.

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
