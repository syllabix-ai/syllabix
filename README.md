# Syllabix

Local voice agent. One native binary. Apache-2.0.

Download `syllabix`, run it, talk. No Python, pip, or API key on the default path.

```text
mic → Silero VAD → STT → LLM → TTS → speakers
              ↑______________ barge-in (interrupt + flush)
```

[Install](#install) ·
[Talk](#talk) ·
[Features](#features) ·
[Compare](#comparison) ·
[Requirements](#requirements) ·
[Engines](#engines) ·
[Architecture](#architecture) ·
[API](#api) ·
[Docs](#docs) ·
[FAQ](#faq)

## At a glance

| | Syllabix |
| --- | --- |
| **Workflow** | Full-duplex spoken conversation on the laptop microphone and speakers |
| **Default stack** | Silero VAD · Whisper `small` · LFM2.5-2.6B · Pocket TTS (~2.2 GB first-run cache) |
| **Platforms** | Linux x64 · macOS Apple Silicon · macOS Intel · Windows x64 |
| **Compute** | Metal + Accelerate on Darwin (Whisper and local LLM); portable CPU on Linux and Windows; Qwen TTS always CPU |
| **Interface** | Native CLI + TUI (`syllabix run`). There is no `serve` command and no HTTP voice API |
| **Config** | Zero-config by default. Optional `syllabix.yaml` from `syllabix init` |
| **Data path** | Audio stays on the machine. An optional online LLM sends transcript text only |
| **License** | Apache-2.0; downloaded models keep their upstream terms |

## Install

Download a package from the [latest release](https://github.com/syllabix-ai/syllabix/releases/latest), verify `SHA256SUMS`, then run.

| Platform | Artifact | Guide |
| --- | --- | --- |
| Linux x64 | `syllabix-Linux-x86_64` | [Install on Linux](docs/install/linux.md) |
| macOS Apple Silicon | `syllabix-Darwin-arm64` | [Install on macOS](docs/install/macos.md) |
| macOS Intel | `syllabix-Darwin-x86_64` | [Install on macOS](docs/install/macos.md) |
| Windows x64 | `syllabix-Windows-x86_64.exe` | [Install on Windows](docs/install/windows.md) |

```bash
# macOS / Linux
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS   # macOS: shasum -a 256 -c SHA256SUMS --ignore-missing
chmod +x syllabix
./syllabix run
```

```powershell
# Windows PowerShell
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Windows-x86_64.exe -o syllabix.exe
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
certutil -hashfile syllabix.exe SHA256    # compare against the SHA256SUMS line
.\syllabix.exe run
```

First `run` fetches Silero, Whisper `small`, LFM2.5-2.6B, and Pocket TTS into `~/.cache/syllabix/models/v1` (`%LOCALAPPDATA%\syllabix\cache\models\v1` on Windows; override with `$SYLLABIX_CACHE_DIR`) and verifies SHA-256. Later runs reuse that cache offline. `--help` and `init` do not need the cache.

The first-run download is large. Time-to-first-spoken-reply on a typical connection has not been published yet; do not plan on three minutes for a cold cache.

macOS Gatekeeper and Windows SmartScreen steps: [docs/install.md](docs/install.md).

## Talk

`syllabix run` uses the built-in microphone and speakers. Grant microphone access when the OS asks. Echo control (WebRTC AEC3) is on by default and calibrates for about 10 seconds — wait before speaking. If laptop speakers still trigger the agent as if you were talking, use headphones and include the device names from the startup line in a bug report.

```bash
./syllabix run              # default: finish the sentence before listening
./syllabix run --barge-in   # stop playback when you talk over the agent
./syllabix init             # optional: write syllabix.yaml in a project folder
```

There is no `serve` command. Default `run` needs no yaml and no API key. After the cache is full, it needs no network.

## Features

| Area | Included |
| --- | --- |
| **Spoken loop** | Mic → VAD → STT → LLM → TTS → speakers in one process |
| **Echo control** | Full-duplex AEC3 on by default; calibrates ~10 s on first listen |
| **Barge-in** | Opt-in `run --barge-in`: SpeechStart cancels playback and the in-flight turn |
| **Zero config** | Built-in defaults talk with no yaml and no API key |
| **Optional yaml** | Whisper size, Moonshine streaming STT, local LLM id, spoken `system_prompt`, TTS engine, online LLM |
| **Local models** | whisper.cpp, llama.cpp, Silero ONNX, Pocket TTS / Kokoro / Qwen TTS |
| **Online LLM (opt-in)** | OpenAI-compatible `chat/completions`; audio stays on the machine |
| **Diagnostics** | Yaml-only per-turn timelines and WAVs ([diagnostics](docs/diagnostics.md)) |
| **Developer harness** | Optional sandboxed shell for `online` LLM or local `lfm2.5-2.6b` |

Full list: [docs/features.md](docs/features.md).

## Comparison

Syllabix is a local spoken agent, not a hosted voice API and not a Python pipeline SDK.

| | **Syllabix** | **Hosted realtime APIs** | **Pipeline SDKs (Pipecat, LiveKit Agents)** | **HF speech-to-speech** |
| --- | --- | --- | --- | --- |
| **Best fit** | Talk on a laptop with one binary | Fast cloud quality, no local weights | Build a custom multi-service voice app | Research loop with many backends |
| **Install** | One Release artifact + first-run cache | Account and API key | Python env, workers, often a cloud SFU | Python, CUDA notes, flag soup |
| **Data path** | Audio on-device; optional LLM may send text | Audio and text leave the machine | Depends on the adapters you wire | Default path often a hosted LLM |
| **Barge-in / echo** | AEC3 default; `--barge-in` opt-in | Provider-managed | You assemble VAD, AEC, and cancel | Weak echo / self-interrupt on speakers |
| **Offline after cache** | Yes on the default local stack | No | Usually no | Possible after you assemble it |
| **HTTP voice server** | Not shipped (`serve` is out of scope) | Yes | Yes, if you run workers | Optional Realtime / WebRTC modes |

Longer notes: [docs/compare.md](docs/compare.md).

## Requirements

| | **Minimum (default stack)** | **Recommended** |
| --- | --- | --- |
| **OS** | Linux x64 · macOS (arm64 or x86_64) · Windows x64 | Current desktop OS with working mic and speakers |
| **RAM** | 8 GB (tight) | 16 GB+ (reference evidence is Apple M4 / 16 GB) |
| **Disk** | ~3 GB free for the default cache | SSD; more if you select larger Whisper / Qwen TTS |
| **GPU** | Optional | Apple Silicon Metal for Whisper and the local LLM |
| **Network** | First `run` only | None after the cache is full |
| **From source** | Rust 1.91+, CMake, C++ compiler; Linux: `libasound2-dev` | Same |

Qwen TTS backbones need several extra gigabytes of RAM. See [docs/requirements.md](docs/requirements.md) and [workload benchmarks](docs/workload_benchmark/workload_benchmark.md).

### Recommended stack by hardware

| Hardware | STT | LLM | TTS | Why |
| --- | --- | --- | --- | --- |
| **Apple Silicon 16 GB (default)** | `whisper-small` | `lfm2.5-2.6b` | `pocket-tts` | Best M4 16 GB combination for first-token consistency, throughput, and RTF |
| **CPU-only Linux / Windows** | `whisper-small` or Moonshine (English) | `lfm2.5-2.6b` or a smaller yaml LLM | `pocket-tts` | Portable CPU; skip Qwen TTS unless you have RAM to spare |
| **English streaming partials** | `moonshine-streaming-small` | unchanged | unchanged | Live partial transcript text; English-only |

## Engines

Ids are yaml `pipeline.*.model` values. Unknown ids fail at load. Sources and cache layout: [docs/contributing.md](docs/contributing.md#models). Capability notes: [docs/engines.md](docs/engines.md).

### Speech to text

| Id | Notes |
| --- | --- |
| `whisper-small` | Default. whisper.cpp; Metal on Darwin |
| `whisper-medium` / `whisper-large-v3-turbo` | Larger Whisper |
| `whisper-medium-q5_0` / `whisper-large-v3-turbo-q5_0` | Published quantizations |
| `moonshine-streaming-small` / `moonshine-streaming-medium` | English-only streaming ASR; live partials |

### Language models

| Id | Notes |
| --- | --- |
| `lfm2.5-2.6b` | Default local GGUF |
| `lfm2.5-350m` / `lfm2.5-230m` | Smaller LFM |
| `llama-3.2-1b` | Llama 3.2 1B Instruct |
| `qwen3.5-0.8b` / `qwen3.5-2b` | Qwen 3.5; `thinking: true` only on `qwen3.5-2b` |
| any id with `provider: online` | Endpoint-served chat model |

### Text to speech

| Id | Notes |
| --- | --- |
| `pocket-tts` | Default. English ONNX, fixed Alba voice (~150 MB) |
| `kokoro` | Kokoro ONNX + default voice |
| `qwen3-0.6` / `qwen3-1.7` | Qwen3-TTS; CPU on every OS |

VAD is Silero only.

## Architecture

```text
syllabix (clap CLI + ratatui TUI)
        │
syllabix-core  —  cpal I/O + AEC3
        │
        ├── Silero ONNX (VAD)
        ├── whisper.cpp / Moonshine ONNX (STT)
        ├── llama.cpp GGUF  or  OpenAI-compatible HTTP (LLM text only)
        └── Pocket TTS / Kokoro ONNX / Qwen TTS (TTS)
```

| Layer | Path | Responsibility |
| --- | --- | --- |
| CLI / TUI | `crates/syllabix/` | `run`, `init`, terminal UI |
| Runtime | `crates/syllabix-core/` | Pipeline, yaml, cache, AEC, adapters, optional harness |
| Native link | `crates/syllabix-native/` | whisper.cpp + llama.cpp (one ggml) |
| Weights | `~/.cache/syllabix/models/v1` | First-run HTTPS fetch, SHA-256, reuse offline |

There is no local HTTP server. Diagram and crate map: [docs/architecture.md](docs/architecture.md).

## API

Syllabix does **not** expose a speech HTTP API. `syllabix run` owns the microphone and speakers in-process.

The only network client on the product path is an **optional outbound** OpenAI-compatible LLM:

```yaml
pipeline:
  llm:
    provider: online
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1
```

```bash
SYLLABIX_LLM_API_KEY=sk-… ./syllabix run
```

Audio never leaves the machine on that path; only transcript text is posted to `chat/completions`. Keys are environment-only, never yaml. Details: [docs/api.md](docs/api.md) and [docs/syllabix-yaml.md](docs/syllabix-yaml.md).

## Docs

| Need | Read |
| --- | --- |
| Install | [Index](docs/install.md) · [macOS](docs/install/macos.md) · [Windows](docs/install/windows.md) · [Linux](docs/install/linux.md) |
| Use it | [Features](docs/features.md) · [syllabix.yaml](docs/syllabix-yaml.md) · [Troubleshooting](docs/troubleshooting.md) |
| Choose models | [Engines](docs/engines.md) · [Requirements](docs/requirements.md) · [Workload benchmarks](docs/workload_benchmark/workload_benchmark.md) |
| How it is built | [Architecture](docs/architecture.md) · [API posture](docs/api.md) · [Compare](docs/compare.md) |
| Measure and debug | [Diagnostics](docs/diagnostics.md) · [Reference profiles](docs/reference-profiles.md) |
| Contribute | [Contributing](docs/contributing.md) · [FAQ](docs/faq.md) · [Docs index](docs/README.md) |

## FAQ

**Does it need Python, Ollama, or an API key?**
No on the default path. One binary, first-run model cache, local LLM.

**Is there a `serve` / REST voice endpoint?**
No. Integrations that need a local OpenAI audio server are out of scope for this release.

**Why is the first run slow?**
It downloads ~2.2 GB and verifies SHA-256. Later runs are offline. Cold-start wall time is not published yet.

**The agent interrupts itself on laptop speakers.**
Wait for the ~10 s AEC calibration. If it continues, use headphones and file a bug with the device names from the startup line.

**Can I interrupt the agent?**
Yes, with `./syllabix run --barge-in`. Without the flag it finishes the sentence first.

More: [docs/faq.md](docs/faq.md).

## Optional config

[`syllabix init`](docs/syllabix-yaml.md) writes `syllabix.yaml` if you want a different Whisper size or language, English-only Moonshine streaming STT, local LLM id, spoken `system_prompt`, Kokoro or Qwen TTS (default remains Pocket TTS), or a BYO-key online LLM. A minimal example is [`examples/demo-agent.yaml`](examples/demo-agent.yaml). Full key guide: [docs/syllabix-yaml.md](docs/syllabix-yaml.md).

Per-turn timelines and WAVs are yaml-only (`diagnostics:`). See [docs/diagnostics.md](docs/diagnostics.md).

## Develop

Requires Rust 1.91+, CMake, and a C++ compiler. Linux also needs ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

From a checkout with a microphone: `cargo run -p syllabix -- run`. Full contributor path: [docs/contributing.md](docs/contributing.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Downloaded model weights keep their upstream terms.
