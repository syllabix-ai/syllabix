<p align="center">
  <img src="docs/assets/logo.png" alt="Syllabix" width="220">
</p>

[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://github.com/syllabix-ai/syllabix/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](./docs/install.md)
[![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)](./docs/install.md)
[![Windows](https://img.shields.io/badge/Windows-0078D4)](./docs/install.md)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange?logo=rust)](./rust-toolchain.toml)

Talk to a voice agent on your computer.

Syllabix is a Rust local-first voice agent (library + binary). One native binary, on-device speech, and real-time interruption. No Python or API key required for the default setup. It is a local voice agent for local speech-to-speech conversation on your machine.

```text
mic → VAD → STT → LLM / agentic LLM (⇄ sandboxed shell, tool exec) → TTS → speakers
              ↑________________________ barge-in (interrupt + flush)
```

Download `syllabix`, run it, and talk. You do not need Python, pip, or an API key for the default setup. Your audio stays on your machine.

Default models: Whisper `small` (STT), LFM2.5-2.6B (local small LLM), and Pocket TTS (TTS).

## Download

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

| Computer | File to download |
| --- | --- |
| Linux x64 | `syllabix-Linux-x86_64` |
| Mac with Apple chip | `syllabix-Darwin-arm64` |
| Mac with Intel chip | `syllabix-Darwin-x86_64` |
| Windows x64 | `syllabix-Windows-x86_64.exe` |

Blocked by macOS Gatekeeper or Windows SmartScreen? See [docs/install.md](docs/install.md).

## Run

### Fully local

No API key. Default `run` uses on-device Whisper, LFM2.5-2.6B, and Pocket TTS.

```bash
./syllabix run
```

The first local run downloads the models (Silero, Whisper `small`, LFM2.5-2.6B, and Pocket TTS — about 2.2 GB) into `~/.cache/syllabix/models/v1` (or `%LOCALAPPDATA%\syllabix\cache\models\v1` on Windows). You can change that folder with `$SYLLABIX_CACHE_DIR`. Once cached, it needs no internet.

![Fully local `syllabix run` terminal UI](docs/assets/tui-run.png)

### OpenAI LLM

Uses OpenAI `gpt-5.4` with the developer harness on. Needs `SYLLABIX_LLM_API_KEY`. STT and TTS stay local; only transcript text goes to OpenAI. The [developer harness](docs/syllabix-yaml.md#developer_harness) can also be enabled for a local model, but accuracy can be poor and the model can be slow.

```bash
curl -L https://raw.githubusercontent.com/syllabix-ai/syllabix/main/examples/demo-open-ai-llm-voice-agent.yaml -o syllabix.yaml
SYLLABIX_LLM_API_KEY=sk-… ./syllabix run
```

![OpenAI LLM `syllabix run` terminal UI](docs/assets/tui-run-openai.png)

### Talk

`syllabix run` uses your microphone and speakers. Allow mic access when your system asks. Echo canceling is on by default.

```bash
./syllabix run --barge-in   # cut off the agent when the user starts speaking
./syllabix init             # optional: create syllabix.yaml — see docs/syllabix-yaml.md
```

Full command list: [docs/cli.md](docs/cli.md).

## Supported models

VAD is always `silero`. Default stack is `whisper-small` + `lfm2.5-2.6b` + `pocket-tts`.

| Stage | Models |
| --- | --- |
| STT | `whisper-small` (default), `whisper-medium`, `whisper-large-v3-turbo`, `whisper-medium-q5_0`, `whisper-large-v3-turbo-q5_0`, `moonshine-streaming-small`, `moonshine-streaming-medium`, `qwen3-asr-0.6` |
| Local LLM | `lfm2.5-2.6b` (default), `lfm2.5-350m`, `lfm2.5-230m`, `llama-3.2-1b`, `qwen3.5-0.8b`, `qwen3.5-2b` |
| Online LLM | Any OpenAI-compatible endpoint, e.g. model `gpt-5.4` at `https://api.openai.com/v1` (needs `SYLLABIX_LLM_API_KEY`) |
| TTS | `pocket-tts` (default), `kokoro`, `qwen3-0.6`, `qwen3-1.7` |

Sources, download sizes, and machine needs: [docs/engines.md](docs/engines.md). Config keys: [docs/syllabix-yaml.md](docs/syllabix-yaml.md).

## Build from source

See [Compile](docs/install.md#compile).

## Optional config

You do not need a config file for the basic setup. Run [`syllabix init`](docs/syllabix-yaml.md) when you want to change the model. Language, voice, system prompt, online LLMs, and other options are in [docs/syllabix-yaml.md](docs/syllabix-yaml.md).

A starter file is in [`examples/demo-fully-self-hosted-voice-agents.yaml`](examples/demo-fully-self-hosted-voice-agents.yaml). More detail: [model list](docs/engines.md), [config guide](docs/syllabix-yaml.md), and [diagnostics](docs/diagnostics.md) (per-turn timelines and WAV files).

## Docs

| Page | What it covers |
| --- | --- |
| [Docs index](docs/README.md) | User and config map |
| [Install](docs/install.md) | Checksums, cache, Gatekeeper, SmartScreen, building |
| [CLI](docs/cli.md) | `run`, `--barge-in`, `init`, `--help` |
| [Engines](docs/engines.md) | VAD / STT / LLM / TTS ids and machine needs |
| [Architecture](docs/architecture.md) | How the pipeline works, barge-in, what leaves your machine |
| [Compare](docs/compare.md) | vs Hugging Face S2S, Pipecat/LiveKit, Ollama |
| [Configuration](docs/configuration.md) | Index of yaml, engines, diagnostics |
| [syllabix.yaml](docs/syllabix-yaml.md) | Optional project file — keys and safe edits |
| [Troubleshooting](docs/troubleshooting.md) | Mic, echo, keys, first-run download |
| [FAQ](docs/faq.md) | Large first run, barge-in, API keys |
| [Contributing](docs/contributing.md) | Tests, packaging, CI |
| [Issues](docs/issues.md) | Bugs, features, and security reports |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
