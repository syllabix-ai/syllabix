# Syllabix

[![Version](https://img.shields.io/badge/version-v0.0.2-blue)](https://github.com/syllabix-ai/syllabix/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)
[![Linux](https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black)](./docs/install.md)
[![macOS](https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white)](./docs/install.md)
[![Windows](https://img.shields.io/badge/Windows-0078D4)](./docs/install.md)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange?logo=rust)](./rust-toolchain.toml)

Talk to a voice agent on your computer. One native binary. Easy to set up. Apache-2.0.

```text
mic → VAD → STT → LLM → TTS → speakers
              ↑______________ barge-in (interrupt + flush)
```

Download `syllabix`, run it, and talk. You do not need Python, pip, or an API key for the default setup. Your audio stays on your machine.

Default models: Whisper `small` (STT), LFM2.5-350M (local small LLM), and Pocket TTS (TTS).

![`syllabix run` terminal UI](docs/assets/tui-run.png)

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

The first time you run it, Syllabix downloads the models (Silero, Whisper `small`, LFM2.5-350M, and Pocket TTS — about 0.9 GB) into `~/.cache/syllabix/models/v1` (or `%LOCALAPPDATA%\syllabix\cache\models\v1` on Windows). You can change that folder with `$SYLLABIX_CACHE_DIR`.

Blocked by macOS Gatekeeper or Windows SmartScreen? See [docs/install.md](docs/install.md).

## Build from source

You need Rust 1.91+, CMake, and a C++ compiler. On Linux, also install ALSA headers (`libasound2-dev`).

```bash
cargo run -p syllabix -- run
```

## Talk

`syllabix run` uses your microphone and speakers. Allow mic access when your system asks. Echo canceling is on by default.

```bash
./syllabix run
./syllabix run --barge-in   # cut off the agent when the user starts speaking
./syllabix init             # optional: create syllabix.yaml — see docs/syllabix-yaml.md
```

Default `run` needs no config file and no API key. Once models are downloaded, it needs no internet.

Full command list: [docs/cli.md](docs/cli.md).

## Optional config

You do not need a config file for the basic setup. Run [`syllabix init`](docs/syllabix-yaml.md) when you want to change the model. Language, voice, system prompt, online LLMs, and other options are in [docs/syllabix-yaml.md](docs/syllabix-yaml.md).

A starter file is in [`examples/demo-agent.yaml`](examples/demo-agent.yaml). More detail: [model list](docs/engines.md), [config guide](docs/syllabix-yaml.md), and [diagnostics](docs/diagnostics.md) (per-turn timelines and WAV files).

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
| [Security](SECURITY.md) | How to report security issues privately |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
