# Syllabix

Local voice agent. One native binary. Apache-2.0.

```text
mic → VAD → STT → LLM → TTS → speakers
              ↑______________ barge-in (interrupt + flush)
```

Download `syllabix`, run it, talk. No Python, pip, or API key on the default path. Audio stays on the machine unless you opt into an online LLM (transcript text only).

Default models: Whisper `small`, LFM2.5-2.6B, and Pocket TTS (~2.2 GB on first run).

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

| Target | Artifact |
| --- | --- |
| Linux x64 | `syllabix-Linux-x86_64` |
| macOS Apple Silicon | `syllabix-Darwin-arm64` |
| macOS Intel | `syllabix-Darwin-x86_64` |
| Windows x64 | `syllabix-Windows-x86_64.exe` |

First `run` fetches Silero, Whisper `small`, LFM2.5-2.6B, and Pocket TTS (~2.2 GB) into `~/.cache/syllabix/models/v1` (`%LOCALAPPDATA%\syllabix\cache\models\v1` on Windows; override with `$SYLLABIX_CACHE_DIR`) and verifies SHA-256. Later runs reuse that cache offline. `--help` and `init` do not need the cache.

macOS Gatekeeper and Windows SmartScreen: [docs/install.md](docs/install.md).

## Compile

Requires Rust 1.91+, CMake, and a C++ compiler. Linux also needs ALSA headers (`libasound2-dev`).

```bash
cargo run -p syllabix -- run
```

Fmt, clippy, tests, and packaging: [docs/contributing.md](docs/contributing.md).

## Talk

`syllabix run` uses the built-in microphone and speakers. Grant microphone access when the OS asks. Echo control (WebRTC AEC3) is on by default and calibrates for about 10 seconds — wait before speaking. If laptop speakers still trigger the agent as if you were talking, use headphones and include the device names from the startup line in a bug report.

```bash
./syllabix run              # default: finish the sentence before listening
./syllabix run --barge-in   # stop playback when you talk over the agent
./syllabix init             # optional: write syllabix.yaml in a project folder
```

Default `run` needs no yaml and no API key. After the cache is full, it needs no network.

Commands: [docs/cli.md](docs/cli.md).

## Optional config

`syllabix run` needs no configuration file. Run [`syllabix init`](docs/syllabix-yaml.md) when you want to change the model, language, voice, system prompt, or use a BYO-key online LLM. Online LLMs receive transcript text only; audio stays on the machine.

Start with [`examples/demo-agent.yaml`](examples/demo-agent.yaml). See the [model catalogue](docs/engines.md), [configuration guide](docs/syllabix-yaml.md), and [diagnostics](docs/diagnostics.md) for per-turn timelines and WAVs.

## Docs

| Page | For |
| --- | --- |
| [Docs index](docs/README.md) | User, config, and contributor map |
| [Install](docs/install.md) | Checksums, cache, Gatekeeper, SmartScreen, build |
| [CLI](docs/cli.md) | `run`, `--barge-in`, `init`, `--help` |
| [Engines](docs/engines.md) | VAD / STT / LLM / TTS ids and machine needs |
| [Architecture](docs/architecture.md) | Cascade, AEC, barge-in, what leaves the machine |
| [Compare](docs/compare.md) | vs HF S2S, Pipecat/LiveKit, Ollama |
| [Configuration](docs/configuration.md) | Index of yaml, engines, diagnostics |
| [syllabix.yaml](docs/syllabix-yaml.md) | Optional project file — keys and safe edits |
| [Troubleshooting](docs/troubleshooting.md) | Mic, echo, keys, first-run fetch |
| [FAQ](docs/faq.md) | Large first run, barge-in, keys |
| [Contributing](docs/contributing.md) | Tests, packaging, CI |
| [Security](SECURITY.md) | Private vulnerability reports |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
