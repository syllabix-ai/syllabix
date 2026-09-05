# Syllabix

Local voice agent. One native binary. Apache-2.0.

```text
mic → Silero VAD → STT → LLM → TTS → speakers
              ↑______________ barge-in (interrupt + flush)
```

Download `syllabix`, run it, talk. No Python, pip, or API key on the default path.

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

First `run` fetches Silero, Whisper `small`, LFM2.5-2.6B, and Pocket TTS (~2.2 GB) into `~/.cache/syllabix/models/v1` (`%LOCALAPPDATA%\syllabix\cache\models\v1` on Windows; override the cache root with `$SYLLABIX_CACHE_DIR`) and verifies SHA-256. Later runs reuse that cache offline. `--help` and `init` do not need the cache. The default is the best M4 16 GB combination for minimal delight (see `docs/reference-profiles.md`).

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

## Optional config

[`syllabix init`](docs/configuration.md) writes `syllabix.yaml` if you want a different Whisper size or language, English-only Moonshine streaming STT (`moonshine-streaming-small` or `moonshine-streaming-medium`, live partial transcript text), local LLM id, spoken `system_prompt`, Kokoro or Qwen TTS, or a BYO-key online LLM (transcript text only; audio stays on the machine). A minimal example is [`examples/demo-agent.yaml`](examples/demo-agent.yaml).

Per-turn timelines and WAVs are yaml-only (`diagnostics:`). See [docs/diagnostics.md](docs/diagnostics.md).

## Docs

| Page | For |
| --- | --- |
| [Install](docs/install.md) | Checksums, cache, Gatekeeper, SmartScreen |
| [Configuration](docs/configuration.md) | `syllabix.yaml` keys |
| [Troubleshooting](docs/troubleshooting.md) | Mic, echo, keys, first-run fetch |
| [Diagnostics](docs/diagnostics.md) | Turn timelines and WAVs |
| [Contributing](docs/contributing.md) | Build, test, packaging |
| [Reference profiles](docs/reference-profiles.md) | How we measure conversation quality |

## Develop

Requires Rust 1.91+, CMake, and a C++ compiler. Linux also needs ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

From a checkout with a microphone: `cargo run -p syllabix -- run`. Full contributor path: [docs/contributing.md](docs/contributing.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
