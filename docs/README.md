# Documentation

Syllabix documentation follows the same coverage as other local-voice inspiration repos: install per OS, features, comparison, requirements, engines, architecture, API posture, and FAQ. Product behavior lives in these pages; measurement protocols stay in [reference-profiles.md](reference-profiles.md).

## Install and run

| Page | For |
| --- | --- |
| [Install](install.md) | Checksums, cache, OS index |
| [macOS](install/macos.md) | Gatekeeper, microphone, Apple Silicon vs Intel |
| [Windows](install/windows.md) | SmartScreen, microphone |
| [Linux](install/linux.md) | Binary path, ALSA for contributors |
| [Troubleshooting](troubleshooting.md) | Mic, echo, keys, first-run fetch |

## Product surface

| Page | For |
| --- | --- |
| [Features](features.md) | What the binary does and does not do |
| [Compare](compare.md) | Hosted APIs, pipeline SDKs, HF speech-to-speech |
| [Requirements](requirements.md) | RAM, disk, GPU, recommended stacks |
| [Engines](engines.md) | STT / LLM / TTS / VAD ids |
| [Architecture](architecture.md) | Crate map and voice loop |
| [API](api.md) | No inbound speech server; optional outbound LLM |
| [FAQ](faq.md) | Common questions |

## Configuration and debug

| Page | For |
| --- | --- |
| [syllabix.yaml](syllabix-yaml.md) | Optional project file — keys and safe edits |
| [Configuration](configuration.md) | Short index of config-related docs |
| [Diagnostics](diagnostics.md) | Turn timelines and WAVs |

## Contributors

| Page | For |
| --- | --- |
| [Contributing](contributing.md) | Build, test, packaging, models |
| [Reference profiles](reference-profiles.md) | How conversation quality is measured |
| [Workload benchmark](workload_benchmark/workload_benchmark.md) | Per-model STT / LLM / TTS timings |
| [Proposals](proposals/skills-and-capability-sandboxed-shell.md) | Design not yet product behavior |
