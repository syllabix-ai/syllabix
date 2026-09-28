# Docs

## Use

| Page | For |
| --- | --- |
| [Install](install.md) | Download, checksums, cache, Gatekeeper / SmartScreen, compile |
| [CLI](cli.md) | `run`, `--barge-in`, `init`, `--help` |
| [Engines](engines.md) | Shipped model ids, fetch-on-use, machine needs |
| [Architecture](architecture.md) | Cascade, AEC, barge-in, network boundary |
| [Compare](compare.md) | vs HF S2S, Pipecat / LiveKit, Ollama |
| [Troubleshooting](troubleshooting.md) | Symptom → fix |
| [FAQ](faq.md) | Large first run, barge-in, keys |

## Configure

| Page | For |
| --- | --- |
| [Configuration](configuration.md) | Index of yaml-related pages |
| [syllabix.yaml](syllabix-yaml.md) | Optional project file — keys and safe edits |
| [Diagnostics](diagnostics.md) | Turn timelines and WAVs |

`syllabix run` needs no yaml. Create a starter file with `syllabix init [dir]`, or copy [`examples/demo-agent.yaml`](../examples/demo-agent.yaml).

## Build / measure

| Page | For |
| --- | --- |
| [Contributing](contributing.md) | Build, test, packaging, CI |
| [Reference profiles](reference-profiles.md) | How conversation quality is measured |
| [Workload benchmark](workload_benchmark/workload_benchmark.md) | Per-model STT / LLM / TTS timings |
| [Issues](issues.md) | Bugs, features, and security reports |
