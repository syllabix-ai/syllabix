# Configuration

`syllabix run` needs no file and no API key. Optional project settings live in **`syllabix.yaml`** in the current working directory.

| Doc | For |
| --- | --- |
| [syllabix.yaml](syllabix-yaml.md) | File identity, keys, safe edits, annotated example |
| [Engines](engines.md) | Model ids you can put in yaml |
| [Hardware](hardware.md) | Disk, RAM, CPU vs Metal |
| [CLI](cli.md) | `run`, `--barge-in`, `init` (barge-in is a flag, not yaml) |
| [Diagnostics](diagnostics.md) | `diagnostics:` timelines and WAVs |
| [Install](install.md) | Cache directory, Gatekeeper / SmartScreen |

Create a starter file with `syllabix init [dir]`, or copy [`examples/demo-agent.yaml`](../examples/demo-agent.yaml).
