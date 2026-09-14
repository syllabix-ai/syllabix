# Configuration

Syllabix is **zero-config** by default: `syllabix run` needs no file and no API key.

Optional project settings live in **`syllabix.yaml`** in the current working directory.

| Doc | For |
| --- | --- |
| **[syllabix.yaml](syllabix-yaml.md)** | File identity, safe edits, annotated example, full key reference |
| [diagnostics.md](diagnostics.md) | Turn timelines and WAVs (`diagnostics:` block) |
| [install.md](install.md) | Checksums, model cache, Gatekeeper / SmartScreen |
| [troubleshooting.md](troubleshooting.md) | Mic, echo, keys, first-run fetch |

Create a starter file with `syllabix init [dir]`, or copy [`examples/demo-agent.yaml`](../examples/demo-agent.yaml).
