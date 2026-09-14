# API posture

VoiceStudio-style inspiration repos document a local OpenAI-compatible **audio** server. Syllabix does not have one.

## What exists

| Surface | Status |
| --- | --- |
| `syllabix run` / `init` | Shipped CLI |
| Inbound REST / WebSocket / MCP speech server | **Not shipped.** There is no `serve` |
| Outbound OpenAI-compatible **LLM** | Opt-in yaml `pipeline.llm.provider: online` |
| Environment API key | `SYLLABIX_LLM_API_KEY` only — never yaml, never `.env` |

## Outbound LLM

Audio stays on the machine. The client sends chat messages built from the transcript (and optional harness tool results) to `base_url`.

```yaml
pipeline:
  llm:
    provider: online
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1
```

```bash
SYLLABIX_LLM_API_KEY=sk-… syllabix run
```

Examples for `base_url`: `https://api.openai.com/v1`, `https://api.groq.com/openai/v1`, `http://127.0.0.1:11434/v1` (Ollama). Keyless loopback still needs a placeholder value (`SYLLABIX_LLM_API_KEY=ollama`). Missing key with `provider: online` fails before devices or weights load.

Failed cloud turns speak a fallback instead of hanging. Diagnostics sidecars record endpoint and request id when enabled.

Full keys: [syllabix-yaml.md](syllabix-yaml.md).

## Developer harness

Not a public HTTP API. Yaml `developer_harness: true` (online LLM or local `lfm2.5-2.6b`) lets the model use a generic sandboxed shell. Admission evals are documented in [contributing.md](contributing.md).

## Why this page exists

Readers coming from studio apps should not hunt for `/v1/audio/speech`. Point those integrations at a different local server, or wait until a later Syllabix sequence explicitly adds `serve`.
