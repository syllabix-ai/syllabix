# CLI

Syllabix user commands: `run`, `init`, and `serve`.

```text
syllabix --help
syllabix run [--barge-in]
syllabix init [DIR]
syllabix serve stt [--bind 127.0.0.1:8091] [--model whisper-small]
syllabix serve tts [--bind 127.0.0.1:8092] [--model pocket-tts]
```

`--help` and `init` do not download weights. `serve` loads the selected STT or TTS id once (same menu as yaml).

## `run`

Start a local voice conversation. Zero config: no yaml file and no API key.

| Flag | Default | Effect |
| --- | --- | --- |
| (none) | — | Finish the current sentence before listening. |
| `--barge-in` | off | Keep VAD running during TTS; user speech stops playback. Press `b` in the terminal UI to toggle during a conversation. |

`syllabix.yaml` in the current working directory is loaded if present. Missing file is not an error. Diagnostics are yaml-only — there is no `--turn-debug` flag. See [diagnostics](diagnostics.md).

Online stages read keys from the environment at `run` start: `SYLLABIX_LLM_API_KEY`, `SYLLABIX_STT_API_KEY`, `SYLLABIX_TTS_API_KEY`. Never yaml. See [syllabix.yaml](syllabix-yaml.md#local-vs-online).

## `init [DIR]`

Write `syllabix.yaml` with launch defaults. `DIR` defaults to `.`. Refuses to overwrite an existing file. Keys: [syllabix.yaml](syllabix-yaml.md).

## `serve`

Model daemon for a fleet of desks: one weight cache, many `run` endpoints. Capture, AEC, VAD, barge-in, and playback stay on the device with the human. Default `run` is unchanged.

```text
syllabix serve stt [--bind 127.0.0.1:8091] [--model whisper-small]
syllabix serve tts [--bind 127.0.0.1:8092] [--model pocket-tts]
```

| | Default | Notes |
| --- | --- | --- |
| Bind | loopback (`127.0.0.1:8091` STT, `:8092` TTS) | Not a public-internet default. Reach other machines with a tunnel or Tailscale, still pointing yaml `base_url` at `/v1`. |
| Token | `SYLLABIX_SERVE_TOKEN` | `Authorization: Bearer …`. Required unless the bind is loopback and the variable is unset (dev only). Endpoints set `SYLLABIX_STT_API_KEY` / `SYLLABIX_TTS_API_KEY` to the same value. |
| Model | yaml menu id | Loaded once at start. STT: Whisper / Qwen3-ASR (`POST /v1/audio/transcriptions`, 16 kHz WAV). TTS: Pocket / Kokoro / Qwen (`POST /v1/audio/speech`; raw PCM is 24 kHz mono like OpenAI Audio; WAV carries 16 kHz in the header). `stream_format=audio` is chunked PCM for any of those ids. Moonshine turns are not on this worker yet. |
| Health | `GET /health` | Model id and busy/idle. |

Point `pipeline.stt` / `pipeline.tts` `provider: online` at the worker. Hosted providers (Groq, OpenAI Audio, whisper-server, Kokoro-FastAPI): [engines](engines.md). Yaml: [local vs online](syllabix-yaml.md#local-vs-online).

## Hidden commands

`bench`, the `bench-*-worker` helpers, and `vad-lab` are contributor-only tools. They are hidden from `--help`. See [component performance evidence](contributing.md#contribute-component-performance-evidence) and the [VAD endpointing lab](contributing.md#vad-endpointing-lab).
