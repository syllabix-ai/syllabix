# Architecture

Syllabix is one native binary that talks on the local microphone and speakers.

```text
mic → Silero VAD → STT → LLM → TTS → speakers
              ↑______________ barge-in (interrupt + flush)
```

| Stage | Default | Role |
| --- | --- | --- |
| Capture / playback | `cpal` | Built-in mic and speakers |
| Echo | WebRTC AEC3 (Sonora), on by default | Stops the agent from treating its own voice as a user turn. Calibrates ~10 s at start. |
| VAD | Silero ONNX | Speech start / end |
| STT | whisper.cpp `small` | Speech → text, in-process |
| LLM | llama.cpp + LFM2.5-350M | Text → reply, in-process |
| TTS | Pocket TTS ONNX | Reply → speech, in-process |

Ids other than the defaults are yaml-selected and fetch on first use — [engines](engines.md), [syllabix.yaml](syllabix-yaml.md).

## Barge-in

Default `run` finishes the current sentence before listening. `run --barge-in` keeps VAD running during TTS and cancels playback on user speech (queued audio dropped, LLM/TTS cancelled, new turn kept).

## Cache

First use of each model id downloads pinned weights over HTTPS, checks SHA-256, and stores them under the cache root (`$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1`). Later default `run`s reuse that cache with no network.

## What leaves the machine

- **Default local stack:** audio and text stay on the machine after the weight cache is filled.
- **`pipeline.llm.provider: online`:** only transcript text is sent to the OpenAI-compatible endpoint. Mic audio is not uploaded. The key comes from `SYLLABIX_LLM_API_KEY`, never from yaml.

## Layout (contributors)

| Path | Responsibility |
| --- | --- |
| `crates/syllabix` | CLI, TUI, `run` / `init` |
| `crates/syllabix-core` | Pipeline, audio I/O, VAD/STT/LLM/TTS, yaml, sandbox |
| `vendor/` | ggml / llama.cpp / whisper.cpp pins |
| `docs/` | This documentation tree |

How we measure the loop: [reference profiles](reference-profiles.md).
