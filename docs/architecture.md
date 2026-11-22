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
| LLM | llama.cpp + LFM3-1.7B | Text → reply, in-process |
| TTS | Pocket TTS ONNX | Reply → speech, in-process |

Ids other than the defaults are yaml-selected and fetch on first use — [engines](engines.md), [syllabix.yaml](syllabix-yaml.md).

## Speech-to-speech (`pipeline.mode: s2s`)

Optional yaml replaces the STT → LLM → TTS chain with one in-process engine (`pipeline.s2s.model: lfm3-audio-2b`). Default `run` without that yaml stays cascade.

```text
mic → Silero VAD → LFM3-Audio S2S → speakers
              ↑______________ barge-in (interrupt + flush)
```

With `pipeline.s2s.reply_with: llm`, the chain is `mic → Silero VAD → LFM3 ASR → LLM → LFM3 TTS → speakers` on one shared LFM3 context (reset between ASR and TTS).

| Stage | Default | Role |
| --- | --- | --- |
| Capture / playback | `cpal` | Built-in mic and speakers |
| Echo | WebRTC AEC3 (Sonora), on by default | Same self-reply handling as cascade |
| VAD | Silero ONNX | Speech start / end — still commits each user turn |
| S2S | LFM3-Audio 2B (when selected) | Utterance audio → reply audio, in-process |

Turn-based, not full duplex. English-only experimental chat via this id; Indic and code-switched voice remain the cascade stack — [engines S2S](engines.md#s2s-speech-to-speech).

## Agent harness

Default `run` uses the built-in harness (`agent_harness.agent: syllabix`). The LLM may call a sandboxed shell before the spoken reply goes to TTS:

```text
mic → Silero VAD → STT → LLM ⇄ sandboxed shell → … → TTS → speakers
              ↑________________________________ barge-in (interrupt + flush)
```

Set `agent_harness.agent: none` for plain voice chat with no tools: one STT transcript goes to the LLM, then TTS.

[Skills](syllabix-yaml.md#skills--local-instruction-packages) are optional instruction packages (`SKILL.md` under configured roots). The harness prompt for `syllabix` and `goose` names each skill, summarizes it, and points at the file path (workspace-relative when the skill lives under the project); the model pulls in the body when it uses a skill. They do not run code and do not widen sandbox permissions. Tool policy and ceilings: [agent_harness.agent](syllabix-yaml.md#agent-backend-and-shell-environment-agent_harness).

## Barge-in

Default `run` finishes the current reply before listening again. `run --barge-in` keeps VAD running during playback and cancels the in-flight generation on user speech (queued audio dropped, S2S or cascade LLM/TTS cancelled, new turn kept).

## Cache

First use of each model id downloads pinned weights over HTTPS, checks SHA-256, and stores them under the cache root (`$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1`). Later default `run`s reuse that cache with no network.

## What leaves the machine

- **Default local stack:** audio and text stay on the machine after the weight cache is filled.
- **`pipeline.llm.provider: openai-compatible`:** only transcript text is sent to the OpenAI-compatible endpoint. Mic audio is not uploaded. The key comes from `SYLLABIX_LLM_API_KEY`, never from yaml.
- **`pipeline.stt.provider: online`:** each finished utterance WAV is posted to `POST /v1/audio/transcriptions`. VAD stays on-device. Key: `SYLLABIX_STT_API_KEY`.
- **`pipeline.tts.provider: online`:** each finished sentence is posted to `POST /v1/audio/speech`. Key: `SYLLABIX_TTS_API_KEY`.
- **`syllabix serve`:** weights live on the worker. Endpoints send the same utterance WAV / sentence text as hosted online STT/TTS. Mic capture does not leave the desk. [CLI serve](cli.md#serve).

## Layout (contributors)

| Path | Responsibility |
| --- | --- |
| `crates/syllabix` | CLI, TUI, `run` / `init` / `serve` |
| `crates/syllabix-core` | Pipeline, audio I/O, VAD/STT/LLM/TTS, yaml, sandbox |
| `crates/syllabix-stt` | `Stt` trait + Whisper, Qwen3-ASR, Moonshine, Hima-ASR, remote STT (features `whisper`, `qwen3-asr`, `moonshine`, `hima`, `remote`) |
| `crates/syllabix-tts` | `Tts` trait + Pocket TTS, Kokoro, Qwen3-TTS, remote TTS (features `pocket`, `kokoro`, `qwen3`, `remote`) |
| `crates/syllabix-native/vendor/` | ggml / llama.cpp / whisper.cpp pins |
| `docs/` | This documentation tree |

How we measure the loop: [reference profiles](reference-profiles.md).
