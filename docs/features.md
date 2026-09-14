# Features

What the shipped `syllabix` binary does. Scope that is not here is not a missing README section — it is not in the product.

## Spoken conversation

- One process owns the microphone and speakers (`syllabix run`).
- Pipeline: Silero VAD → STT → LLM → TTS, with bounded queues between stages.
- Empty or whitespace transcripts never start LLM or TTS.
- Markdown and Qwen `<think>…</think>` are stripped before TTS. Thinking stays off unless yaml sets `pipeline.llm.thinking: true` on `qwen3.5-2b`.

## Echo and barge-in

- WebRTC AEC3 is **on by default** and calibrates for about 10 seconds.
- Default `run` finishes the current sentence before listening again.
- `syllabix run --barge-in` keeps VAD running during TTS, stops playback on SpeechStart, drops queued audio, and cancels the in-flight LLM/TTS turn. A 200 ms post-AEC preroll is prepended to the Whisper utterance so the first words are not clipped.

## Install and defaults

- One GitHub Release artifact per OS. No Python, pip, or Ollama on the user path.
- Zero-config defaults: Whisper `small`, LFM2.5-2.6B, Pocket TTS.
- First `run` fetches those weights (~2.2 GB), verifies SHA-256, and reuses `~/.cache/syllabix/models/v1` offline.
- `syllabix init` is optional.

## Optional yaml

Covered in [syllabix-yaml.md](syllabix-yaml.md):

- STT model and language (including English-only Moonshine streaming partials)
- Local LLM id, spoken `system_prompt`, Qwen thinking
- TTS engine
- Online LLM (`provider: online`) with environment key
- Idle mic-mute / exit timers
- Diagnostics sidecars and WAVs
- Developer harness + local skills (opt-in)

## Not in this release

These are documented so the README is not mistaken for a hosted-platform spec:

- No `serve`, WebSocket, WebRTC, or OpenAI-compatible **audio** server
- No SIP / telephony
- No voice cloning, dubbing, or speaker diarization
- No Homebrew / PyPI as the user install
- No second live STT/TTS cloud provider (online LLM is text-only)

See [compare.md](compare.md) for how this sits next to those products.
