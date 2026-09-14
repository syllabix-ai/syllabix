# Comparison

Syllabix v0 is a **single-binary local voice agent**. It is not a replacement for a hosted speech API, a WebRTC SFU, or a frame-level Python SDK.

## Hosted realtime / speech APIs

OpenAI Realtime, Gemini Live, ElevenLabs Agents, and similar services win on managed quality and zero local disk. They require an account, send audio (or a live stream) off-machine, and bill per use.

Syllabix wins when the default path must work **offline after first fetch**, with **no API key**, and with echo/barge-in under your control. You supply RAM, disk, and a microphone.

## Pipeline SDKs (Pipecat, LiveKit Agents)

Those projects are the right tool when you are assembling STT, LLM, TTS, and transport yourself — often against Daily or LiveKit cloud.

Syllabix does not try to migrate that stack. The install is `curl` + `run`, not a worker fleet. If you need `serve`, telephony, or a React client protocol, this binary does not provide it.

## Hugging Face speech-to-speech

[huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) is the loop Syllabix distilled: Silero VAD → STT → LLM → TTS with cancel for barge-in.

What we cut for launch: OpenAI Realtime as the default, Python/CUDA install notes, and a large matrix of backends. What we added as product: in-process Rust, one ggml for whisper.cpp + llama.cpp, AEC3 on by default, a Release binary, and a first-run SHA-256 cache.

## Voice studios (cloning, dubbing, catalogues)

Apps such as VoiceStudio are **creation tools**: clone, dub, catalogue many engines, often with a local HTTP API.

Syllabix is a **conversation runtime**: one blessed default stack, optional yaml ids, no engine marketplace and no inbound speech HTTP API. The README matches that product, not a 16-engine catalogue.

## Summary

| Need | Prefer |
| --- | --- |
| Talk to a local agent with one file | Syllabix |
| Highest hosted voice quality, no weights | Hosted realtime API |
| Custom transport, phones, multi-agent graphs | Pipecat / LiveKit (or later Syllabix scope) |
| Clone voices / dub video locally | A studio app, not this repo |
| Research many STT/TTS backends in Python | HF speech-to-speech |
