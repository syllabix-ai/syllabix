# Compare

Syllabix is a **local spoken-conversation runtime** packaged as one native binary: microphone → Silero VAD → STT → LLM → TTS → speakers. It integrates WebRTC AEC3 and optional barge-in so the first conversation needs no application code or configuration file.

It is not a telephony platform, a Python audio SDK, a model zoo, or a voice-cloning studio. The distinction is not whether the other stacks can run locally; it is whether a local spoken conversation is the default product experience.

This page is positioning, not a benchmark. End-to-end latency and interrupt p95 are not published yet ([reference profiles](reference-profiles.md)).

## At a glance

| | Syllabix | Hugging Face [speech-to-speech](https://github.com/huggingface/speech-to-speech) | Pipecat / LiveKit | Ollama (plus your own STT/TTS) |
| --- | --- | --- | --- | --- |
| **Job** | **Download a native binary and talk locally** | Experiment with a configurable speech-agent cascade | Build programmable production voice applications | Run a local **text** LLM |
| **First conversation** | `syllabix run` | Install a Python environment and select components | Build or configure an agent and transport | Assemble STT, LLM, TTS, and turn handling |
| **Runtime shape** | Single native runtime process | Python pipeline plus selected runtimes or providers | Agent framework plus transport or server runtime | Ollama process plus a separate audio stack |
| **Default pipeline** | Silero → whisper.cpp → llama.cpp → Pocket TTS | Many interchangeable backends | Application-selected providers and plugins | Text model only |
| **Where audio goes by default** | Audio and text stay local after the model cache is filled | Depends on the selected components and transport | Depends on the client, transport, and providers | LLM can stay local; the audio pipeline is yours |
| **Laptop speaker echo** | **WebRTC AEC3 integrated in the native runtime** | Browser or WebRTC paths can use client AEC | Usually handled by the client or transport path | You own the audio stack |
| **Interruptions** | `run --barge-in`: cancel playback, flush queued audio, and keep the new turn | Supported in realtime paths | Framework/runtime support; application behavior is configurable | You implement the conversation loop |
| **Configuration surface** | None for the first run; optional `syllabix.yaml` later | Broad backend and CLI surface | Application code plus framework configuration | Several independent tools and configurations |
| **WebRTC / SIP / server runtime** | No `serve`, SIP, or SFU in this repo today | Realtime and transport modes are available | **Core strength** | No conversation runtime |

## When to use which

**Syllabix** — you want to download one executable, grant microphone access, and start talking to a local agent without writing an audio pipeline. The default stack is intentionally opinionated; yaml is for changing models and providers later, not for getting the first conversation working.

**Hugging Face speech-to-speech** — you want a flexible reference or laboratory stack with interchangeable STT, LLM, TTS, realtime, and transport options. Syllabix uses the same broad VAD → STT → LLM → TTS pattern, but optimizes it for a native local conversation experience with a much smaller configuration surface.

**Pipecat or LiveKit** — you are building an application that needs browsers, WebRTC, telephony, transports, workers, multiple sessions, or production orchestration. Syllabix does not currently provide that transport or deployment infrastructure.

**Ollama** — you primarily want to run a local **text** model. Syllabix can call an OpenAI-compatible endpoint such as Ollama (`pipeline.llm.provider: online` with an Ollama `base_url`) while keeping STT/TTS local. Ollama is therefore an optional external LLM endpoint, not part of the default Syllabix runtime.

## What this page does not claim

- Faster than HF S2S or Pipecat. No public silence-end → first-audio number yet.
- Better interruption, quality, or privacy than another stack without a defined path and measurements.
- Feature parity with voice studios (cloning, dubbing, 10+ engines).
- HIPAA, GDPR-as-a-product, or stream-level PII redaction.
- Linux/Windows GPU. Darwin uses Metal for STT/LLM; Linux and Windows Release builds are portable CPU. Linux can opt into ggml-Vulkan (`SYLLABIX_GGML_VULKAN=1`) for whisper/llama/Qwen when a device is present.

Loop internals: [architecture](architecture.md). Model ids: [engines](engines.md). Optional yaml: [syllabix.yaml](syllabix-yaml.md).
