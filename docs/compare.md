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

## Rust voice-agent stacks

If you searched for a Rust **voice agent**, **speech-to-speech** loop, **whisper** STT, **barge-in**, or **AEC**, you likely saw these repos before Syllabix. This section positions them; it is not a benchmark and it does not rank latency, accuracy, or audio quality.

| | Syllabix | [Skadoosh](https://github.com/Hot-Coco/Skadoosh) | [rustvani](https://github.com/Allenmylath/rustvani) | [Flowcat](https://github.com/AreevAI/flowcat) | [vox](https://github.com/mrtozner/vox) |
| --- | --- | --- | --- | --- | --- |
| **Job** | **Download a native binary and talk locally** | Fully local voice agent as a binary or a library with pluggable engines | Pipecat-style pipeline framework for production voice deployments | Self-hosted call runtime you deploy in your own VPC | Local STT/TTS toolkit plus CLI, voice chat, and a web UI |
| **First conversation** | `syllabix run` (models fetch about 2.2 GB on first run) | `cargo install`, repo model-download script, plus an Ollama endpoint | Build the `quickstart` example; bring Sarvam / OpenAI / Deepgram keys | Build `flowcat-server`, write a yaml config, bring one provider key, talk in the browser | `cargo install --git` with features; models auto-download; `vox chat` needs Ollama |
| **Runtime shape** | Single native process plus the `syllabix-core` SDK lanes | `Agent` builder library plus a binary; audio behind a feature | Tokio `FrameProcessor` library plus examples and bins; providers behind features | Workspace of library crates plus `flowcat-server` / CLI; providers and transports behind features | Async `Vox` builder library plus a `vox` CLI and an HTTP/WebSocket server with an embedded UI |
| **Default pipeline** | Silero → whisper.cpp → llama.cpp → Pocket TTS | Silero VAD → Whisper → LLM → Kokoro clause streaming | WebSocket transport → VAD → Sarvam / Deepgram STT → OpenAI / Sarvam LLM → Sarvam / Deepgram / Piper TTS | Transport → VAD / turn-taking → STT · LLM · TTS or a single speech-to-speech model | Silero → Whisper / Sherpa → optional speaker id → Ollama LLM → Kokoro / Piper / Qwen3 / Pocket / Chatterbox |
| **Where audio goes by default** | Audio and text stay local after the model cache is filled | Local by default; OpenAI-compatible endpoints also supported | Depends on the selected providers | Stays in infrastructure you control; you bring provider credentials | Fully local and offline after download; Ollama chat stays local when the endpoint is local |
| **Laptop speaker echo** | **WebRTC AEC3 integrated in the native runtime** | Not positioned as a laptop AEC box | In-process denoise and AGC chain; no stated laptop speaker-canceller | Transport-side; not a laptop mic/speaker AEC box | Not stated; not positioned as a laptop AEC box |
| **Interruptions** | `run --barge-in`: cancel playback, flush queued audio, and keep the new turn | Barge-in documented (speak anytime, millisecond stop) | `allow_interruptions` plus `InterruptionFrame` | Pipeline interruption model (see their processor design) | Live Talk experimental barge-in |
| **Configuration surface** | None for the first run; optional `syllabix.yaml` later | `Config` plus flags and env vars | Rust pipeline assembly plus provider configs and env keys | yaml/JSON graph plus env provider keys, or a Rust embedder | CLI flags plus auto-downloaded models and Cargo feature flags |
| **Phone / browser transport** | No `serve`, SIP, or SFU in this repo today | Text chat, repl, and selftest examples; no SIP/SFU positioning | WebSocket plus opt-in P2P WebRTC and a Twilio serializer | **Core strength** (in-process SIP/RTP, WebRTC, carrier serializers) | HTTP/WebSocket server plus a browser UI; no SIP/SFU positioning |

**Syllabix** — you want the laptop conversation to be the product: download, `run`, talk, with laptop echo cancellation and barge-in already wired. Library use ([embed](embed.md)) is additive; the binary stays the default.

**Skadoosh** — you want a Rust local voice-agent framework with a small `Agent` builder, clause-level TTS streaming, barge-in, and mock engines for tests. Its README positions fully-local operation with OpenAI-compatible backends as an option. Check whether its default LLM path expects your own Ollama endpoint before comparing first-run weight downloads.

**rustvani** — you want Pipecat's frame/pipeline mental model in Rust for server-side agents: WebSocket transport, pluggable STT/LLM/TTS services, a Dhara flow engine, and phone-facing pieces such as a Twilio serializer. It is a framework you assemble per connection with provider keys, not a download-and-talk laptop binary.

**Flowcat** — you want to own the call infrastructure: one self-contained binary in your VPC or air-gapped, carrying phone or WebRTC audio through a composable media pipeline with in-process SIP/RTP. It optimizes for call density and transport breadth, not for opening a laptop mic and talking with no config.

**vox** — you want local speech building blocks with many voices: transcribe (`listen`), synthesize (`speak`), voice chat against Ollama (`chat`), speaker diarization, and a `serve` mode with a browser UI and HTTP/WebSocket APIs. Its Live Talk barge-in is marked experimental; compare its toolkit breadth against Syllabix's single opinionated loop.

Syllabix does not compete with Sonora as an AEC crate; it embeds WebRTC AEC3 (via Sonora) inside a full local conversation runtime. See [architecture](architecture.md) for the loop and [embed](embed.md) for the SDK lanes.

## What this page does not claim

- Faster than HF S2S or Pipecat. No public silence-end → first-audio number yet.
- Better interruption, quality, or privacy than another stack without a defined path and measurements.
- A ranking of the Rust stacks above. Their READMEs move; check Skadoosh, rustvani, Flowcat, and vox for current features, defaults, and keys.
- Feature parity with voice studios (cloning, dubbing, 10+ engines).
- HIPAA, GDPR-as-a-product, or stream-level PII redaction.
- Linux/Windows GPU. Darwin uses Metal for STT/LLM; Linux and Windows Release builds are portable CPU. Linux can opt into ggml-Vulkan (`SYLLABIX_GGML_VULKAN=1`) for whisper/llama/Qwen when a device is present.

Loop internals: [architecture](architecture.md). Model ids: [engines](engines.md). Optional yaml: [syllabix.yaml](syllabix-yaml.md).
