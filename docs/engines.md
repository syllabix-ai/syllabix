# Engines

The ids you can put in `syllabix.yaml`. Key syntax: [syllabix.yaml](syllabix-yaml.md). Measured timings: [workload benchmark](workload_benchmark/workload_benchmark.md). Licenses of fetched weights: [NOTICE](../NOTICE).

Weights download on first use of each id (HTTPS URL pinned in the binary, SHA-256 check, then reuse from cache). Default `run` fetches only the launch stack: Silero, Whisper `small`, LFM3-1.7B, Pocket TTS (~2.5 GB). `--help` and `init` download nothing.

Cache path (files land in `models/v1` under the cache root), first match wins:

- `$SYLLABIX_CACHE_DIR/models/v1` if `SYLLABIX_CACHE_DIR` is set (that variable is the cache root, not `models/v1`)
- else `$XDG_CACHE_HOME/syllabix/models/v1`
- else, on Windows, `%LOCALAPPDATA%\syllabix\cache\models\v1`
- else `~/.cache/syllabix/models/v1`
- else `.syllabix-cache/models/v1` (cwd)

Share downloads across the CLI and embed hosts by leaving the variable unset or pointing every process at the same root. Details: [embed — Cache](embed.md#cache).

## Machine needs

| | What ships / what is known |
| --- | --- |
| **Release artifacts** | Linux x64, macOS Apple Silicon, Windows x64. Intel Macs are not supported. |
| **Default cache** | ~2.5 GB on first `run` |
| **Compute** | Darwin: Metal + Accelerate for STT/LLM. Linux/Windows: portable CPU by default. Opt-in Linux Vulkan builds (`SYLLABIX_GGML_VULKAN=1`) offload whisper/llama/Qwen when a device is present. Qwen TTS/ASR stay CPU unless `compute` selects Metal/Vulkan (or Auto probes them). |
| **RAM** | Not a published gate. The default stack is the combination measured for an Apple M4 / 16 GiB class machine; see [reference profiles](reference-profiles.md). |
| **Network** | First fetch of each id needs HTTPS. After the cache is full, default `run` needs none. |

## VAD

| Id | Notes | Source |
| --- | --- | --- |
| `silero` | Only provider. Always in the default stack. | https://github.com/snakers4/silero-vad |

## STT

| Yaml `pipeline.stt.model` | Notes | Source |
| --- | --- | --- |
| `whisper-small` | Default | https://huggingface.co/ggerganov/whisper.cpp |
| `whisper-medium` | Larger Whisper | same |
| `whisper-large-v3-turbo` | Largest Whisper menu id | same |
| `whisper-medium-q5_0` | Published quantization | same |
| `whisper-large-v3-turbo-q5_0` | Published quantization | same |
| `moonshine-streaming-small` | English-only streaming ASR; live partials (~360 MB) | pinned ONNX export; see NOTICE |
| `moonshine-streaming-medium` | Larger English-only streaming ASR (~590 MB) | pinned ONNX export; see NOTICE |
| `qwen3-asr-0.6` | Qwen3-ASR 0.6B via llama.cpp mtmd; finalize only (~1.0 GB). Optional `pipeline.stt.compute` (auto/cpu/metal/vulkan), same placement as Qwen TTS. | https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF |
| `hima-asr` | Optional Indic / code-switched finalize-only ONNX. Strongest on Telugu, Tamil, Kannada, and Malayalam; not meant for English. Optional `style` is `native` or `mixed`. Gemma terms apply ([NOTICE](../NOTICE)); see the model card for quality caveats. | https://huggingface.co/Prahaladha/hima-asr |
| `fluid-speech-pro-gguf-q8_0` | Fluid Speech Pro through parakeet.cpp; finalize only, auto language (25 European languages) (~0.94 GB) | https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-gguf |

`pipeline.stt.provider: online` posts each finished utterance to OpenAI-compatible `POST /v1/audio/transcriptions`. No STT weights are fetched. Key: `SYLLABIX_STT_API_KEY`. A self-hosted worker: [`syllabix serve stt`](cli.md#serve).

## LLM

Local menu (unknown ids fail at load):

| Yaml `pipeline.llm.model` | Notes | Source |
| --- | --- | --- |
| `lfm3-1.7b` | Default; first-run cache (~1.85 GB) | https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF |
| `lfm3-1.7b-bf16` | Fetched when selected (~3.5 GB) | same |
| `lfm2.5-2.6b` | Fetched when selected | https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF |
| `lfm2.5-350m` | Fetched when selected | https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF |
| `lfm2.5-230m` | Fetched when selected | https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF |
| `llama-3.2-1b` | Fetched when selected | https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF |
| `qwen3.5-0.8b` | Fetched when selected | https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF |
| `qwen3.5-2b` | Fetched when selected; only id that may set `thinking: true` | https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF |

`pipeline.llm.provider: openai-compatible` uses any id the OpenAI-compatible endpoint serves. Audio stays on the machine; only transcript text leaves. The API key is `SYLLABIX_LLM_API_KEY` (environment only, never yaml).

`agent_harness.agent: goose` downloads a pinned Goose CLI for this OS/arch into the model cache and checks its SHA-256 before unpacking. Syllabix writes Goose's provider and model from `pipeline.llm` into `{SYLLABIX_CACHE_DIR}/goose/config/config.yaml` before each run; see Goose's [provider docs](https://goose-docs.ai/docs/getting-started/providers/). On Linux and macOS, unpacking needs the `bzip2` program. `agent_harness.agent: none` and `syllabix` do not download Goose. Goose bundled with the pinned CLI cannot load LFM3 GGUFs yet; use `agent_harness.agent: syllabix` for LFM3 tools. Goose can search the web and fetch pages.

## TTS

| Yaml `pipeline.tts.model` | Notes | Source |
| --- | --- | --- |
| `pocket-tts` | Default — English ONNX + fixed Alba voice (~150 MB). No user voice data. | https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts |
| `kokoro` | Kokoro ONNX + default voice (~310 MB) | https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX |
| `qwen3-0.6` | Qwen3-TTS 0.6B + speech tokenizer | https://huggingface.co/mradermacher/Qwen3-TTS-12Hz-0.6B-Base-GGUF |
| `qwen3-1.7` | Qwen3-TTS 1.7B + speech tokenizer | https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF |
| `indic-mio` | Language coverage per the [Indic-Mio model card](https://huggingface.co/SPRINGLab/Indic-Mio) ([MioCodec](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz) MIT). Select with `pipeline.tts.model: indic-mio`; only `hi-female`, `hi-male`, and `ta-female` speaker embeddings ship in-tree. Weights download on first use. See [NOTICE](../crates/syllabix-tts/NOTICE). | https://huggingface.co/bytepass0/indic-mio-q8-gguf |

`pipeline.tts.provider: online` posts each finished sentence to OpenAI-compatible `POST /v1/audio/speech` (`stream_format=audio`). No TTS weights are fetched. Key: `SYLLABIX_TTS_API_KEY`. A self-hosted worker: [`syllabix serve tts`](cli.md#serve). Yaml: [local vs online](syllabix-yaml.md#local-vs-online).

Hosted endpoints that speak this Audio REST (not OpenAI Realtime): [OpenAI speech-to-text](https://platform.openai.com/docs/guides/speech-to-text) / [text-to-speech](https://platform.openai.com/docs/guides/text-to-speech), [Groq speech-to-text](https://console.groq.com/docs/speech-to-text) / [text-to-speech](https://console.groq.com/docs/text-to-speech), [whisper.cpp whisper-server](https://github.com/ggml-org/whisper.cpp/tree/master/examples/server), [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI). Default `run` stays local. Yaml: [local vs online](syllabix-yaml.md#local-vs-online).

## S2S (speech-to-speech)

Local speech-to-speech is not an STT, LLM, or TTS menu id. Select it with `pipeline.mode: s2s` and `pipeline.s2s.model` — [syllabix.yaml](syllabix-yaml.md#pipeline-mode--cascade-vs-s2s). Default `run` without that yaml stays cascade.

| Yaml `pipeline.s2s.model` | Notes | Source |
| --- | --- | --- |
| `lfm3-audio-2b` | English-only experimental voice chat (LFM 1.0). The turn uses the model’s default GGUF chat template in mixed text+audio mode ([Voice chat](https://huggingface.co/LiquidAI/LFM3-Audio-2B#voice-chat-speech-and-text-responses)). Weights fetch on first s2s run only. Indic / code-switched voice chat remains the cascade path (Hima, Indic-Mio, and related ids). | [model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B), [GGUF repo](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF), [NOTICE](../NOTICE) |

Silero VAD still commits each user turn. `run --barge-in` still interrupts playback on user speech; this path is turn-based, not full duplex. Assistant replies stay in the conversation until the context is full; a barged-in reply is closed in-cache so the next user turn can continue. Loop diagram: [architecture](architecture.md#speech-to-speech-pipelinemode-s2s).

Contributor pins, ggml patches, and llama-bench: [contributing.md](contributing.md#models).
