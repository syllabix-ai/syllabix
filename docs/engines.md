# Engines

The ids you can put in `syllabix.yaml`. Key syntax: [syllabix.yaml](syllabix-yaml.md). Measured timings: [workload benchmark](workload_benchmark/workload_benchmark.md). Licenses of fetched weights: [NOTICE](../NOTICE).

Weights download on first use of each id (HTTPS URL pinned in the binary, SHA-256 check, then reuse from cache). Default `run` fetches only the launch stack: Silero, Whisper `small`, LFM2.5-2.6B, Pocket TTS (~2.2 GB). `--help` and `init` download nothing.

Cache (`cache_root()`, then `models/v1`), in lookup order:

- `$SYLLABIX_CACHE_DIR/models/v1` if `SYLLABIX_CACHE_DIR` is set (that variable is the cache root, not `models/v1`)
- else `$XDG_CACHE_HOME/syllabix/models/v1`
- else, on Windows, `%LOCALAPPDATA%\syllabix\cache\models\v1`
- else `~/.cache/syllabix/models/v1`
- else `.syllabix-cache/models/v1` (cwd)

Share downloads across the CLI and embed hosts by leaving the variable unset or pointing every process at the same root. Details: [embed — Cache](embed.md#cache).

## Machine needs

| | What ships / what is known |
| --- | --- |
| **Release artifacts** | Linux x64, macOS Apple Silicon, macOS Intel, Windows x64 |
| **Default cache** | ~2.2 GB on first `run` |
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

## LLM

Local menu (unknown ids fail at load):

| Yaml `pipeline.llm.model` | Notes | Source |
| --- | --- | --- |
| `lfm2.5-2.6b` | Default; first-run cache | https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF |
| `lfm2.5-350m` | Fetched when selected | https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF |
| `lfm2.5-230m` | Fetched when selected | https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF |
| `llama-3.2-1b` | Fetched when selected | https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF |
| `qwen3.5-0.8b` | Fetched when selected | https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF |
| `qwen3.5-2b` | Fetched when selected; only id that may set `thinking: true` | https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF |

`pipeline.llm.provider: online` uses any id the OpenAI-compatible endpoint serves. Audio stays on the machine; only transcript text leaves. The API key is `SYLLABIX_LLM_API_KEY` (environment only, never yaml).

## TTS

| Yaml `pipeline.tts.model` | Notes | Source |
| --- | --- | --- |
| `pocket-tts` | Default — English ONNX + fixed Alba voice (~150 MB). No user voice data. | https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts |
| `kokoro` | Kokoro ONNX + default voice (~310 MB) | https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX |
| `qwen3-0.6` | Qwen3-TTS 0.6B + speech tokenizer | https://huggingface.co/mradermacher/Qwen3-TTS-12Hz-0.6B-Base-GGUF |
| `qwen3-1.7` | Qwen3-TTS 1.7B + speech tokenizer | https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF |

Contributor pins, ggml patches, and llama-bench: [contributing.md](contributing.md#models).
