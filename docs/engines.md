# Engines

Syllabix does not ship a model catalogue UI. Engines are **yaml ids** under `pipeline.stt`, `pipeline.llm`, and `pipeline.tts`. Unknown ids fail at load. First use of an id fetches pinned HTTPS assets and checks SHA-256. URLs: [contributing.md](contributing.md#models).

VAD is a single engine: Silero ONNX (`pipeline.vad.provider: silero`). Sample rate, frame size, and scoring are not yaml-tunable; threshold / min-speech / hangover / preroll are.

## Speech to text (`pipeline.stt`)

`provider` is `local` only.

| Id | Runtime | Languages | Best fit |
| --- | --- | --- | --- |
| `whisper-small` | whisper.cpp (Metal on Darwin) | Whisper codes or `auto` | Default conversation STT |
| `whisper-medium` | whisper.cpp | same | Higher quality, slower / larger |
| `whisper-large-v3-turbo` | whisper.cpp | same | Largest Whisper menu id |
| `whisper-medium-q5_0` | whisper.cpp quant | same | Smaller medium |
| `whisper-large-v3-turbo-q5_0` | whisper.cpp quant | same | Smaller turbo |
| `moonshine-streaming-small` | ONNX | `en` only | Live partials, lower power |
| `moonshine-streaming-medium` | ONNX INT8 (~590 MB) | `en` only | Larger streaming English |

Moonshine rejects `language: auto` and non-`en` codes. Partials appear while VAD owns the turn; `end_silence_ms` is still the turn-end control.

Measured STT rows: [workload benchmark — STT](workload_benchmark/workload_benchmark.md#stt).

## Language models (`pipeline.llm`)

`provider: local` (default) loads a GGUF in-process via llama.cpp. `provider: online` posts transcript text to an OpenAI-compatible `chat/completions` URL; see [api.md](api.md).

| Id | Notes |
| --- | --- |
| `lfm2.5-2.6b` | Default. First-run cache. Darwin Metal |
| `lfm2.5-350m` | Smaller LFM, fetched when selected |
| `lfm2.5-230m` | Smaller LFM, fetched when selected |
| `llama-3.2-1b` | Llama 3.2 1B Instruct Q4_K_M |
| `qwen3.5-0.8b` | Qwen 3.5 0.8B |
| `qwen3.5-2b` | Qwen 3.5 2B; only id that may set `thinking: true` |

Context length is the GGUF `n_ctx_train`. There is no yaml `n_predict` / `n_ctx`. Generate stops on EOS, cancel, or a full window.

Measured LLM rows: [workload benchmark — LLM](workload_benchmark/workload_benchmark.md#llm). Tok/s lab notes: [`vendor/llama-bench.md`](../vendor/llama-bench.md).

## Text to speech (`pipeline.tts`)

`provider` is `local` only.

| Id | Languages | Notes |
| --- | --- | --- |
| `pocket-tts` | English (fixed Alba) | Default ~150 MB ONNX. No user voice cloning |
| `kokoro` | Fixed ONNX voice (`language` ignored) | ~310 MB |
| `qwen3-0.6` | yaml `language` (`en`, `zh`, `de`, …) | CPU on every OS |
| `qwen3-1.7` | same | Larger Qwen TTS; CPU on every OS |

Pocket TTS accepts no user voice data. Qwen Base backbones use a pinned speaker embedding so the voice does not change every utterance.

Measured TTS rows: [workload benchmark — TTS](workload_benchmark/workload_benchmark.md#tts). Qwen compute placement: [reference-profiles.md](reference-profiles.md) §8.
