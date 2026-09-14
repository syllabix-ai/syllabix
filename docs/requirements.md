# Requirements

Requirements below are for the **default** local stack (Silero, Whisper `small`, LFM2.5-2.6B, Pocket TTS). Larger yaml ids need more disk and RAM. Component timings: [workload_benchmark/workload_benchmark.md](workload_benchmark/workload_benchmark.md). Conversation-quality protocols: [reference-profiles.md](reference-profiles.md).

## Machine

| | **Minimum** | **Recommended** |
| --- | --- | --- |
| **OS** | 64-bit Linux, macOS, or Windows matching a [Release artifact](install.md) | Current desktop OS |
| **RAM** | 8 GB | 16 GB+ (reference profile is Apple M4 / 16 GB) |
| **Disk** | ~3 GB free (binary + ~2.2 GB default cache) | SSD; budget extra GBs per optional GGUF / Whisper size / Qwen TTS |
| **Audio** | One capture device and one playback device at launch | Built-in laptop mic + speakers for echo testing; headphones if AEC fails |
| **GPU** | None | Apple Silicon (Metal + Accelerate for Whisper and the local LLM) |
| **Network** | HTTPS on first `run` of each model id | None after that id is cached |

Linux and Windows Release binaries use portable CPU for llama.cpp / whisper.cpp (`n_gpu_layers=0`). Darwin builds use Metal (`n_gpu_layers=-1`, Whisper `use_gpu`). Qwen TTS graphs stay on CPU on every OS.

## Cache locations

- `$SYLLABIX_CACHE_DIR/models/v1` if set
- otherwise `~/.cache/syllabix/models/v1`
- Windows: `%LOCALAPPDATA%\syllabix\cache\models\v1`

`--help` and `init` do not download weights.

## From source

Rust 1.91+, CMake, a C++ compiler. Linux also needs ALSA headers (`libasound2-dev`). See [contributing.md](contributing.md).

## Recommended ids by hardware

| Hardware | STT | LLM | TTS |
| --- | --- | --- | --- |
| Apple Silicon 16 GB | `whisper-small` | `lfm2.5-2.6b` | `pocket-tts` |
| CPU-only x64, 16 GB | `whisper-small` or Moonshine (English) | `lfm2.5-2.6b` or smaller LFM / Llama 3.2 1B | `pocket-tts` |
| Need live English partials | `moonshine-streaming-small` | default | default |
| Extra TTS quality (RAM) | default | default | `kokoro` or `qwen3-0.6` |

Do not select `qwen3-1.7` on an 8 GB machine. Workload rows show multi-gigabyte RSS for Qwen TTS backbones.

Windows x64 and macOS Intel compile in CI; audio quality on those profiles has no dedicated owner yet. Do not assume M4 numbers transfer.
