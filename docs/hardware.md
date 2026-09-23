# Hardware

What a laptop needs to run `syllabix`, and how to shrink the local LLM if memory is tight. This is a recommendation from the measured default stack, not a CI gate.

The default `run` combination (Whisper `small` + LFM2.5-2.6B + Pocket TTS) was measured on an **Apple M4 / 16 GiB** class machine. Isolated component numbers live in the [workload benchmark](workload_benchmark/workload_benchmark.md). Full-loop resident memory (STT + LLM + TTS + AEC + audio I/O together) is **not** a published figure yet.

## Disk

First default `run` fetches about **2.2 GB** (Silero, Whisper `small`, LFM2.5-2.6B, Pocket TTS) into the model cache, then reuses it offline. Other yaml ids fetch on first use of that id and add to disk. Cache path: [install](install.md).

## Compute

| Target | What the Release binary uses |
| --- | --- |
| macOS Apple Silicon / Intel | Metal + Accelerate for STT and the local LLM |
| Linux x64 / Windows x64 | Portable CPU (`n_gpu_layers=0`) |
| Qwen TTS (`qwen3-0.6` / `qwen3-1.7`) | CPU on every OS |

CUDA, Vulkan, and ROCm are off in the shipped binary. Linux with an NVIDIA GPU does not get a separate artifact today.

## Memory

Isolated **LLM-only** after-load RSS on the published benches (not the full voice loop):

| Yaml `pipeline.llm.model` | ~after-load RSS (M4 / 16 GiB) | ~after-load RSS (Linux CPU / ~16 GiB) |
| --- | ---: | ---: |
| `lfm2.5-2.6b` (default) | ~3.6 GiB | ~4.8 GiB |
| `lfm2.5-350m` | ~1.8 GiB | ~1.9 GiB |
| `lfm2.5-230m` | ~1.7 GiB | ~1.8 GiB |

If memory is tight, `syllabix init` and try a smaller local id, or set `pipeline.llm.provider: online` with `base_url` and `SYLLABIX_LLM_API_KEY` so the LLM runs in the cloud (transcript text only; mic audio stays local). FAQ: [memory](faq.md#memory-is-tight-can-i-use-a-smaller-model). Keys: [syllabix.yaml](syllabix-yaml.md).

## OS and devices

Release artifacts: Linux x64, macOS Apple Silicon, macOS Intel, Windows x64 — names in [install](install.md).

`run` needs a microphone **and** speakers at start. Headless boxes fail fast. AEC calibrates ~10 s; headphones are the fallback if laptop speakers still look like the user.

Windows x64 and macOS Intel compile in CI. Open-speaker echo and barge-in quality on those machines is not a published profile yet.

## What this page does not claim

- A pass/fail RAM number for the full live loop
- Cold-start wall time (binary on disk → first spoken reply)
- Silence-end → first-audio p50/p95

Those stay in [reference profiles](reference-profiles.md) until someone fills the tables.
