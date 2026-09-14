# Architecture

Syllabix is one native executable. The desktop never talks to a localhost HTTP backend.

```text
mic
  → cpal capture
  → AEC3 (WebRTC, on by default)
  → Silero VAD (ONNX, 8 kHz pair-average of 16 kHz frames)
  → STT worker (whisper.cpp or Moonshine)
  → LLM worker (llama.cpp GGUF or outbound chat/completions)
  → TTS worker (Pocket TTS / Kokoro / Qwen)
  → cpal playback
       ↑
       barge-in (opt-in): SpeechStart cancels TTS + LLM and flushes the speaker
```

## Crates

| Layer | Path | Responsibility |
| --- | --- | --- |
| CLI / TUI | `crates/syllabix/` | `clap` commands, ratatui transcript, `run` / `init` |
| Runtime | `crates/syllabix-core/` | yaml, model cache, pipeline queues, VAD/STT/LLM/TTS adapters, AEC, diagnostics, optional developer harness and sandbox |
| Native FFI | `crates/syllabix-native/` | Shared ggml: whisper.cpp + llama.cpp (Darwin Metal + Accelerate; Linux/Windows portable CPU). KleidiAI off |

Weights are **not** linked into the binary. `syllabix-core` downloads manifest-pinned files into the cache on first use of each id.

## Process and network boundary

- Default `run`: no listening port, no API key, no yaml.
- Online LLM: the process is an HTTP **client**. Audio PCM does not go on the wire; transcript text does.
- Developer harness: the model may request a sandboxed shell (Seatbelt / Bubblewrap / Landlock). Session permissions are a yaml ceiling and cannot widen in-session.
- Diagnostics write local JSON and WAV files only when enabled in yaml.

## Delivery

| Piece | Role |
| --- | --- |
| `dist` Cargo profile | Thin-LTO Release artifacts |
| `.github/workflows/release.yml` | Attach four OS binaries + `SHA256SUMS` on `v*` tags |
| `.github/workflows/ci.yml` | Per-PR fmt, clippy, llvm-cov ≥85% (no native weights) |
| `scripts/package-release.sh` | Maintainer packaging if Actions cannot run |

Contributor build and CI detail: [contributing.md](contributing.md).
