# Reference test profiles and measurement protocols

How Syllabix measures conversation quality. This is the measurement lab,
not the user README. Every number claimed in a product PR must come from a
command in this file, on one of the profiles below. Launch gates that
still need human machines (signed macOS artifacts, Windows first-open,
measured cold start, end-to-end latency p50/p95) stay empty until someone
runs the protocol — do not invent them in user-facing docs.

## 1. Reference profiles

| Profile | Hardware | Role |
|---|---|---|
| **A — MacBook Air** | Apple Silicon laptop, built-in microphone and speakers at 48 kHz, no headphones, quiet room | The echo/barge-in/3-minute gate. Full-duplex AEC3 is only honest on open speakers; headphones hide self-echo. |
| **B — Linux x64** | GitHub runner or contributor box, ALSA null / no devices, `libasound2-dev` installed | The merge bar: fmt/clippy/release link/dylibs/one-ggml, llvm-cov ≥85% without native weights, `cargo test --workspace` with native inference once. |
| **C — Clean machine** | Any OS image with **no rustc/cargo/python**, network available for the first download only | The scripted README smoke (`scripts/smoke-setup.sh`) and the human 3-minute spoken-reply gate after a Release exists. |

Windows x64 and macOS Intel have no profile owner yet: CI compiles them
(`ci.yml` matrix), but nobody has measured audio quality on them. Say so in
PRs rather than assuming Linux numbers transfer.

## 2. Echo — AEC zero false turns (profile A)

The launch gate. The mic stays open; nothing is muted.

```bash
cargo test -p syllabix-core --test audio_io \
  hardware_aec_1_minute_playback_has_zero_false_turns \
  -- --ignored --exact --nocapture
```

Protocol: 10 s automatic calibration, then 1 minute of looping
`tests/fixtures/vad/speech.wav` on built-in speakers while nobody talks.
**Pass = 0 Silero `SpeechStart` events.** Debug dumps land in
`$SYLLABIX_AEC_DEBUG_DIR` or `target/aec-debug`
(`render.wav`, `capture.wav`, `clean.wav`, sidecar).

Do not pass this by muting capture during playback, adding DeepFilterNet,
or raising Silero threshold/min-speech.

## 3. Barge-in p95 interrupt latency (profile A)

Enable diagnostics in the project's `syllabix.yaml` (field guide:
[`diagnostics.md`](diagnostics.md)), then run with the flag:

```yaml
diagnostics:
  timestamps: true
  audio: true
  directory: target/turn-debug
```

```bash
cargo run -p syllabix --release -- run --barge-in
```

While the agent speaks, talk over it repeatedly. With diagnostics enabled each
interrupted turn dumps the interrupted TTS plus your new utterance.

Measurement: timestamped speech-onset event → playback silence. Launch gate
is **<200 ms at p95 across 100 scripted interruptions per reference OS
profile**. Scripted cancel/flush/preroll behavior is covered by unit tests
(`barge_in_*`, `cancel_*` in syllabix-core/syllabix); the live p95 on open
speakers needs profile A and stays recorded in the launch evidence until a
human runs it.

Without `--barge-in`, VAD SpeechStart during TTS must **not** cancel
playback — that difference is itself a test
(`parses_run_barge_in*`, flag-off cancellation tests).

## 4. Spoken-text quality

- **Intelligibility:** text → Kokoro → Whisper `small` → text must reach
  ≥80 % in-order word match (`TTS_ASR_MIN_WORD_MATCH`). Non-silent PCM is
  not acceptance.
- **Markdown:** headings, list markers, emphasis, links are stripped before
  TTS (`markdown_fixtures_remove_headings_lists_and_emphasis`,
  `markdown_is_stripped_before_synthesis`).
- **Chain of thought:** `<think>…</think>` is stripped before TTS and hidden
  in the TUI even when yaml enables `thinking: true`.
- **Empty turns:** empty or whitespace STT never starts LLM/TTS
  (`empty_utterance_is_rejected_before_decode`).

## 5. Latency lines

The TUI reports STT, TTFT, TTFB, and total per turn (legacy enqueue-based
clocks, unchanged since row 25). The finer, device-true numbers come from the
row-34 timeline sidecars — see §10. Record them from profile A runs when
claiming speedups; do not quote wall-clock numbers from CI. LLM tok/s
comparisons belong in `vendor/llama-bench.md`.

## 6. Coverage and native inference (profile B)

```bash
cargo llvm-cov --workspace --fail-under-lines 85 --cobertura --output-path coverage.xml
cargo test --workspace   # last; launch-stack native inference runs here exactly once
```

llvm-cov sets `--cfg coverage` and must not load whisper.cpp / llama.cpp /
Kokoro weights or open devices. Floor is 85 % lines.

## 7. 3-minute gate (profile C)

After a published GitHub Release:

```bash
SMOKE_RELEASE_URL=https://github.com/syllabix-ai/syllabix/releases/download/<tag> \
  scripts/smoke-setup.sh syllabix-Linux-x86_64    # per OS you can touch
```

covers checksum verify, clean-toolchain `--help`, `init`, and the offline
second run. The spoken part stays human: download on the clean machine,
`./syllabix run`, talk, then block the network and confirm a second `run`
still works. Cold-start wall time (binary on disk → first spoken reply) is
recorded here when measured; the user README must not claim three minutes
until that row is filled.

## 8. TTS provider compute placement (row 31, profile A)

`provider: qwen` runs its backbone plus mtmd audio graphs on **CPU on every
OS**, while STT/LLM keep their shipped placement (Metal on Darwin). This is
not a fallback: the audio gen_code graph asks for a ~870 MiB Metal compute
buffer that `ggml_backend_sched` fails to place while the STT and LLM
contexts are resident (the normal Syllabix configuration), and llama-bench
shows the backbone does not want the GPU anyway.

Protocol (llama-bench at the vendored llama.cpp commit `ad1de39`, 4 threads,
pp512 / tg128, 2 repetitions, MacBook Air Apple Silicon):

```bash
llama-bench -m Llama-3.2-1B-Instruct-Q4_K_M.gguf   -p 512 -n 128 -t 4 -r 2 -ngl 99,0
llama-bench -m Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf -p 512 -n 128 -t 4 -r 2 -ngl 0,99
```

| Model | backend | pp512 t/s | tg128 t/s |
|---|---|---:|---:|
| Llama 3.2 1B Q4_K_M (LLM slot) | Metal, BLAS | 1137.9 ± 160.6 | 106.3 ± 2.0 |
| Llama 3.2 1B Q4_K_M | CPU | 299.3 ± 8.0 | 81.2 ± 1.7 |
| Qwen3-TTS 1.7B Q4_K_M (TTS slot) | CPU | 832.0 ± 28.5 | **91.5 ± 0.7** |
| Qwen3-TTS 1.7B Q4_K_M | Metal (solo reference) | 220.2 ± 4.0 | 74.7 ± 0.3 |

Read: the LLM slot is ~31% faster at generation on Metal and stays there;
the TTS backbone is ~22% *faster* on CPU than on Metal, so the CPU pin
costs nothing and buys coexistence. Revisit Metal for the audio graph only
after upstream splits it smaller.

Sentence-level TTFB/RTF for the qwen provider land via the opt-in capture:

```bash
SYLLABIX_CACHE_DIR=<cache> SYLLABIX_NATIVE_LATENCY=1 \
  SYLLABIX_NATIVE_MODELS=qwen3-0.6,qwen3-1.7 \
  cargo test --release -p syllabix-core --test native_inference \
  qwen_latency_capture -- --nocapture
```

Run it `--release`: a debug-profile build compiles ggml unoptimized and
produces meaningless RTF (measured ~50x audio time in debug).

## 9. Qwen3-TTS backbone menu and voice pinning (row 32, profile A)

Row 32 adds `qwen3-0.6` alongside the row-31 `qwen3-1.7` under
`pipeline.tts.model`. Each backbone pairs with its own mmproj: the upstream
speech-tokenizer *encoder* weights are byte-identical across sizes, but the
mmproj also carries the projector into the LM embedding space (2048-d for
1.7B, 1024-d for 0.6B) — pairing across sizes fails at load with an
n_embd mismatch. Both 0.6B files (backbone Q4_K_M + mmproj Q8_0) ship from
the community conversion at `mradermacher/Qwen3-TTS-12Hz-0.6B-Base-GGUF`,
sha256-pinned in the manifest; they must pass the same native gates as the
1.7B weight — round-trip ≥80%, numbers gate, cancel <5 s, first sentence
before completion.

**Voice pinning:** the Base backbones are speaker-unconditioned, so every
cold-start generation used to sample a new speaker — a different voice per
utterance. At load the engine now synthesizes a short anchor clip on a
chain pinned to a fixed seed, encodes it through the mmproj speaker
encoder, and conditions every sentence on that x-vector (`speaker_ref`).
Fixed anchor seed ⇒ the same Syllabix voice across sentences, runs, and
backbone sizes; the runtime sampler seed still varies prosody. The machine
proof is `qwen_voice_is_deterministic_under_a_pinned_seed` (two independent
engines, same seed ⇒ identical PCM); speaker consistency itself stays a
human listening-gate item on profile A.

Native synthesis evidence lands through the opt-in capture:

```bash
SYLLABIX_CACHE_DIR=<cache> SYLLABIX_NATIVE_LATENCY=1 \
  SYLLABIX_NATIVE_MODELS=qwen3-0.6,qwen3-1.7 \
  cargo test --release -p syllabix-core --test native_inference \
  qwen_latency_capture -- --nocapture
```

| Model | RTF p50 | RTF p95 |
|---|---:|---:|
| `qwen3-1.7` | 3.87 | 4.85 |
| `qwen3-0.6` | 2.62 | 3.65 |
| `pocket-tts` | 0.25 | 0.27 |

Each row is a release-build run over 20 fixed English sentences on the
reference MacBook Air (Apple M4, 16 GB).

## 10. Turn timeline instrumentation (profile A)

Diagnostics are yaml-only (enable + field guide:
[`diagnostics.md`](diagnostics.md)):

```yaml
# syllabix.yaml
diagnostics:
  timestamps: true   # turn-*/turn.json sidecars under `directory`
  audio: true        # additionally capture/clean/utterance/tts.wav (implies timestamps)
  directory: target/turn-debug
```

Each sidecar carries `"timeline"` — monotonic milliseconds from the turn's
SpeechStart, captured at eleven stage boundaries: `speech_start`,
`speech_end`, `stt_queued`, `stt_done`, `llm_start`, `llm_first_token`,
`llm_last_token`, `tts_first_pcm`, `tts_last_pcm`, `playback_first` (first
speaker-callback samples), and `playback_done` (final audible drain).
Anchors a cancelled turn never reached render as `null`.

### Segments

| Segment | Formula | Reads as |
|---|---|---|
| `user_speech_ms` | speech_end − speech_start | How long the user spoke |
| `vad_queue_wait_ms` | stt_queued − speech_end | Utterance sat in queue |
| `stt_decode_ms` | stt_done − stt_queued | Whisper decode alone |
| `stt_total_ms` | stt_done − speech_end | Silence-end → transcript |
| `handoff_llm_ms` | llm_start − stt_done | Transcript → generation start |
| `llm_ttft_ms` | llm_first_token − llm_start | Time-to-first-token |
| `llm_stream_ms` | llm_last_token − llm_first_token | Generation duration |
| `tts_lead_ms` | tts_first_pcm − llm_first_token | First token → first PCM (sentence buffering visible) |
| `llm_end_to_first_pcm_ms` | tts_first_pcm − llm_last_token | Signed: negative ⇒ synthesis overlapped generation (Kokoro sentence streaming); positive ⇒ TTS waited for the whole cleaned reply (Qwen whole-utterance buffering) |
| `tts_synthesis_ms` | tts_last_pcm − tts_first_pcm | Synthesis throughput |
| `playback_handoff_ms` | playback_first − tts_first_pcm | PCM ready → device audible |
| **`audible_latency_ms`** | **playback_first − speech_end** | **Silence-end → first audio out of the speaker — the G4 budget metric** |
| `spoken_duration_ms` | playback_done − playback_first | Reply playback duration |
| `total_turn_ms` | playback_done − speech_end | Full round-trip |

The legacy TUI clocks are unchanged; `audible_latency_ms` will read higher
than the old TTFB because it now includes device handoff — that delta is the
instrumentation gap this row exists to expose.

### Capture protocol

Per configuration to measure (default stack first: whisper `small` +
`llama-3.2-1b` + Kokoro; then `qwen3-0.6`, `qwen3-1.7`, `qwen3.5-0.8b`,
`qwen3.5-2b`, and the `online` LLM if relevant):

1. Write a scratch `syllabix.yaml` with diagnostics on.
2. Hold ≥20 real conversations turns on profile A (built-in speakers, quiet room).
3. Summarize:

```bash
python3 scripts/summarize-timelines.py target/turn-debug   # contributor-only tool
```

4. Record per-segment p50/p95 below.

### Results

*(to be filled by the reference-Air capture; the split decides the next
latency optimization row)*

| Configuration | n | audible_latency p50/p95 | llm_ttft p50/p95 | stt_total p50/p95 | llm_end_to_first_pcm p50/p95 |
|---|---:|---:|---:|---:|---:|
| default (kokoro + llama-3.2-1b) | | | | | |
