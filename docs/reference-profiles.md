# Reference test profiles and measurement protocols

How Syllabix measures conversation quality. These are the protocols behind
the launch definition of done in `V0_LAUNCH.md`. Every number claimed in a
product PR must come from a command in this file, on one of the profiles
below.

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

```bash
cargo run -p syllabix --release -- run --barge-in --turn-debug target/turn-debug
```

While the agent speaks, talk over it repeatedly. With `--turn-debug` each
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

The TUI reports STT, TTFT, TTFB, and total per turn. Record them from
profile A runs when claiming speedups; do not quote wall-clock numbers from
CI. LLM tok/s comparisons belong in `vendor/llama-bench.md`.

## 6. Coverage and native inference (profile B)

```bash
cargo llvm-cov --workspace --fail-under-lines 85 --cobertura --output-path coverage.xml
cargo test --workspace   # last; native inference runs here exactly once
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
`./syllabix run`, start talking within three minutes of putting the binary
on disk, then block the network and confirm a second `run` still works.

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
SYLLABIX_CACHE_DIR=<cache> SYLLABIX_QWEN_LATENCY=1 \
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

Latency evidence lands via the same opt-in capture, which prints one line
per backbone:

```bash
SYLLABIX_CACHE_DIR=<cache> SYLLABIX_QWEN_LATENCY=1 \
  cargo test --release -p syllabix-core --test native_inference \
  qwen_latency_capture -- --nocapture
```

| Backbone | TTFB p50 | TTFB p95 | RTF p50 | RTF p95 |
|---|---:|---:|---:|---:|
| `qwen3-1.7` (row 31 baseline) | 8625 ms | 8627 ms | 3.87 | 4.85 |
| `qwen3-0.6` (row 32) | 7285 ms | 6256 ms | 2.62 | 3.65 |

Budget: ≤4.5 s TTFB p50 on portable CPU (G4). Row 31 measured ≈8.6 s p50 /
RTF ≈3.6 for the 1.7B slot; the row-32 capture reproduces that baseline
(8625 ms / 3.87) on the same machine and brings the qwen provider to
7285 ms / 2.62 with the 0.6B backbone — a real cut, still above budget.
The capture times full-sentence synthesis (one chunk = one complete
generate + vocoder flush); playback cannot start before decode finishes.
That structural remainder is row 33's incremental vocoder streaming, not a
backbone-size problem. The delta is recorded here for founder acceptance
per the row-32 merge gate ("inside budget or founder-accepted delta").
Metric note: with n=20 the p95 slot is the second-largest sample, which is
why the 0.6B p95 sits below its p50.
