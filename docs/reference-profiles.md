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
