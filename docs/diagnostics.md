# Diagnostics — turn timeline and audio capture

Per-turn instrumentation introduced by sequence row 34. Diagnostics are
yaml-only (the old `--turn-debug` flag is gone); default `run` records
nothing. Measurement protocols, segment definitions, and p50/p95 targets
live in [`reference-profiles.md`](reference-profiles.md) §10 — this file is
the field guide: how to enable, what each anchor means, and what a real
report looks like.

## Enable

```yaml
# syllabix.yaml
diagnostics:
  timestamps: true              # turn-*/turn.json sidecars under `directory`
  audio: true                   # additionally capture/clean/utterance/tts.wav; implies timestamps
  directory: target/turn-debug  # optional; omit when the default is fine
```

| Configuration | Sidecar | WAVs | PCM buffered |
|---|---|---|---|
| block absent / both false | no | no | none |
| `timestamps: true` | yes | no | none |
| `audio: true` | yes | yes | yes |

## Timeline anchors

Each `turn.json` carries `"timeline"` — monotonic milliseconds from the
turn's SpeechStart, rendered in pipeline order. Anchors the turn never
reached render as `null` (a barge-in-interrupted turn, for example, has no
`llm_last_token_ms`; a turn cut short by quitting may have no
`playback_done_ms` because the speaker never drained).

| Anchor | Captured at |
|---|---|
| `speech_start` | First Silero-positive frame opens the turn (epoch, renders as 0) |
| `speech_end` | VAD hangover satisfied; utterance closed |
| `stt_queued` | STT worker dequeued the utterance |
| `stt_done` | Transcript text and language ready |
| `llm_start` | LLM worker began generating |
| `llm_first_token` | First token streamed |
| `llm_last_token` | Last token of the generation |
| `tts_first_pcm` | First PCM the TTS worker synthesized (worker-side, includes row-33 incremental vocoder windows) |
| `tts_last_pcm` | Final PCM chunk |
| `playback_first` | Speaker callback first consumed samples of this turn |
| `playback_done` | Speaker drained after the final chunk (final audible sample) |

Named latency segments (`llm_ttft_ms`, `audible_latency_ms`,
`llm_end_to_first_pcm_ms`, …) are computed from these anchors by
`scripts/summarize-timelines.py`; the formulas are in
[`reference-profiles.md`](reference-profiles.md) §10.

## Capture and summarize

```bash
cargo run -p syllabix --release -- run          # talk ≥20 turns for a real sample
python3 scripts/summarize-timelines.py target/turn-debug   # contributor-only tool
```

Interrupted turns dump immediately as `"outcome": "cancelled"` with whatever
anchors exist, so barge-in captures work with `run --barge-in` too.

## Sample report

Output of `scripts/summarize-timelines.py` over a three-turn live capture:

```text
segment                      n       p50       p95     min     max
------------------------------------------------------------------
user_speech_ms               3      2237      2241    1279    2241
vad_queue_wait_ms            3         0         0       0       0
stt_decode_ms                3       389       411     384     413
stt_total_ms                 3       389       411     384     413
handoff_llm_ms               3         0         0       0       0
llm_ttft_ms                  3       200       255     149     261
llm_stream_ms                3       545       678     229     693
tts_lead_ms                  3      1019      1700     980    1776
llm_end_to_first_pcm_ms      3       790      1187     287    1231
tts_synthesis_ms             3      1339      1788     489    1838
playback_handoff_ms          3        10        11       2      11
audible_latency_ms           3      1595      2352    1563    2436
spoken_duration_ms           0         —         —       —       —
total_turn_ms                0         —         —       —       —

raw anchors (ms from speech_start):
anchor                       n       p50       p95
--------------------------------------------------
speech_start_ms              3         0         0
speech_end_ms                3      2237      2241
stt_queued_ms                3      2237      2241
stt_done_ms                  3      2630      2648
llm_start_ms                 3      2630      2648
llm_first_token_ms           3      2850      2887
llm_last_token_ms            3      3436      3532
tts_first_pcm_ms             3      3830      4583
tts_last_pcm_ms              3      5668      5972
playback_first_ms            3      3832      4592
```

Reading the sample:

- **`n=0` rows are expected when sessions end abruptly.** `spoken_duration_ms`
  and `total_turn_ms` need `playback_done_ms`, which only fires when the
  speaker callback drains after the last chunk — quitting right after a reply
  leaves it `null`. Let the agent finish, wait a beat, then quit if you want
  complete turns.
- **`llm_end_to_first_pcm_ms` is signed by design.** Here it is positive
  (~790 ms p50): first PCM landed *after* the last token, i.e. synthesis
  waited on the text rather than overlapping generation. Negative values mean
  the opposite (sentence streaming started while tokens were still arriving).
- **`vad_queue_wait_ms` / `handoff_llm_ms` ≈ 0** shows the bounded queues are
  not backing up on this machine; sustained non-zero values would point at
  queue sizing or a slow stage upstream.
- **`audible_latency_ms` (~1.6 s p50)** is the G4-style silence-end → first
  speaker-audio number; the split above says most of it sits in STT decode
  (~390 ms) plus LLM time-to-first-token (~200 ms) plus TTS lead/synthesis.
