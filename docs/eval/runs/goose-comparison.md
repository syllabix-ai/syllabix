# Goose comparison: spoken-reply latency

Status: **pending capture** on the reference Linux machine. Every result below
is a placeholder for the operator to fill after the run.

## Setup

| Field | Value |
| --- | --- |
| Commit | `_pending_` |
| Host / OS | `_pending_` |
| CPU / RAM / GPU | `_pending_` |
| Goose | pinned `1.54.0` |
| Speech-to-text (STT) / text-to-speech (TTS) | `small` / `pocket-tts` (identical for every arm) |
| Local Liquid Foundation Model (LFM) | `LFM2.5-2.6B-QAD-Q4_0.gguf` |
| Local GPT-Generated Unified Format (GGUF) SHA-256 | `_pending_` |
| Diagnostics | `timestamps: true`, `audio: true` |

Use the same machine, the same STT and TTS settings, and the same LFM GGUF file
for the builtin and Goose local arms. Use one hosted model for the third arm.

Give each arm its own diagnostics folder so captures do not overwrite each other:

| Arm | `diagnostics.directory` |
| --- | --- |
| Builtin, in-process LFM | `target/turn-debug-builtin` |
| Goose, same LFM GGUF, local | `target/turn-debug-goose-local` |
| Goose, hosted model | `target/turn-debug-goose-hosted` |

## Method

The metric is the existing turn-timeline `audible_latency_ms`, defined as
`playback_first_ms - speech_end_ms` (silence-end to first audio out of the
speaker). The legacy time-to-first-byte (TTFB) clock is not used. No new harness
is added.

1. Capture the builtin in-process LFM arm with `developer_harness.agent: builtin`
   (default). Point `pipeline.llm` at the local model:

   ```yaml
   developer_harness:
     agent: builtin
   pipeline:
     llm:
       provider: local
       model: lfm2.5-2.6b
   diagnostics:
     timestamps: true
     audio: true
     directory: target/turn-debug-builtin
   ```

2. Capture Goose against the same LFM GGUF with `developer_harness.agent: goose`.
   Goose reads its own provider and model from
   `{SYLLABIX_CACHE_DIR}/goose/config/config.yaml` (Goose owns the LLM path;
   `pipeline.llm` is still required in yaml but is ignored):

   ```yaml
   developer_harness:
     agent: goose
   diagnostics:
     timestamps: true
     audio: true
     directory: target/turn-debug-goose-local
   ```

   ```yaml
   # {SYLLABIX_CACHE_DIR}/goose/config/config.yaml
   active_provider: local
   providers:
     local:
       enabled: true
       model: /absolute/path/to/LFM2.5-2.6B-QAD-Q4_0.gguf
   ```

3. Capture Goose with one hosted model, again with `developer_harness.agent: goose`.
   Use a separate Goose config: set `active_provider` to that hosted provider
   (not `local`) and its model id or environment variables — do not reuse the
   local GGUF block from step 2. Example shape (names depend on your provider):

   ```yaml
   developer_harness:
     agent: goose
   diagnostics:
     timestamps: true
     audio: true
     directory: target/turn-debug-goose-hosted
   ```

   ```yaml
   # {SYLLABIX_CACHE_DIR}/goose/config/config.yaml
   active_provider: openai
   providers:
     openai:
       enabled: true
       model: <hosted-model-id>
   ```

4. For each arm, hold at least 20 real conversational turns with diagnostics
   enabled, then summarize the sidecars:

   ```bash
   python3 scripts/summarize-timelines.py target/turn-debug-builtin
   python3 scripts/summarize-timelines.py target/turn-debug-goose-local
   python3 scripts/summarize-timelines.py target/turn-debug-goose-hosted
   ```

5. Record `audible_latency_ms` p50/p95, plus `llm_ttft_ms` and
   `llm_end_to_first_pcm_ms` for context. The summarize script prints how many
   sidecars it read and how many have `"outcome": "completed"`, but each segment
   row includes every sidecar that has both anchors (cancelled turns can still
   contribute if they reached playback). For the results table, either keep only
   completed turns (drop cancelled `turn-*` folders first, or filter
   `turn.json` to `"outcome": "completed"` before you copy numbers) or note in
   Notes when cancelled turns are included.

## Results

| Arm | n | audible p50 (ms) | audible p95 (ms) | llm_ttft p50/p95 (ms) | llm_end_to_first_pcm p50/p95 (ms) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Builtin, in-process LFM | `_pending_` | `_pending_` | `_pending_` | `_pending_` | `_pending_` | |
| Goose, same LFM GGUF, local | `_pending_` | `_pending_` | `_pending_` | `_pending_` | `_pending_` | |
| Goose, hosted model | `_pending_` | `_pending_` | `_pending_` | `_pending_` | `_pending_` | model: `_pending_` |

Pass rule: Goose on the same local GGUF must not be slower than the builtin
in-process path at median (`p50`) `audible_latency_ms`. Fill the verdict after
the capture:

```text
Latency verdict: PASS / FAIL / BLOCKED
Builtin p50: <ms>
Goose local p50: <ms>
Delta: <ms> (<percent>)
Notes:
```

## Capture checklist

- [ ] Same machine for all arms.
- [ ] Same LFM GGUF file and SHA-256 for both local arms.
- [ ] At least 20 completed turns per arm.
- [ ] Same STT and TTS ids/settings for all arms.
- [ ] `developer_harness.agent` set per arm (`builtin` vs `goose`).
- [ ] `diagnostics.directory` set per arm (table above).
- [ ] Hosted model id and provider recorded.
- [ ] Results table and latency verdict filled.
- [ ] Any failed or unusable turn described under Notes.
