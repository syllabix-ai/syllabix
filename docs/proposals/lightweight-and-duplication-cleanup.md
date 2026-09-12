# Proposal — Lightweight and duplication cleanup

**Status:** Plan only. No product behavior is proposed here. **Lifecycle:**
temporary; remove this file once the sequenced PRs ship or the plan is
rejected.

**Links:** grounded in the current tree at `main` (`f0d21de`, Pocket TTS
terminal-marker follow-up). Several items in the original audit are already
addressed; those are recorded so later PRs do not re-open finished work.

**Decision requested:** approve a sequence of small cleanup PRs whose target is
**one turn lifecycle and one implementation of each policy**, not zero
`match` statements. Startup provider selection, sample-format handling,
cancellation, and recoverable errors stay as explicit branches.

## 1. Outcome

The default spoken path, dialect adapters, and native evaluation callers stay
behavior-identical. Shared policies (tool capability, TTS turn/terminal
markers, generation cancel/callback mapping, YAML unknown keys, diagnostic
sidecar shape) live in one place each. Cheap copies and unused dependencies
go away where they still exist.

## 2. Non-goals

- Do not mechanically merge every `match`.
- Do not introduce a plugin/provider framework for tools.
- Do not change default `syllabix run` (still tool-free).
- Do not delete native/evaluation helpers that existing callers still use;
  isolate them instead.
- Do not change Pocket FlowLM / Mimi graph names, Qwen whole-reply
  prosody, or LFM vs Qwen tool dialects.
- Do not rewrite YAML or diagnostic JSON schemas; preserve current error
  messages, unknown-key rejection, and sidecar field names.

## 3. Current tree vs the audit

| Audit item | Current evidence | Disposition |
|---|---|---|
| Pocket `RawTensor::f32` dual `data` + cloned `f32` field | `OnnxTensor` / `OnnxData` in `crates/syllabix-core/src/onnx.rs` already stores one typed buffer. Remaining copies are `.to_vec()` when wrapping session outputs (embeddings, conditioning, latents) for the next graph. | Skip the dual-field removal. Optional later: take ownership of `OnnxTensor` outputs instead of copy-out. |
| Pocket `voice_state` rereads the voice file per sentence | `voice_state` in `pocket_tts.rs` is called from `synthesize_sentence_into` and `synthesize_fixture`. The file is parsed and tensors built every sentence. | **Do.** Load/validate once at engine construction; clone only mutable recurrent state per synthesis. Benchmark before further work. |
| Duplicate Pocket/Moonshine RawTensor/RawValue/RawRunner | Both engines already use `onnx.rs`: `OnnxSession`, `OrtSession` C-API `Run`, `RawValue` Drop, `check()`. Type-and-shape info is still released by hand; there is no `RawRunner` type. | **Small follow-up:** RAII for `OrtTensorTypeAndShapeInfo`. Do not invent a second session stack. |
| LLM generation glue | `LlamaLlm::generate`, `generate_tool_turn`, `generate_lfm_tool_turn`, and `generate_lfm_harness` repeat shutdown/stale checks, call logging, collect-then-`on_token`, parse, `normalize_local_tool_call`, and `note_tool_event`. Native `LlamaEngine::{generate, generate_with_tools, generate_with_lfm_tools}` triplicates abort/stall/callback mapping. Native FFI wrappers triplicates CString/ggml-lock/rc mapping. | **Do.** One generation driver + dialect adapters. Keep Qwen vs LFM prompt/parse/budget as separate branches. |
| Tool schemas ×3 | `local_tool_definitions_json()` in `llm.rs` and `tool_definitions()` in `openai.rs` independently list `web_fetch` / `web_search` / `shell`. `executor::validate_call` is a third name map. Descriptions already diverge (online `shell` examples vs local). | **Do.** One capability table generates both JSON dialects and the executor allow-list. |
| Advertised `cargo metadata` / `cargo tree` with `command_path("cargo")` always erroring | Phase 4 replaced the argv allowlist with generic `shell`. Those command names and `command_path` are gone. Online `shell` description still cites `` `cargo test` `` as an example. | **Do not restore a cargo provider.** Optionally drop cargo from the example string so the model is not steered at a binary the host does not guarantee. |
| Pocket TTS state vs Kokoro/Qwen | `tts.rs` already has `ChunkState` + `synthesize_chunk_shared` for Kokoro. Qwen uses `ChunkState` but buffers the whole reply. Pocket reimplements reset/think/`take_sentences`/terminal-silence in `PocketTts`. | **Do.** Share `ChunkState` / terminal-marker policy. Keep Pocket frame streaming and Qwen whole-reply vocoder as engine-specific. |
| Config YAML | `AgentConfig` is typed, but parse walks `serde_yaml::Value` by hand (`deny_unknown`, duplicated `optional_u32` / `optional_timeout_ms`) and `to_yaml` is a format string. | **Do.** Private serde DTO + explicit validation. Golden `to_yaml` tests and current error strings stay. |
| Diagnostics JSON | `turn_debug.rs` hand-renders `turn.json` then `replacen`s `tool_events` into the string. Timing collection is already independent. | **Do.** Sidecar struct + serde. Keep pretty-print shape the tests pin. |
| `ratatui` | Workspace pin plus `crates/syllabix/Cargo.toml`. Zero Rust uses. Live UI is `crossterm` in `crates/syllabix/src/tui.rs`. | **Do.** Remove the direct dependency and workspace pin. |
| `toggle_agent_muted` | No such API. Live control is mic mute (`RuntimeControls::toggle_mic_muted`, `m` in the TUI). | **Skip** unless product later wants a distinct agent-mute. Do not invent a dead control. |
| Evaluation-only surface | `LlamaLlm::generate_tool_turn` is `pub` and used from unit tests, not `Llm::generate`. `PocketTts::{text_fixture,c_api_fixture,synthesize_fixture}` are `pub` for `tests/native_inference/pocket_tts.rs`. `lib.rs` re-exports the whole `fake` suite for integration tests. | **Do where cheap** (`pub(crate)`, `#[cfg(test)]`, or a test-only module). Do not delete native callers. |

## 4. Design rule

Share **policy**, not **control flow**.

A valid shared component owns one of: terminal `is_last` rules, think-filter
plus sentence buffering, tool capability names/parameters, cancel→error
mapping, unknown YAML keys, sidecar field set.

A valid remaining `match` / separate function owns one of: provider
construction, PCM sample format, Qwen vs LFM dialect text, Pocket vs Qwen
streaming shape, recoverable skip vs cancel vs shutdown.

## 5. Sequenced PRs

Each PR should stay reviewable on its own, keep tests green, and avoid mixing
behavior changes with refactors. Suggested order is cheapest / lowest-risk
first so later refactors sit on a thinner tree.

### PR 1 — Unused `ratatui`

- Remove `ratatui` from `crates/syllabix/Cargo.toml` and the workspace
  `[workspace.dependencies]` pin/comment.
- Confirm `crossterm` remains the TUI implementation.
- Verify: `cargo clippy --workspace --all-targets -- -D warnings`.

### PR 2 — Pocket voice state load-once

- Parse the fixed alba SafeTensors file once in `PocketTts::from_parts` /
  `from_paths` (after `flow_state` specs exist). Store `Vec<OnnxTensor>` (or a
  compact cloneable buffer) on the engine.
- `synthesize_sentence_into` / `synthesize_fixture` clone that snapshot into
  the recurrent `flow_state` they mutate.
- Keep `voice_state` as the loader used by construction and by
  `pocket_tts_tests.rs` (unreadable-tensor cases).
- Benchmark before any further Pocket optimization (pooling allocators,
  skipping clones of unchanged tensors). Suggested check: existing native
  Pocket RTF print in `tests/native_inference/pocket_tts.rs` plus a unit test
  that the voice path is not opened per sentence (inject a missing file after
  load, or count reads).
- Optional in the same PR only if it stays mechanical: take `OnnxTensor` by
  value from `run()` instead of `f32_data(...).to_vec()` for embeddings /
  conditioning. Do not chase every copy.

### PR 3 — One tool capability table

- Add a private capability description (name, description, JSON Schema
  parameters, executor validator). Render:
  - local `<tools>` JSON (`local_tool_definitions_json`)
  - online chat-completions `tools` array (`tool_definitions`)
- `validate_call` iterates that table instead of a third string match.
- Preserve Phase-4 drain adapters (`argv`/`cwd` → `command`/`workdir`) in
  the dialect normalizers, not in the advertised schema.
- Do not add `cargo metadata` / `tree`. If the online description still
  names `cargo test`, replace it with a command the host actually runs in
  fixtures (`rg`, tests) or drop the example.
- Verify: `llm.rs` / `openai.rs` schema unit tests, `executor` validation
  tests, harness-quality only if `executor.rs` / `openai.rs` change
  (see `docs/contributing.md`).

### PR 4 — Shared TTS chunk / terminal policy

- Move `ChunkState` (think filter, buffer, turn/generation, `next_index`,
  `reset`, `emit`) to `speech_text.rs` or a small `tts_chunk.rs`.
- Pocket and Kokoro both: think → sentences → skip empty speak-text →
  synthesize → exactly one terminal `is_last` (silence fallback when the
  engine emits nothing).
- Qwen keeps whole-reply buffering and vocoder streaming; it still uses
  `ChunkState` for turn reset and the silence terminal.
- Pocket keeps `synthesize_sentence_into` (FlowLM + Mimi frame streaming).
- Do not force Pocket through `WaveformEngine::synthesize` returning one
  `Vec<i16>`.

### PR 5 — Shared local generation driver

Two layers, both in existing modules (no new crate):

1. **Native FFI / `LlamaEngine`:** one helper
   `run_native(cancel, on_piece, |abort, abort_user, sink| unsafe { ctx.… })`
   used by plain, Qwen-tools, and LFM-tools. The three shim entry points
   stay; only abort/stall/callback/error mapping is shared. Optionally
   collapse the near-identical CString packing in `syllabix-native`.
2. **`LlamaLlm` tool turns:** one helper that logs the call, checks
   shutdown/stale, runs an engine entry, emits one `TokenChunk`, then maps
   parse/normalize failures to `rejected` + empty calls (or harness spoken
   fallback). Adapters remain: Qwen parse vs LFM parse, LFM char budget,
   `generate_lfm_harness` loop / executor / continuation messages.

`Llm::generate` on the default path stays tool-free and byte-identical.
`generate_tool_turn` can become `pub(crate)` in a later eval-isolation PR.

### PR 6 — Config DTO

- Introduce a private serde struct matching `syllabix.yaml` keys.
- Deserialize with serde; run today’s `deny_unknown` / provider / range
  checks as a second pass so error `field` paths and messages stay stable.
- Render through serde_yaml **or** keep a tiny explicit renderer if serde
  cannot match the current `init` document (omitted default `diagnostics`,
  omitted `base_url` on local, double-quoted system prompt). Prefer matching
  `to_yaml` goldens over a prettier but different file.
- Collapse `optional_u32` / `optional_timeout_ms` into one integer helper
  parameterized by the existing `"must be a positive integer"` vs
  `"must be a non-negative integer"` messages.
- Verify: the large `config.rs` unit suite (unknown keys, thinking rules,
  permissions, diagnostics omit/include).

### PR 7 — Diagnostics sidecar via serde

- Define a `TurnSidecar` (plus nested timings/timeline/tool event structs)
  with `serde_json`.
- Collect timings as today; persist only at dump.
- Drop `json_string` / `replacen` insertion of `tool_events`.
- Pin formatting with `serde_json::to_string_pretty` **or** a snapshot test
  of a completed/cancelled/skipped dump so consumers of `turn.json` do not
  silently drift.
- Verify: `turn_debug.rs` tests (`dumps_four_wavs_and_sidecar` and timeline
  nulls).

### PR 8 — Evaluation-only surfaces (where practical)

- `PocketTts::{text_fixture,c_api_fixture,synthesize_fixture}`: `pub(crate)`
  or `#[cfg(feature = "native-inference")]` so the public crate surface is
  the `Tts` impl.
- `LlamaLlm::generate_tool_turn`: `pub(crate)` unless an external crate
  needs it (today: unit tests in `llm.rs` plus possible native tests).
- Fakes: keep re-exported; they are the in-tree test harness, not dead
  code. Optionally `#[doc(hidden)]` if the public docs are the concern.
- Do **not** `cfg(test)` native fixtures: `cargo test -p syllabix-core --test native_inference` is a separate harness.

### Optional later — ONNX TypeAndShape RAII

Only if PR 2 does not already touch `read_value`: wrap
`OrtTensorTypeAndShapeInfo` in a Drop guard so error paths cannot leak.
Leave model-specific I/O names in Pocket/Moonshine.

## 6. What not to merge

Leave these as separate branches even when they sit next to shared code:

- `LlmProvider` / `SttProvider` / `TtsProvider` selection at startup
  (`real.rs`, `config.rs`)
- PCM format conversion (`audio/convert.rs`)
- Cancel vs recoverable provider skip vs shutdown (`pipeline.rs`)
- Qwen `<tool_call>` vs LFM `<|tool_call_start|>` parse
- Pocket per-sentence streaming vs Qwen whole-reply streaming

## 7. Verification

Every PR:

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- targeted unit tests for the touched module
- `cargo test --workspace` before merge

Additionally:

- PR 2: Pocket native fixture RTF (or a recorded before/after note) if
  weights are available; otherwise unit-level “load once” proof
- PR 3: harness-quality comment if `executor.rs` / `openai.rs` /
  `types.rs` / `policy.rs` change
- PR 4: existing Pocket terminal-marker tests (`pocket_tts_tests.rs`) plus
  Kokoro/Qwen chunk tests in `tts.rs`
- PR 6: config goldens / unknown-key tests
- PR 7: sidecar dump tests

Coverage gate is unchanged (`cargo llvm-cov` floors in
`docs/contributing.md`). Native inference remains opt-in.

## 8. Risks

- **Serde YAML / JSON formatting** can fail goldens without a behavior
  change. Treat renderer output as part of the contract.
- **Pocket voice clone** still copies tensors per sentence; that is
  intended. The win is dropping filesystem parse. Further sharing of
  immutable tensors can wait for a benchmark.
- **Harness-quality** will fire if tool JSON text changes even when names
  and parameters do not. Keep descriptions stable unless PR 3 explicitly
  unifies them.
- **Public API churn** on `generate_tool_turn` / Pocket fixtures: grep
  in-tree callers first; this repo has no downstream crates in the
  workspace besides `syllabix` / `syllabix-native`.
