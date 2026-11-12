# syllabix-riley-eval

Offline Riley / VAmoS-style card-ops eval for
[#329](https://github.com/syllabix-ai/syllabix/issues/329). Phase A1: the five
card tools, the Postgres seed (20 users, 35 cards), and the agent prompt.
Phase A2: voice sessions through the production cascade. Phase A3: scored
runs over a scenario pack.

Scores from this eval are our own approximation, for regression tracking and
demos only. They are not comparable with the
[VAmoS leaderboard](https://www.veris.ai/leaderboard/vamos).

This crate is not linked into `syllabix`, so the default `run` tool menu is
unchanged. `syllabix-riley-mcp` serves the tools over MCP stdio. It reads the
Postgres URL from `SYLLABIX_RILEY_DATABASE_URL` (optional schema:
`SYLLABIX_RILEY_SCHEMA`, default `riley`) and reloads the seed on start.
Postgres tests skip when the URL is unset; CI's coverage job sets it.

## Voice sessions (A2)

`session::run_session` drives
[`run_loop_captured`](../syllabix-core/src/pipeline.rs) with injected caller
audio (`voice::InjectedCapture`) and a recording speaker
(`voice::RecordingSink`); barge-in is optional. The agent stage is
`llm::RileyLlm`: an OpenAI-compatible chat-completions client with the Riley
prompt and tools, so the default harness tool menu stays unchanged. Keys come
only from `SYLLABIX_STT_API_KEY`, `SYLLABIX_LLM_API_KEY`, and
`SYLLABIX_TTS_API_KEY`. Each call returns the transcript, the tool trace, and
the database before and after; `checks::evaluate` fails a call on any error.
Tests use a local mock of the three endpoints, so CI makes no paid calls.

## Scored runs (A3)

`syllabix-riley-eval` (see `--help`) runs the [scenario pack](scenarios/):
at least one simple, one complex, and one adversarial call, each with a
database overlay, a caller goal and traits, and yes/no checks. An LLM plays
the caller (`caller.rs`) and its lines are spoken by TTS; an LLM judge
(`judge.rs`) decides the assertions. A call passes only when every
deterministic check and every assertion passes. Results are one JSON line
per call plus a summary with the pass rate, the hosted stages used (host and
model only), and the label above; the agent stage is `riley-openai`.
`--mode text` skips TTS and STT and is reported as a smoke check, never as a
voice score. Every run calls hosted endpoints, so it needs `--allow-hosted`;
a key in the environment is not consent. CI uses only the local mock.

## Third-party files

[`vendor/riley-agent/`](vendor/riley-agent/) holds `agent_desc.txt` and
`init.sql`, unmodified from
[`veris-ai/riley-agent` `gradbot/`](https://github.com/veris-ai/riley-agent/tree/f54b220e21cd55d812e7a4386ea865d4791b6b98/gradbot)
at `f54b220e21cd55d812e7a4386ea865d4791b6b98`. The tool schemas in
`src/tools.rs` and the logic in `src/db.rs` are ported from `gradbot/app/`.
MIT License, Copyright (c) 2026 Veris Technologies Inc.; full text in
[`vendor/riley-agent/LICENSE`](vendor/riley-agent/LICENSE).
