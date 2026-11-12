# syllabix-riley-eval

Offline Riley / VAmoS-style card-ops eval, phase A1 of
[#329](https://github.com/syllabix-ai/syllabix/issues/329): the five card
tools, the Postgres seed (20 users, 35 cards), and the agent prompt. No audio.

Scores from this eval are our own approximation, for regression tracking and
demos only. They are not comparable with the
[VAmoS leaderboard](https://www.veris.ai/leaderboard/vamos).

This crate is not linked into `syllabix`, so the default `run` tool menu is
unchanged. `syllabix-riley-mcp` serves the tools over MCP stdio. It reads the
Postgres URL from `SYLLABIX_RILEY_DATABASE_URL` (optional schema:
`SYLLABIX_RILEY_SCHEMA`, default `riley`) and reloads the seed on start.
Postgres tests skip when the URL is unset; CI's coverage job sets it.

## Third-party files

[`vendor/riley-agent/`](vendor/riley-agent/) holds `agent_desc.txt` and
`init.sql`, unmodified from
[`veris-ai/riley-agent` `gradbot/`](https://github.com/veris-ai/riley-agent/tree/f54b220e21cd55d812e7a4386ea865d4791b6b98/gradbot)
at `f54b220e21cd55d812e7a4386ea865d4791b6b98`. The tool schemas in
`src/tools.rs` and the logic in `src/db.rs` are ported from `gradbot/app/`.
MIT License, Copyright (c) 2026 Veris Technologies Inc.; full text in
[`vendor/riley-agent/LICENSE`](vendor/riley-agent/LICENSE).
