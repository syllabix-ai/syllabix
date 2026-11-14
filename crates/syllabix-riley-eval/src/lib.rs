//! Offline Riley / VAmoS-style card-ops eval for
//! <https://github.com/syllabix-ai/syllabix/issues/329>. Phase A1: five card
//! tools, the Postgres seed, and the agent prompt. Phase A2: voice sessions
//! through the production cascade with injected audio, a tool trace, and
//! deterministic DB checks. Phase A3: a scenario pack, an LLM caller voiced
//! by TTS, an LLM judge, and a runner that writes per-call JSONL plus a
//! pass-rate summary.
//!
//! Our own approximation of Veris's Riley agent. Scores are for regression
//! tracking and demos only and are not comparable with the VAmoS leaderboard.
//! This crate is not a dependency of the `syllabix` binary, so the default
//! `run` tool menu is unchanged.
//!
//! `vendor/riley-agent/` holds unmodified files from
//! <https://github.com/veris-ai/riley-agent> (MIT, Copyright (c) 2026 Veris
//! Technologies Inc.); see `vendor/riley-agent/LICENSE`.

pub mod caller;
pub mod chat;
pub mod checks;
pub mod db;
pub mod fixtures;
pub mod judge;
pub mod llm;
pub mod mcp;
pub mod runner;
pub mod scenario;
pub mod session;
pub mod tools;
pub mod voice;

/// Riley system prompt (`gradbot/agent_desc.txt`, unmodified).
pub const AGENT_PROMPT: &str = include_str!("../vendor/riley-agent/agent_desc.txt");

/// Schema plus seed data (`gradbot/db/init.sql`, unmodified): 20 users, 35 cards.
pub const INIT_SQL: &str = include_str!("../vendor/riley-agent/init.sql");

/// Postgres connection URL. Read from the environment only; never printed.
pub const DATABASE_URL_ENV: &str = "SYLLABIX_RILEY_DATABASE_URL";

/// Optional Postgres schema name (default [`DEFAULT_SCHEMA`]).
pub const SCHEMA_ENV: &str = "SYLLABIX_RILEY_SCHEMA";

/// Schema used when [`SCHEMA_ENV`] is unset.
pub const DEFAULT_SCHEMA: &str = "riley";

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256_hex(text: &str) -> String {
        Sha256::digest(text.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn vendored_prompt_and_seed_match_upstream_bytes() {
        assert_eq!(
            sha256_hex(AGENT_PROMPT),
            "f454e98bd43b27573ba05963652a1666dd4288051b0bc1b264f2361426be93d2"
        );
        assert_eq!(
            sha256_hex(INIT_SQL),
            "77dd2bb5738a5eb8086550238686fc4dcb966c36adb6df1868e7a26f03b1e509"
        );
        assert!(AGENT_PROMPT.starts_with("You are Riley"));
        assert_eq!(INIT_SQL.matches("\n('u_").count(), 20);
        assert_eq!(INIT_SQL.matches("\n('c_").count(), 35);
    }

    #[test]
    fn vendored_license_is_mit_from_veris() {
        let license = include_str!("../vendor/riley-agent/LICENSE");
        assert!(license.starts_with("MIT License"));
        assert!(license.contains("Copyright (c) 2026 Veris Technologies Inc."));
    }
}
