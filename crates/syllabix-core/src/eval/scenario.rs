//! Scenario corpus for the performance ledger (issue #45).
//!
//! One JSON object per line. `turns` names an embedded user-audio fixture and
//! supplies its STT fidelity reference, so every scenario is deterministic
//! to replay.
//!
//! The loader is strict: duplicate or empty ids, empty prompt lists (outside
//! `silence_only`), whitespace prompts, and thresholds outside `(0, 1]` fail
//! fast with [`Error::Config`] so a malformed corpus never produces silently
//! meaningless rows.

use std::sync::OnceLock;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Embedded launch corpus (`syllabix bench` ships it inside the binary so a
/// downloaded release can measure itself without a checkout).
pub const BUILTIN_SCENARIOS_JSONL: &str = include_str!("scenarios.jsonl");

/// Default in-order word-match floor for STT transcripts (issue #45 gates).
pub const DEFAULT_STT_MIN_MATCH: f64 = 0.8;

/// Bounds for `silence_seconds`.
const SILENCE_SECONDS_MAX: f64 = 10.0;

/// One scripted multi-turn conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct Scenario {
    /// Unique corpus id (`greeting_001`).
    pub id: String,
    /// Free-form bucket (`smalltalk`, `math`, …) copied into ledger rows.
    pub category: String,
    /// Spoken prompts, one per turn. Empty only for `silence_only`.
    pub turns: Vec<String>,
    /// Silence-only scenario: no speech is synthesized; zero turns must complete.
    pub silence_only: bool,
    /// Seconds of silence fed to VAD when `silence_only` (default 2.0).
    pub silence_seconds: f64,
    /// Per-turn and per-scenario gate configuration.
    pub expect: Expectation,
}

/// Gate configuration shared by every turn of one scenario.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Expectation {
    /// Each inner group is a set of synonyms; at least one member of **every**
    /// group must appear (case-insensitive) somewhere in the scenario replies.
    pub reply_contains_any: Vec<Vec<String>>,
    /// No member may appear (case-insensitive) in any reply.
    pub reply_not_contains: Vec<String>,
}

impl Scenario {
    /// Uniform STT word-match floor for every spoken-input scenario.
    pub fn stt_min_match(&self) -> f64 {
        DEFAULT_STT_MIN_MATCH
    }

    /// Number of turns the loop must complete (0 when silence-only).
    pub fn expected_turns(&self) -> usize {
        self.turns.len()
    }
}

impl<'de> Deserialize<'de> for Scenario {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            id: String,
            #[serde(default)]
            category: Option<String>,
            #[serde(default)]
            turns: Vec<String>,
            #[serde(default)]
            silence_only: bool,
            #[serde(default)]
            silence_seconds: Option<f64>,
            #[serde(default)]
            expect: RawExpectation,
        }
        #[derive(Deserialize, Default)]
        #[serde(deny_unknown_fields)]
        struct RawExpectation {
            #[serde(default)]
            reply_contains_any: Vec<Vec<String>>,
            #[serde(default)]
            reply_not_contains: Vec<String>,
        }

        let raw = Raw::deserialize(de)?;
        let expect = Expectation {
            reply_contains_any: raw.expect.reply_contains_any,
            reply_not_contains: raw.expect.reply_not_contains,
        };
        Self::build(
            &raw.id,
            raw.category.as_deref().unwrap_or("general"),
            raw.turns,
            raw.silence_only,
            raw.silence_seconds,
            expect,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl Scenario {
    /// Validate and build. Kept separate from [`Deserialize`] so error text is
    /// identical whether the line failed serde or our own rules.
    fn build(
        id: &str,
        category: &str,
        turns: Vec<String>,
        silence_only: bool,
        silence_seconds: Option<f64>,
        expect: Expectation,
    ) -> Result<Self> {
        let id = id.trim();
        if id.is_empty() {
            return Err(config_err("scenario id must not be empty"));
        }
        let category = category.trim();
        if category.is_empty() {
            return Err(config_err("scenario category must not be empty"));
        }
        if !silence_only && turns.is_empty() {
            return Err(config_err(format!(
                "scenario {id:?} needs at least one turn (or \"silence_only\": true)"
            )));
        }
        if silence_only && !turns.is_empty() {
            return Err(config_err(format!(
                "silence-only scenario {id:?} must have an empty \"turns\" list"
            )));
        }
        if let Some(text) = turns.iter().find(|t| t.trim().is_empty()) {
            return Err(config_err(format!(
                "scenario {id:?} has a whitespace-only turn ({text:?})"
            )));
        }
        if let Some(group) = expect
            .reply_contains_any
            .iter()
            .find(|g| g.iter().any(|k| k.trim().is_empty()))
        {
            return Err(config_err(format!(
                "scenario {id:?} has an empty keyword in reply_contains_any ({group:?})"
            )));
        }
        if let Some(keyword) = expect
            .reply_not_contains
            .iter()
            .find(|k| k.trim().is_empty())
        {
            return Err(config_err(format!(
                "scenario {id:?} has an empty keyword in reply_not_contains ({keyword:?})"
            )));
        }
        Ok(Self {
            id: id.to_string(),
            category: category.to_string(),
            turns,
            silence_only,
            silence_seconds: match silence_seconds {
                None => 2.0,
                Some(secs) if secs > 0.0 && secs <= SILENCE_SECONDS_MAX => secs,
                Some(secs) => {
                    return Err(config_err(format!(
                    "scenario {id:?} silence_seconds {secs} is outside (0, {SILENCE_SECONDS_MAX}]"
                )))
                }
            },
            expect,
        })
    }
}

fn config_err(message: impl Into<String>) -> Error {
    Error::Config {
        field: "eval.scenarios".into(),
        message: message.into(),
    }
}

/// Parse strict JSONL: one scenario per non-empty line, unique ids.
pub fn parse_scenarios(jsonl: &str) -> Result<Vec<Scenario>> {
    let mut out = Vec::new();
    for (index, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let scenario: Scenario = serde_json::from_str(line).map_err(|err| {
            config_err(format!(
                "line {}: {}",
                index + 1,
                sanitize_serde(&err.to_string())
            ))
        })?;
        if out
            .iter()
            .any(|existing: &Scenario| existing.id == scenario.id)
        {
            return Err(config_err(format!(
                "line {}: duplicate scenario id {:?}",
                index + 1,
                scenario.id
            )));
        }
        out.push(scenario);
    }
    if out.is_empty() {
        return Err(config_err("corpus is empty"));
    }
    Ok(out)
}

/// Serde errors embed our own messages after validation; strip the wrapping
/// noise but keep everything readable in the fail-fast output.
fn sanitize_serde(message: &str) -> String {
    message.to_string()
}

/// The embedded launch corpus, parsed once.
pub fn builtin_scenarios() -> &'static [Scenario] {
    static CELL: OnceLock<Vec<Scenario>> = OnceLock::new();
    CELL.get_or_init(|| {
        parse_scenarios(BUILTIN_SCENARIOS_JSONL).expect("embedded eval corpus is valid")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_message(jsonl: &str) -> String {
        parse_scenarios(jsonl).expect_err("must fail").to_string()
    }

    #[test]
    fn builtin_corpus_parses_and_covers_the_issue_workload() {
        let scenarios = builtin_scenarios();
        assert_eq!(scenarios.len(), 6);
        assert_eq!(scenarios.iter().filter(|s| s.silence_only).count(), 1);
        // ~6 scenarios x 2-3 turns (issue #45 workload), silence excluded.
        let turn_count: usize = scenarios.iter().map(|s| s.expected_turns()).sum();
        assert!((9..=14).contains(&turn_count), "turn count {turn_count}");
        for scenario in scenarios {
            assert!(!scenario.id.is_empty());
            assert!(!scenario.category.is_empty());
        }
    }

    #[test]
    fn parses_the_documented_shape() {
        let scenarios = parse_scenarios(
            r#"{"id": "greeting_001", "category": "smalltalk", "turns": ["Hello! What can you do?", "Tell me a short fun fact about space."], "expect": {"reply_contains_any": [["hello","hi"]], "reply_not_contains": ["<think>", "**"]}}"#,
        )
        .expect("documented shape");
        assert_eq!(scenarios.len(), 1);
        assert_eq!(scenarios[0].expected_turns(), 2);
        assert_eq!(scenarios[0].stt_min_match(), 0.8);
        assert_eq!(scenarios[0].expect.reply_contains_any.len(), 1);
    }

    #[test]
    fn defaults_apply() {
        let scenarios = parse_scenarios(r#"{"id": "a", "turns": ["hi there"]}"#).expect("parse");
        assert_eq!(scenarios[0].stt_min_match(), DEFAULT_STT_MIN_MATCH);
        assert_eq!(scenarios[0].category, "general");
        assert!(!scenarios[0].silence_only);
        assert_eq!(scenarios[0].silence_seconds, 2.0);
    }

    #[test]
    fn duplicate_id_fails() {
        let line = r#"{"id": "dup", "turns": ["one"]}"#;
        let message = err_message(&format!("{line}\n{line}\n"));
        assert!(message.contains("duplicate"), "{message}");
        assert!(message.contains("dup"), "{message}");
    }

    #[test]
    fn empty_turns_fail_unless_silence_only() {
        let message = err_message(r#"{"id": "empty", "turns": []}"#);
        assert!(message.contains("at least one turn"), "{message}");

        let scenarios =
            parse_scenarios(r#"{"id": "quiet", "silence_only": true, "turns": []}"#).expect("ok");
        assert!(scenarios[0].silence_only);
        assert_eq!(scenarios[0].expected_turns(), 0);
    }

    #[test]
    fn silence_only_rejects_prompts_and_bad_durations() {
        let message = err_message(r#"{"id": "s", "silence_only": true, "turns": ["hello there"]}"#);
        assert!(message.contains("empty"), "{message}");

        let message = err_message(r#"{"id": "s", "silence_only": true, "silence_seconds": 99}"#);
        assert!(message.contains("silence_seconds"), "{message}");
    }

    #[test]
    fn per_scenario_stt_gate_override_is_rejected() {
        let message =
            err_message(r#"{"id": "t", "turns": ["hi"], "expect": {"stt_min_match": 0.6}}"#);
        assert!(message.contains("unknown field"), "{message}");
    }

    #[test]
    fn whitespace_turn_fails() {
        let message = err_message(r#"{"id": "w", "turns": ["   "]}"#);
        assert!(message.contains("whitespace-only"), "{message}");
    }

    #[test]
    fn empty_keywords_fail() {
        let message = err_message(
            r#"{"id": "k", "turns": ["hi"], "expect": {"reply_contains_any": [[""]]}}"#,
        );
        assert!(message.contains("reply_contains_any"), "{message}");
        let message = err_message(
            r#"{"id": "k", "turns": ["hi"], "expect": {"reply_not_contains": ["  "]}}"#,
        );
        assert!(message.contains("reply_not_contains"), "{message}");
    }

    #[test]
    fn unknown_fields_fail() {
        let message = err_message(r#"{"id": "u", "turns": ["hi"], "nope": true}"#);
        assert!(message.contains("unknown field"), "{message}");
    }

    #[test]
    fn blank_lines_are_skipped_and_empty_corpus_fails() {
        assert!(parse_scenarios("\n\n").is_err());
        let scenarios = parse_scenarios("\n{\"id\": \"a\", \"turns\": [\"x\"]}\n\n").expect("ok");
        assert_eq!(scenarios.len(), 1);
    }
}
