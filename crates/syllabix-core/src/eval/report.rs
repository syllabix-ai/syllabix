//! Ledger row shape and JSONL output (issue #45).
//!
//! One JSON object per scenario, one line each — the format CI concatenates
//! into `docs/eval/results.csv`. Rows are named profiles, never averaged:
//! `profile` names the axis that moved (`default`, `stt-medium`,
//! `llm-qwen3.5-0.8b`), and `config` carries the exact ids.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::eval::fingerprint::Fingerprint;
use crate::eval::gates::{ScenarioGateVerdicts, TurnGateVerdicts};
use crate::eval::scenario::Scenario;
use crate::eval::BenchOptions;

/// Bump when the row shape changes; the CSV compiler rejects other versions.
pub const SCHEMA_VERSION: u32 = 2;

/// Harness kind for a row. `fixture` rows are compute-bound (no device
/// pacing); `live` is reserved for the separate Mac G4 capture using the
/// same schema.
pub const MODE_FIXTURE: &str = "fixture";
pub const MODE_LIVE: &str = "live";

/// Whether the scenario produced a measurement or was unavailable before it
/// could start (for example, a model backend could not initialize).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchStatus {
    Measured,
    Unavailable,
}

/// The three model ids a row measured.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConfigIds {
    /// STT yaml id (`small`, …).
    pub stt: String,
    /// LLM yaml id (`llama-3.2-1b`, …).
    pub llm: String,
    /// TTS yaml id (`kokoro`, …).
    pub tts: String,
}

impl ConfigIds {
    /// The launch stack ids.
    pub fn v0() -> Self {
        let defaults = BuiltinDefaults::v0();
        Self {
            stt: defaults.stt_model.as_str().to_string(),
            llm: defaults.llm_model.to_string(),
            tts: defaults.tts_model.as_str().to_string(),
        }
    }

    /// Named-profile id: `default`, or the axes that moved from launch ids.
    pub fn profile_name(&self) -> String {
        let defaults = Self::v0();
        let mut parts: Vec<String> = Vec::new();
        if self.stt != defaults.stt {
            parts.push(format!("stt-{}", self.stt));
        }
        if self.llm != defaults.llm {
            parts.push(format!("llm-{}", self.llm));
        }
        if self.tts != defaults.tts {
            parts.push(format!("tts-{}", self.tts));
        }
        if parts.is_empty() {
            "default".to_string()
        } else {
            parts.join("+")
        }
    }
}

/// One completed turn inside a ledger row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TurnRecord {
    /// Zero-based turn index within the scenario.
    pub index: usize,
    /// Spoken prompt matching the frozen microphone fixture.
    pub prompt: String,
    /// Whisper transcript of the synthesized user audio.
    pub stt_text: String,
    /// Raw LLM reply (think tokens included in the text if any leaked).
    pub reply: String,
    /// Post-strip text TTS actually spoke.
    pub speak_text: String,
    /// Utterance ready → transcript.
    pub stt_ms: u64,
    /// LLM start → first token.
    pub ttft_ms: u64,
    /// LLM start → first audio chunk played.
    pub ttfb_ms: u64,
    /// Utterance ready → last audio chunk played.
    pub total_ms: u64,
    /// Per-turn gate verdicts.
    pub gates: TurnGateVerdicts,
}

/// One ledger row: one scenario on one machine with one config.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchRecord {
    /// Row-shape version ([`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// [`MODE_FIXTURE`] today; `live` shares this schema later.
    pub mode: &'static str,
    /// A completed measurement or an explicit N/A record.
    pub status: BenchStatus,
    /// Named profile (`default`, `llm-qwen3.5-0.8b`, …).
    pub profile: String,
    /// Measured model ids.
    pub config: ConfigIds,
    /// Machine + build identity.
    pub fingerprint: Fingerprint,
    /// Scenario id from the corpus.
    pub scenario_id: String,
    /// Scenario category bucket.
    pub category: String,
    /// Per-turn measurements and verdicts.
    pub turns: Vec<TurnRecord>,
    /// Scenario-level verdicts (turn count, keywords, TTS→ASR).
    pub gates: ScenarioGateVerdicts,
    /// Turns dropped by recoverable provider errors or blank STT.
    pub skipped_turns: usize,
    /// All per-turn and scenario-level gates passed.
    pub passed: bool,
    /// Why this record is N/A. Never present for a completed measurement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
}

impl BenchRecord {
    /// Make an explicit N/A row for a profile that could not load. This keeps
    /// a failed model visible in JSONL without pretending that its gates ran.
    pub fn unavailable(options: &BenchOptions, scenario: &Scenario, reason: &str) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            mode: options.mode,
            status: BenchStatus::Unavailable,
            profile: options.profile.clone(),
            config: options.config.clone(),
            fingerprint: Fingerprint::collect(),
            scenario_id: scenario.id.clone(),
            category: scenario.category.clone(),
            turns: Vec::new(),
            gates: crate::eval::gates::scenario_verdicts(
                scenario.expected_turns(),
                0,
                &[],
                &scenario.expect,
                None,
            ),
            skipped_turns: 0,
            passed: false,
            unavailable_reason: Some(reason.to_string()),
        }
    }
}

/// Write records as JSONL, creating parent directories.
pub fn write_jsonl(path: &Path, records: &[BenchRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut out = std::io::BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut out, record)
            .map_err(|err| Error::Io(std::io::Error::other(err)))?;
        out.write_all(b"\n")?;
    }
    out.flush()?;
    Ok(())
}

/// `<profile>-<os>-<arch>-<sha12>.jsonl` under the repo's runs directory.
///
/// This is where `scripts/contribute-performance.sh` points contributors by
/// default; CI concatenates everything below `docs/eval/runs/`.
pub fn default_output_path(config: &ConfigIds) -> PathBuf {
    let fingerprint = Fingerprint::collect();
    let sha: String = fingerprint.git_sha.chars().take(12).collect();
    PathBuf::from("docs")
        .join("eval")
        .join("runs")
        .join(format!(
            "{}-{}-{}-{}.jsonl",
            config.profile_name(),
            fingerprint.os,
            fingerprint.arch,
            sha
        ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::{SttModel, TtsModel};
    use crate::eval::fingerprint::BUILD_PROFILE;
    use crate::eval::gates;
    use crate::eval::scenario;

    fn sample_record() -> BenchRecord {
        let expect = scenario::Expectation::default();
        let turn_gates = gates::turn_verdicts("hello", "hello there", "hello there", "hello", 0.8);
        let scenario_gates = gates::scenario_verdicts(1, 1, &["hello there"], &expect, Some(0.9));
        BenchRecord {
            schema_version: SCHEMA_VERSION,
            mode: MODE_FIXTURE,
            status: BenchStatus::Measured,
            profile: "default".to_string(),
            config: ConfigIds::v0(),
            fingerprint: Fingerprint {
                os: "macos".to_string(),
                arch: "aarch64".to_string(),
                cpu_model: "Apple M2".to_string(),
                ram_gb: Some(16.0),
                build_profile: "release",
                syllabix_version: "0.1.0".to_string(),
                git_sha: "abc123def456".to_string(),
            },
            scenario_id: "greeting_001".to_string(),
            category: "smalltalk".to_string(),
            turns: vec![TurnRecord {
                index: 0,
                prompt: "hello".to_string(),
                stt_text: "hello".to_string(),
                reply: "hello there".to_string(),
                speak_text: "hello there".to_string(),
                stt_ms: 120,
                ttft_ms: 80,
                ttfb_ms: 210,
                total_ms: 1_500,
                gates: turn_gates,
            }],
            gates: scenario_gates,
            skipped_turns: 0,
            passed: true,
            unavailable_reason: None,
        }
    }

    #[test]
    fn jsonl_is_one_object_per_line_and_round_trips() {
        // The process id isolates concurrent `cargo test` processes while
        // keeping the fixture path deterministic (no clock-based randomness).
        let dir = std::env::temp_dir().join(format!(
            "syllabix-eval-report-jsonl-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("nested").join("run.jsonl");
        write_jsonl(&path, &[sample_record(), sample_record()]).expect("write");

        let text = std::fs::read_to_string(&path).expect("read");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        let parsed: serde_json::Value = serde_json::from_str(lines[0]).expect("json");
        assert_eq!(parsed["schema_version"], SCHEMA_VERSION);
        assert_eq!(parsed["mode"], MODE_FIXTURE);
        assert_eq!(parsed["config"]["llm"], "llama-3.2-1b");
        assert_eq!(parsed["fingerprint"]["cpu_model"], "Apple M2");
        assert_eq!(parsed["turns"][0]["stt_ms"], 120);
        assert_eq!(parsed["gates"]["expected_turns"], 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn snake_case_keys_survive_serialization() {
        let value = serde_json::to_value(sample_record()).expect("value");
        assert!(value.get("scenario_id").is_some());
        assert!(value.get("skipped_turns").is_some());
        let turn = &value["turns"][0];
        assert!(turn.get("stt_text").is_some());
        assert!(turn.get("ttft_ms").is_some());
        assert!(turn["gates"].get("spoken_markdown_free").is_some());
    }

    #[test]
    fn profile_names_follow_the_one_axis_rule() {
        assert_eq!(ConfigIds::v0().profile_name(), "default");
        let mut ids = ConfigIds::v0();
        ids.stt = "medium".to_string();
        assert_eq!(ids.profile_name(), "stt-medium");
        let mut ids = ConfigIds::v0();
        ids.llm = "qwen3.5-0.8b".to_string();
        assert_eq!(ids.profile_name(), "llm-qwen3.5-0.8b");
        let mut ids = ConfigIds::v0();
        ids.tts = TtsModel::Qwen06.as_str().to_string();
        assert_eq!(ids.profile_name(), "tts-qwen3-0.6");
        // Two axes at once stay honest about what moved.
        let mut ids = ConfigIds::v0();
        ids.stt = SttModel::Medium.as_str().to_string();
        ids.tts = "qwen3-1.7".to_string();
        assert_eq!(ids.profile_name(), "stt-medium+tts-qwen3-1.7");
    }

    #[test]
    fn default_output_path_lands_under_docs_eval_runs() {
        let path = default_output_path(&ConfigIds::v0());
        let as_text = path.to_string_lossy().replace('\\', "/");
        assert!(as_text.starts_with("docs/eval/runs/"), "{as_text}");
        assert!(as_text.ends_with(".jsonl"));
        assert!(as_text.contains("default-"), "{as_text}");
    }

    #[test]
    fn build_profile_constant_matches_cfg() {
        assert!(
            BUILD_PROFILE == "release" || BUILD_PROFILE == "debug",
            "{BUILD_PROFILE}"
        );
    }
}
