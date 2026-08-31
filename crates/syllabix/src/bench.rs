//! Contributor-only component benchmarks.  This intentionally does not run
//! the voice loop: no VAD, AEC, capture, playback, queues, or device pacing.

use std::fs::{self, OpenOptions};
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames},
    build_llm, build_tts, word_match_ratio, AgentConfig, Cancel, Error, HistoryTurn, HttpFetcher,
    Llm, ModelCache, Result, StderrProgress, Stt, TokenChunk, Transcript, Tts, TurnId, Utterance,
    WhisperStt, LIBRISPEECH_MIN_WORD_MATCH,
};

const SCHEMA_VERSION: u8 = 1;
const SCENARIOS_JSONL: &str = include_str!("../../../docs/eval/scenarios.jsonl");

#[derive(Debug, Deserialize)]
struct Scenario {
    component: String,
    case: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    expected_words: Vec<String>,
    #[serde(default)]
    history: Vec<HistoryCase>,
}

#[derive(Debug, Deserialize)]
struct HistoryCase {
    user: String,
    assistant: String,
}

#[derive(Serialize)]
struct Fingerprint {
    os: String,
    arch: String,
    cpu: String,
    cpu_cores: usize,
    ram_bytes: Option<u64>,
    binary_version: &'static str,
    git_sha: &'static str,
}

#[derive(Serialize)]
struct Record {
    schema_version: u8,
    fingerprint: Fingerprint,
    component: &'static str,
    model: String,
    case: String,
    elapsed_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_output_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_units: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    word_match: Option<f64>,
    passed: bool,
}

pub(crate) fn run(out: PathBuf) -> Result<()> {
    if out.exists() {
        return Err(Error::Config {
            field: "bench.out".into(),
            message: format!("refusing to overwrite {}", out.display()),
        });
    }
    let config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    if !matches!(config.llm, syllabix_core::LlmProvider::Local) {
        return Err(Error::Config {
            field: "pipeline.llm.provider".into(),
            message: "bench measures local component ids only".into(),
        });
    }

    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let scenarios = scenarios()?;
    let fingerprint = fingerprint();
    let mut records = Vec::new();

    // Benchmark one provider at a time. This keeps the first-run fetch path
    // honest and avoids co-resident model memory affecting any component's
    // measurement. Do not replace this with `load_real_providers`: that also
    // loads Silero/VAD and all three providers before a single metric exists.
    {
        let mut stt = WhisperStt::from_cache(
            &cache,
            &HttpFetcher,
            &mut progress,
            &cancel,
            config.stt_model,
        )?
        .with_language(&config.language)?;
        records.extend(benchmark_stt(
            &mut stt,
            &config,
            &cancel,
            &fingerprint,
            &scenarios,
        )?);
    }
    {
        let mut llm = build_llm(&cache, &HttpFetcher, &mut progress, &cancel, &config, None)?;
        records.extend(benchmark_llm(
            &mut llm,
            &config,
            &cancel,
            &fingerprint,
            &scenarios,
        )?);
    }
    {
        let mut tts = build_tts(&cache, &HttpFetcher, &mut progress, &cancel, &config)?;
        records.extend(benchmark_tts(
            &mut tts,
            &config,
            &cancel,
            &fingerprint,
            &scenarios,
        )?);
    }
    write_jsonl(&out, &records)?;
    println!(
        "wrote {} component benchmark records to {}",
        records.len(),
        out.display()
    );
    Ok(())
}

fn benchmark_stt(
    stt: &mut impl Stt,
    config: &AgentConfig,
    cancel: &Cancel,
    fingerprint: &Fingerprint,
    scenarios: &[Scenario],
) -> Result<Vec<Record>> {
    scenarios
        .iter()
        .filter(|scenario| scenario.component == "stt")
        .map(|scenario| {
            let wav = read_wav(Cursor::new(stt_fixture(&scenario.case)?))?;
            let utterance = Utterance {
                turn: TurnId(0),
                frames: record_fixture_to_frames(&wav)?,
            };
            let started = Instant::now();
            let transcript = stt.transcribe(&utterance, cancel)?;
            let elapsed_ms = started.elapsed().as_millis();
            let expected: Vec<&str> = scenario.expected_words.iter().map(String::as_str).collect();
            let ratio = word_match_ratio(&transcript.text, &expected);
            Ok(Record {
                schema_version: SCHEMA_VERSION,
                fingerprint: fingerprint.clone(),
                component: "stt",
                model: config.stt_model.as_str().into(),
                case: scenario.case.clone(),
                elapsed_ms,
                first_output_ms: None,
                output_units: Some(transcript.text.len()),
                word_match: Some(ratio),
                passed: ratio >= LIBRISPEECH_MIN_WORD_MATCH,
            })
        })
        .collect()
}

fn benchmark_llm(
    llm: &mut impl Llm,
    config: &AgentConfig,
    cancel: &Cancel,
    fingerprint: &Fingerprint,
    scenarios: &[Scenario],
) -> Result<Vec<Record>> {
    scenarios
        .iter()
        .filter(|scenario| scenario.component == "llm")
        .enumerate()
        .map(|(index, scenario)| {
            let started = Instant::now();
            let mut first_output_ms = None;
            let mut output = String::new();
            let user = Transcript {
                turn: TurnId(index as u64),
                text: scenario.text.clone(),
                language: config.language.clone(),
            };
            let history: Vec<HistoryTurn> = scenario
                .history
                .iter()
                .enumerate()
                .map(|(turn, prior)| HistoryTurn {
                    user: Transcript {
                        turn: TurnId(turn as u64),
                        text: prior.user.clone(),
                        language: config.language.clone(),
                    },
                    assistant: prior.assistant.clone(),
                })
                .collect();
            llm.generate(&history, &user, cancel, &mut |token| {
                first_output_ms.get_or_insert_with(|| started.elapsed().as_millis());
                output.push_str(&token.text);
                Ok(())
            })?;
            Ok(Record {
                schema_version: SCHEMA_VERSION,
                fingerprint: fingerprint.clone(),
                component: "llm",
                model: config.llm_model.clone(),
                case: scenario.case.clone(),
                elapsed_ms: started.elapsed().as_millis(),
                first_output_ms,
                output_units: Some(output.split_whitespace().count()),
                word_match: None,
                passed: !output.trim().is_empty() && !output.contains("<think>"),
            })
        })
        .collect()
}

fn benchmark_tts(
    tts: &mut impl Tts,
    config: &AgentConfig,
    cancel: &Cancel,
    fingerprint: &Fingerprint,
    scenarios: &[Scenario],
) -> Result<Vec<Record>> {
    scenarios
        .iter()
        .filter(|scenario| scenario.component == "tts")
        .enumerate()
        .map(|(index, scenario)| {
            let started = Instant::now();
            let mut first_output_ms = None;
            let mut samples = 0usize;
            let token = TokenChunk {
                turn: TurnId(index as u64),
                // These are independent synthesis calls, not barge-in
                // generations. The standalone cancel state remains live at
                // generation zero for the full benchmark.
                generation: cancel.generation(),
                index: 0,
                text: scenario.text.clone(),
                is_last: true,
            };
            tts.synthesize_chunk_into(&token, cancel, &mut |audio| {
                first_output_ms.get_or_insert_with(|| started.elapsed().as_millis());
                samples += audio.samples.len();
                Ok(())
            })?;
            Ok(Record {
                schema_version: SCHEMA_VERSION,
                fingerprint: fingerprint.clone(),
                component: "tts",
                model: config.tts_model.as_str().into(),
                case: scenario.case.clone(),
                elapsed_ms: started.elapsed().as_millis(),
                first_output_ms,
                output_units: Some(samples),
                word_match: None,
                passed: samples > 0,
            })
        })
        .collect()
}

fn scenarios() -> Result<Vec<Scenario>> {
    let scenarios: std::result::Result<Vec<Scenario>, _> = SCENARIOS_JSONL
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect();
    let scenarios = scenarios.map_err(|err| Error::Config {
        field: "bench.scenarios".into(),
        message: err.to_string(),
    })?;
    if scenarios.iter().any(|scenario| {
        scenario.component != "stt" && scenario.component != "llm" && scenario.component != "tts"
    }) {
        return Err(Error::Config {
            field: "bench.scenarios".into(),
            message: "component must be stt, llm, or tts".into(),
        });
    }
    Ok(scenarios)
}

fn stt_fixture(case: &str) -> Result<&'static [u8]> {
    match case {
        "jfk" => Ok(include_bytes!(
            "../../syllabix-core/tests/fixtures/stt/jfk.wav"
        )),
        "librispeech-121" => Ok(include_bytes!(
            "../../syllabix-core/tests/fixtures/stt/librispeech-121-127105-0009.wav"
        )),
        "librispeech-1995" => Ok(include_bytes!(
            "../../syllabix-core/tests/fixtures/stt/librispeech-1995-1837-0005.wav"
        )),
        _ => Err(Error::Config {
            field: "bench.scenarios".into(),
            message: format!("unknown STT fixture {case:?}"),
        }),
    }
}

fn fingerprint() -> Fingerprint {
    Fingerprint {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu: cpu_name(),
        cpu_cores: std::thread::available_parallelism().map_or(0, usize::from),
        ram_bytes: ram_bytes(),
        binary_version: env!("CARGO_PKG_VERSION"),
        git_sha: option_env!("SYLLABIX_GIT_SHA").unwrap_or("unknown"),
    }
}

fn cpu_name() -> String {
    #[cfg(target_os = "macos")]
    let value = std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok());
    #[cfg(target_os = "linux")]
    let value = fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
        text.lines()
            .find_map(|line| {
                line.split_once(':')
                    .filter(|(key, _)| key.trim() == "model name")
            })
            .map(|(_, value)| value.trim().to_owned())
    });
    #[cfg(target_os = "windows")]
    let value = std::env::var("PROCESSOR_IDENTIFIER").ok();
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let value: Option<String> = None;
    value
        .map(|name| name.trim().to_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(target_os = "linux")]
fn ram_bytes() -> Option<u64> {
    fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
        text.lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))
            .and_then(|value| value.split_whitespace().next())
            .and_then(|kib| kib.parse::<u64>().ok())
            .map(|kib| kib * 1024)
    })
}

#[cfg(target_os = "macos")]
fn ram_bytes() -> Option<u64> {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|value| value.trim().parse().ok())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn ram_bytes() -> Option<u64> {
    None
}

fn write_jsonl(path: &std::path::Path, records: &[Record]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
    fs::create_dir_all(parent)?;
    let mut contents = String::new();
    for record in records {
        contents.push_str(&serde_json::to_string(record).map_err(|err| Error::Config {
            field: "bench.output".into(),
            message: err.to_string(),
        })?);
        contents.push('\n');
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    use std::io::Write;
    file.write_all(contents.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

impl Clone for Fingerprint {
    fn clone(&self) -> Self {
        Self {
            os: self.os.clone(),
            arch: self.arch.clone(),
            cpu: self.cpu.clone(),
            cpu_cores: self.cpu_cores,
            ram_bytes: self.ram_bytes,
            binary_version: self.binary_version,
            git_sha: self.git_sha,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_creates_jsonl_and_refuses_overwrite() {
        let path =
            std::env::temp_dir().join(format!("syllabix-bench-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&path);
        let record = Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fingerprint(),
            component: "tts",
            model: "kokoro".into(),
            case: "test".into(),
            elapsed_ms: 1,
            first_output_ms: Some(0),
            output_units: Some(1),
            word_match: None,
            passed: true,
        };
        write_jsonl(&path, &[record]).expect("write");
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 1);
        assert!(write_jsonl(&path, &[]).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn embedded_corpus_covers_each_isolated_component() {
        let corpus = scenarios().expect("parse embedded JSONL");
        assert!(corpus.iter().any(|scenario| scenario.component == "stt"));
        assert!(corpus.iter().any(|scenario| scenario.component == "llm"));
        assert!(corpus.iter().any(|scenario| scenario.component == "tts"));
        assert!(corpus
            .iter()
            .filter(|scenario| scenario.component == "stt")
            .all(|scenario| !scenario.expected_words.is_empty()));
        assert_eq!(
            corpus
                .iter()
                .filter(|scenario| scenario.component == "llm")
                .count(),
            5
        );
        assert!(corpus.iter().any(|scenario| {
            scenario.component == "llm"
                && scenario.case == "history-follow-up"
                && !scenario.history.is_empty()
        }));
    }
}
