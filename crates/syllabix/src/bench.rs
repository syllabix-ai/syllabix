//! Contributor-only component benchmarks.  This intentionally does not run
//! the voice loop: no VAD, AEC, capture, playback, queues, or device pacing.

use std::fs::{self, OpenOptions};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames, FrameSplitter, WavPcm},
    build_stt, build_tts, contains_words_in_order, process_rss_bytes, word_match_ratio,
    AgentConfig, Cancel, Error, HistoryTurn, HttpFetcher, LlamaLlm, Llm, ModelCache, Result,
    StderrProgress, Stt, SttModel, TokenChunk, Transcript, Tts, TurnId, Utterance, WhisperStt,
    DEFAULT_SAMPLE_RATE_HZ, TTS_ASR_MIN_WORD_MATCH,
};

const SCHEMA_VERSION: u8 = 3;
const SCENARIOS_JSONL: &str = include_str!("../../../docs/eval/scenarios.jsonl");
const NUMBER_FIXTURES: [&str; 5] = ["100", "$5", "3:45 pm", "USD", "API"];

#[derive(Debug, Deserialize)]
struct Scenario {
    component: String,
    case: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    expected_words: Vec<String>,
    #[serde(default = "default_word_match_threshold")]
    min_word_match: f64,
    #[serde(default)]
    history: Vec<HistoryCase>,
    #[serde(default)]
    numbers: bool,
    #[serde(default)]
    required_keywords_any: Vec<String>,
    #[serde(default)]
    forbidden_keywords: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HistoryCase {
    user: String,
    assistant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Fingerprint {
    os: String,
    arch: String,
    cpu: String,
    cpu_cores: usize,
    ram_bytes: Option<u64>,
    binary_version: String,
    git_sha: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    schema_version: u8,
    fingerprint: Fingerprint,
    component: String,
    model: String,
    case: String,
    elapsed_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_output_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_units: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    word_match: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    word_match_threshold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_audio_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_audio_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    real_time_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transcript: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_before_load_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_after_load_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_peak_bytes: Option<usize>,
    passed: bool,
}

fn default_word_match_threshold() -> f64 {
    TTS_ASR_MIN_WORD_MATCH
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

    let mut records = Vec::new();

    // Each selected provider runs in a dedicated process so host-RSS snapshots
    // cover one model lifetime rather than earlier components.
    records.extend(run_named_worker("bench-asr-worker")?);
    records.extend(run_named_worker("bench-llm-worker")?);
    records.extend(run_named_worker("bench-tts-worker")?);
    write_jsonl(&out, &records)?;
    println!(
        "wrote {} component benchmark records to {}",
        records.len(),
        out.display()
    );
    Ok(())
}

/// Private `bench-asr-worker` entry point. A distinct process gives the ASR
/// memory snapshot one model lifetime with no LLM or TTS residency.
pub(crate) fn run_asr_worker(out: PathBuf) -> Result<()> {
    if out.exists() {
        return Err(Error::Config {
            field: "bench.worker.out".into(),
            message: format!("refusing to overwrite {}", out.display()),
        });
    }
    let config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    let scenarios = scenarios()?;
    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let memory_before_load = required_rss("before ASR model load")?;
    let mut stt = build_stt(&cache, &HttpFetcher, &mut progress, &cancel, &config)?
        .with_language(&config.language)?;
    let memory_after_load = required_rss("after ASR model load")?;
    let records = benchmark_stt(
        &mut stt,
        &config,
        &cancel,
        &fingerprint(),
        &scenarios,
        memory_before_load,
        memory_after_load,
    )?;
    write_jsonl(&out, &records)
}

/// Private `bench-llm-worker` entry point. It owns precisely one local GGUF
/// for comparable before-load, after-load, and peak host-RSS measurements.
pub(crate) fn run_llm_worker(out: PathBuf) -> Result<()> {
    if out.exists() {
        return Err(Error::Config {
            field: "bench.worker.out".into(),
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
    let memory_before_load = required_rss("before LLM model load")?;
    let mut llm = LlamaLlm::from_cached_model(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        &config.llm_model,
        config.thinking,
    )?
    .with_system_prompt(config.system_prompt.clone());
    let memory_after_load = required_rss("after LLM model load")?;
    let records = benchmark_llm(
        &mut llm,
        &config,
        &cancel,
        &fingerprint(),
        &scenarios()?,
        memory_before_load,
        memory_after_load,
    )?;
    write_jsonl(&out, &records)
}

/// Private `bench-tts-worker` entry point. TTS host-RSS is captured before
/// Whisper `small` loads as the TTS→ASR scorer.
pub(crate) fn run_tts_worker(out: PathBuf) -> Result<()> {
    if out.exists() {
        return Err(Error::Config {
            field: "bench.worker.out".into(),
            message: format!("refusing to overwrite {}", out.display()),
        });
    }
    let config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    let scenarios = scenarios()?;
    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let memory_before_load = required_rss("before TTS model load")?;
    let mut tts = build_tts(&cache, &HttpFetcher, &mut progress, &cancel, &config)?;
    let memory_after_load = required_rss("after TTS model load")?;
    let records = benchmark_tts(
        &mut tts,
        &config,
        &cancel,
        &fingerprint(),
        &scenarios,
        memory_before_load,
        memory_after_load,
    )?;
    write_jsonl(&out, &records)
}

fn run_named_worker(subcommand: &str) -> Result<Vec<Record>> {
    let temp = std::env::temp_dir().join(format!(
        "syllabix-{subcommand}-{}-{}.jsonl",
        std::process::id(),
        unique_suffix()
    ));
    let output = Command::new(std::env::current_exe()?)
        .args([subcommand, "--out"])
        .arg(&temp)
        .output()?;
    if !output.status.success() {
        let _ = fs::remove_file(&temp);
        return Err(Error::Provider {
            provider: "bench-worker",
            message: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    let contents = fs::read_to_string(&temp)?;
    fs::remove_file(&temp)?;
    contents
        .lines()
        .map(|line| {
            serde_json::from_str(line).map_err(|err| Error::Config {
                field: "bench.worker.output".into(),
                message: err.to_string(),
            })
        })
        .collect()
}

fn unique_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}

fn required_rss(when: &str) -> Result<usize> {
    process_rss_bytes().ok_or_else(|| Error::Provider {
        provider: "bench",
        message: format!("host RSS is unavailable {when}"),
    })
}

fn benchmark_stt(
    stt: &mut impl Stt,
    config: &AgentConfig,
    cancel: &Cancel,
    fingerprint: &Fingerprint,
    scenarios: &[Scenario],
    memory_before_load: usize,
    memory_after_load: usize,
) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    let mut memory_peak = memory_after_load;
    for scenario in scenarios
        .iter()
        .filter(|scenario| scenario.component == "stt")
    {
        let wav = read_wav(Cursor::new(stt_fixture(&scenario.case)?))?;
        let input_audio_ms = wav_duration_ms(&wav);
        let utterance = Utterance {
            turn: TurnId(0),
            frames: record_fixture_to_frames(&wav)?,
        };
        let started = Instant::now();
        let transcript = stt.transcribe(&utterance, cancel)?;
        let elapsed_ms = started.elapsed().as_millis();
        let expected: Vec<&str> = scenario.expected_words.iter().map(String::as_str).collect();
        let ratio = word_match_ratio(&transcript.text, &expected);
        memory_peak = memory_peak.max(required_rss("during ASR benchmark")?);
        records.push(Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fingerprint.clone(),
            component: "stt".into(),
            model: config.stt_model.as_str().into(),
            case: scenario.case.clone(),
            elapsed_ms,
            // whisper.cpp returns a completed transcript, not partial text.
            first_output_ms: Some(elapsed_ms),
            output_units: Some(transcript.text.len()),
            prompt_tokens: None,
            generated_tokens: None,
            prompt_tokens_per_second: None,
            generated_tokens_per_second: None,
            word_match: Some(ratio),
            word_match_threshold: Some(scenario.min_word_match),
            input_audio_ms: Some(input_audio_ms),
            generated_audio_ms: None,
            real_time_factor: Some(real_time_factor(elapsed_ms, input_audio_ms)),
            transcript: Some(transcript.text),
            response: None,
            memory_before_load_bytes: Some(memory_before_load),
            memory_after_load_bytes: Some(memory_after_load),
            memory_peak_bytes: None,
            passed: ratio >= scenario.min_word_match,
        });
    }
    for record in &mut records {
        record.memory_peak_bytes = Some(memory_peak);
    }
    Ok(records)
}

fn benchmark_llm(
    llm: &mut LlamaLlm,
    config: &AgentConfig,
    cancel: &Cancel,
    fingerprint: &Fingerprint,
    scenarios: &[Scenario],
    memory_before_load: usize,
    memory_after_load: usize,
) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    let mut memory_peak = memory_after_load;
    for (index, scenario) in scenarios
        .iter()
        .filter(|scenario| scenario.component == "llm")
        .enumerate()
    {
        let started = Instant::now();
        let mut first_output_ms = None;
        let mut output = String::new();
        let mut generated_tokens = 0usize;
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
        let prompt_tokens = llm.benchmark_prompt_tokens(&history, &user)?;
        llm.generate(&history, &user, cancel, &mut |token| {
            first_output_ms.get_or_insert_with(|| started.elapsed().as_millis());
            generated_tokens += 1;
            output.push_str(&token.text);
            Ok(())
        })?;
        let elapsed_ms = started.elapsed().as_millis();
        memory_peak = memory_peak.max(required_rss("during LLM benchmark")?);
        let passed = llm_response_passes(scenario, &output);
        records.push(Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fingerprint.clone(),
            component: "llm".into(),
            model: config.llm_model.clone(),
            case: scenario.case.clone(),
            elapsed_ms,
            first_output_ms,
            output_units: Some(generated_tokens),
            prompt_tokens: Some(prompt_tokens),
            generated_tokens: Some(generated_tokens),
            // Prompt throughput is prefill-to-first-token. Generation
            // throughput starts after that first emitted token.
            prompt_tokens_per_second: Some(tokens_per_second(
                prompt_tokens,
                first_output_ms.unwrap_or(elapsed_ms),
            )),
            generated_tokens_per_second: Some(tokens_per_second(
                generated_tokens,
                elapsed_ms.saturating_sub(first_output_ms.unwrap_or(elapsed_ms)),
            )),
            word_match: None,
            word_match_threshold: None,
            input_audio_ms: None,
            generated_audio_ms: None,
            real_time_factor: None,
            transcript: None,
            response: Some(output),
            memory_before_load_bytes: None,
            memory_after_load_bytes: None,
            memory_peak_bytes: None,
            passed,
        });
    }
    for record in &mut records {
        record.memory_before_load_bytes = Some(memory_before_load);
        record.memory_after_load_bytes = Some(memory_after_load);
        record.memory_peak_bytes = Some(memory_peak);
    }
    Ok(records)
}

fn tokens_per_second(tokens: usize, elapsed_ms: u128) -> f64 {
    tokens as f64 / (elapsed_ms.max(1) as f64 / 1_000.0)
}

fn llm_response_passes(scenario: &Scenario, response: &str) -> bool {
    let response = response.to_lowercase();
    !response.trim().is_empty()
        && !response.contains("<think>")
        && !response.contains("</think>")
        && (scenario.required_keywords_any.is_empty()
            || scenario
                .required_keywords_any
                .iter()
                .any(|keyword| response.contains(&keyword.to_lowercase())))
        && scenario
            .forbidden_keywords
            .iter()
            .all(|keyword| !response.contains(&keyword.to_lowercase()))
}

fn benchmark_tts(
    tts: &mut impl Tts,
    config: &AgentConfig,
    cancel: &Cancel,
    fingerprint: &Fingerprint,
    scenarios: &[Scenario],
    memory_before_load: usize,
    memory_after_load: usize,
) -> Result<Vec<Record>> {
    let tts_scenarios: Vec<&Scenario> = scenarios
        .iter()
        .filter(|scenario| scenario.component == "tts")
        .collect();
    let mut records = Vec::new();
    let mut pcm_by_case = Vec::new();
    let mut memory_peak = memory_after_load;
    for (index, scenario) in tts_scenarios.iter().enumerate() {
        let started = Instant::now();
        let mut first_output_ms = None;
        let mut samples = Vec::new();
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
            samples.extend_from_slice(&audio.samples);
            Ok(())
        })?;
        let elapsed_ms = started.elapsed().as_millis();
        let generated_audio_ms = pcm_duration_ms(samples.len());
        memory_peak = memory_peak.max(required_rss("during TTS benchmark")?);
        records.push(Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fingerprint.clone(),
            component: "tts".into(),
            model: config.tts_model.as_str().into(),
            case: scenario.case.clone(),
            elapsed_ms,
            first_output_ms,
            output_units: Some(samples.len()),
            prompt_tokens: None,
            generated_tokens: None,
            prompt_tokens_per_second: None,
            generated_tokens_per_second: None,
            word_match: None,
            word_match_threshold: None,
            input_audio_ms: None,
            generated_audio_ms: Some(generated_audio_ms),
            real_time_factor: Some(real_time_factor(elapsed_ms, generated_audio_ms)),
            transcript: None,
            response: None,
            memory_before_load_bytes: Some(memory_before_load),
            memory_after_load_bytes: Some(memory_after_load),
            memory_peak_bytes: None,
            passed: !samples.is_empty(),
        });
        pcm_by_case.push(samples);
    }
    for record in &mut records {
        record.memory_peak_bytes = Some(memory_peak);
    }

    // Freeze TTS RSS before loading the scorer. Whisper `small` is the
    // canonical TTS→ASR ear; it is not part of the selected TTS axis.
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut stt =
        WhisperStt::from_cache(&cache, &HttpFetcher, &mut progress, cancel, SttModel::Small)?
            .with_language(&config.language)?;
    for ((record, scenario), samples) in records
        .iter_mut()
        .zip(tts_scenarios.iter())
        .zip(pcm_by_case.iter())
    {
        if samples.is_empty() {
            record.passed = false;
            continue;
        }
        let transcript = stt.transcribe(&pcm_to_utterance(samples)?, cancel)?;
        record.transcript = Some(transcript.text.clone());
        if scenario.numbers {
            record.passed = number_heavy_passed(&transcript.text);
        } else {
            let expected: Vec<&str> = scenario.expected_words.iter().map(String::as_str).collect();
            let ratio = word_match_ratio(&transcript.text, &expected);
            record.word_match = Some(ratio);
            record.word_match_threshold = Some(scenario.min_word_match);
            record.passed = ratio >= scenario.min_word_match;
        }
    }
    Ok(records)
}

fn pcm_to_utterance(samples: &[i16]) -> Result<Utterance> {
    let mut splitter = FrameSplitter::new();
    let mut frames = splitter.push(samples)?;
    frames.extend(splitter.flush()?);
    if frames.is_empty() {
        return Err(Error::Provider {
            provider: "bench",
            message: "TTS PCM produced no 16 kHz frames".into(),
        });
    }
    Ok(Utterance {
        turn: TurnId(0),
        frames,
    })
}

fn pcm_duration_ms(samples: usize) -> u128 {
    (samples as u128 * 1_000) / u128::from(DEFAULT_SAMPLE_RATE_HZ).max(1)
}

/// Listener-shaped checks for the number-heavy fixture. Whisper may emit
/// `$100` for a correctly spoken "one hundred"; digit-by-digit "zero" is a
/// fail. Failed verdicts stay on the record.
fn number_heavy_passed(transcript: &str) -> bool {
    NUMBER_FIXTURES
        .iter()
        .all(|fixture| number_fixture_passed(fixture, transcript))
}

fn number_fixture_passed(fixture: &str, transcript: &str) -> bool {
    let heard = transcript.to_ascii_lowercase();
    match fixture {
        "100" => (heard.contains("hundred") || heard.contains("100")) && !heard.contains("zero"),
        "$5" => {
            heard.contains("five")
                || heard.contains("$5")
                || heard.contains("5 dollar")
                || heard.contains("5 dollars")
        }
        "3:45 pm" => {
            let has_pm = heard.contains("pm")
                || heard.contains("p.m")
                || contains_words_in_order(transcript, &["p", "m"]);
            has_pm
                && (heard.contains("3:45")
                    || heard.contains("3 45")
                    || heard.contains("345")
                    || heard.contains("45")
                    || (heard.contains("three")
                        && (heard.contains("forty") || heard.contains("four"))))
        }
        "USD" => {
            heard.contains("usd")
                || heard.contains("us dollars")
                || heard.contains("u.s.d")
                || contains_words_in_order(transcript, &["u", "s", "d"])
        }
        "API" => heard.contains("api") || contains_words_in_order(transcript, &["a", "p", "i"]),
        _ => false,
    }
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
    if scenarios.iter().any(|scenario| {
        scenario.component == "tts"
            && !scenario.numbers
            && (scenario.expected_words.is_empty()
                || !(TTS_ASR_MIN_WORD_MATCH..=1.0).contains(&scenario.min_word_match))
    }) {
        return Err(Error::Config {
            field: "bench.scenarios".into(),
            message: "TTS intelligibility cases need expected words and a 0.8–1.0 gate".into(),
        });
    }
    Ok(scenarios)
}

fn stt_fixture(case: &str) -> Result<&'static [u8]> {
    match case {
        "jfk" => Ok(include_bytes!(
            "../../syllabix-core/tests/fixtures/stt/jfk.wav"
        )),
        "librispeech-1089" => Ok(include_bytes!(
            "../../syllabix-core/tests/fixtures/stt/librispeech-1089-134686-0000.wav"
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
        binary_version: env!("CARGO_PKG_VERSION").into(),
        git_sha: option_env!("SYLLABIX_GIT_SHA").unwrap_or("unknown").into(),
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

fn write_jsonl(path: &Path, records: &[Record]) -> Result<()> {
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

fn wav_duration_ms(wav: &WavPcm) -> u128 {
    let samples_per_second =
        u128::from(wav.format.sample_rate_hz) * u128::from(wav.format.channels);
    (wav.samples.len() as u128 * 1_000) / samples_per_second.max(1)
}

fn real_time_factor(elapsed_ms: u128, input_audio_ms: u128) -> f64 {
    elapsed_ms as f64 / input_audio_ms.max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> Record {
        Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fingerprint(),
            component: "tts".into(),
            model: "kokoro".into(),
            case: "test".into(),
            elapsed_ms: 1,
            first_output_ms: Some(0),
            output_units: Some(1),
            prompt_tokens: None,
            generated_tokens: None,
            prompt_tokens_per_second: None,
            generated_tokens_per_second: None,
            word_match: None,
            word_match_threshold: None,
            input_audio_ms: None,
            generated_audio_ms: Some(1),
            real_time_factor: Some(1.0),
            transcript: None,
            response: None,
            memory_before_load_bytes: None,
            memory_after_load_bytes: None,
            memory_peak_bytes: None,
            passed: true,
        }
    }

    #[test]
    fn writer_creates_jsonl_and_refuses_overwrite() {
        let path =
            std::env::temp_dir().join(format!("syllabix-bench-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&path);
        write_jsonl(&path, &[sample_record()]).expect("write");
        assert_eq!(fs::read_to_string(&path).unwrap().lines().count(), 1);
        assert!(write_jsonl(&path, &[]).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn asr_duration_and_real_time_factor_use_input_audio() {
        let wav = WavPcm {
            format: syllabix_core::audio::PcmFormat {
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![0; 24_000],
        };
        assert_eq!(wav_duration_ms(&wav), 1_500);
        assert!((real_time_factor(375, wav_duration_ms(&wav)) - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn tts_rtf_uses_generated_audio_duration() {
        assert_eq!(pcm_duration_ms(16_000), 1_000);
        assert!((real_time_factor(500, pcm_duration_ms(16_000)) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn asr_corpus_has_short_and_long_hashed_fixtures() {
        let corpus = scenarios().expect("parse embedded JSONL");
        let durations: Vec<u128> = corpus
            .iter()
            .filter(|scenario| scenario.component == "stt")
            .map(|scenario| {
                let wav = read_wav(Cursor::new(stt_fixture(&scenario.case).expect("fixture")))
                    .expect("wav");
                assert!(!scenario.expected_words.is_empty());
                assert!((0.8..=1.0).contains(&scenario.min_word_match));
                wav_duration_ms(&wav)
            })
            .collect();
        assert!(durations.len() >= 2);
        assert!(durations.iter().max() > durations.iter().min());
    }

    #[test]
    fn tts_corpus_has_short_long_and_number_heavy_text() {
        let corpus = scenarios().expect("parse embedded JSONL");
        let tts: Vec<&Scenario> = corpus
            .iter()
            .filter(|scenario| scenario.component == "tts")
            .collect();
        let short = tts
            .iter()
            .find(|scenario| scenario.case == "short")
            .expect("short");
        let long = tts
            .iter()
            .find(|scenario| scenario.case == "long")
            .expect("long");
        let numbers = tts
            .iter()
            .find(|scenario| scenario.case == "numbers")
            .expect("numbers");
        assert!(short.text.len() < long.text.len());
        assert!(!short.expected_words.is_empty());
        assert!(!long.expected_words.is_empty());
        assert!(numbers.numbers);
        for fixture in NUMBER_FIXTURES {
            assert!(
                numbers.text.contains(fixture),
                "number-heavy text must include {fixture}"
            );
        }
    }

    #[test]
    fn number_heavy_verdict_accepts_listener_forms() {
        assert!(number_heavy_passed(
            "That costs one hundred dollars. Please charge five dollars at 3:45 pm in USD through the API."
        ));
        assert!(number_heavy_passed(
            "that costs $100. please charge $5 at 3:45 p.m. in u s d through the A.P.I."
        ));
    }

    #[test]
    fn number_heavy_verdict_rejects_digit_string_zero() {
        assert!(!number_fixture_passed(
            "100",
            "that costs one zero zero dollars"
        ));
        assert!(!number_heavy_passed(
            "that costs one zero zero dollars. please charge five at 3:45 pm in usd through the api."
        ));
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
        assert!(corpus.iter().any(|scenario| {
            scenario.component == "stt" && scenario.case == "librispeech-1089"
        }));
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
                && scenario
                    .required_keywords_any
                    .iter()
                    .any(|keyword| keyword == "lisbon")
        }));
        assert_eq!(
            corpus
                .iter()
                .filter(|scenario| scenario.component == "tts")
                .count(),
            3
        );
    }

    #[test]
    fn llm_response_gate_requires_behavior_without_exact_output() {
        let scenario = Scenario {
            component: "llm".into(),
            case: "history".into(),
            text: "Where am I going?".into(),
            expected_words: Vec::new(),
            min_word_match: 0.8,
            history: Vec::new(),
            numbers: false,
            required_keywords_any: vec!["lisbon".into()],
            forbidden_keywords: vec!["123".into()],
        };
        assert!(llm_response_passes(&scenario, "You said Lisbon."));
        assert!(!llm_response_passes(&scenario, "I do not know."));
        assert!(!llm_response_passes(
            &scenario,
            "Lisbon <think>hidden</think>"
        ));
        assert!(!llm_response_passes(&scenario, "Lisbon 123"));
    }

    #[test]
    fn token_rates_are_per_second_and_never_divide_by_zero() {
        assert!((tokens_per_second(25, 500) - 50.0).abs() < f64::EPSILON);
        assert_eq!(tokens_per_second(1, 0), 1_000.0);
    }
}
