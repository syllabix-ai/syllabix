//! Contributor-only component benchmarks.  This intentionally does not run
//! the voice loop: no VAD, AEC, capture, playback, queues, or device pacing.

use std::fs::{self, OpenOptions};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use syllabix_core::defaults::{SttModel, TtsModel};
use syllabix_core::llm::{is_v0_llm_model, LlamaLlm, V0_LLM_MODELS};
use syllabix_core::memory::process_rss_bytes;
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::real::{build_stt, build_tts, LiveStt, LiveTts};
use syllabix_core::stt::{contains_words_in_order, word_match_ratio, WhisperStt};
use syllabix_core::tts::TTS_ASR_MIN_WORD_MATCH;
use syllabix_core::types::DEFAULT_SAMPLE_RATE_HZ;
use syllabix_core::{
    audio::{read_wav, record_fixture_to_frames, FrameSplitter, WavPcm},
    AgentConfig, Cancel, Error, HistoryTurn, Llm, ModelCache, Result, Stt, TokenChunk, Transcript,
    Tts, TurnId, Utterance,
};

const SCHEMA_VERSION: u8 = 5;
const SCENARIOS_JSONL: &str = include_str!("../../../docs/eval/scenarios.jsonl");
const NUMBER_FIXTURES: [&str; 5] = ["100", "$5", "3:45 pm", "USD", "API"];
/// Safety rail for contributor LLM evidence. Product `run` still has no
/// `n_predict`; fixtures pass well under 100 generated tokens, so 256 keeps
/// gates meaningful while stopping a CPU host from filling the whole GGUF
/// window when a model never emits EOS.
const BENCH_LLM_MAX_GENERATED_TOKENS: usize = 256;

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
    /// ggml placement for STT/LLM: `cpu`, `metal`, or `vulkan`.
    ggml_backend: String,
    /// GPU marketing name when known (Vulkan device 0, or Apple chip on Metal).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gpu_name: Option<String>,
    /// Device-local VRAM (or unified memory on Metal) when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gpu_vram_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    schema_version: u8,
    fingerprint: Fingerprint,
    component: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    backend: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    wer: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ref_words: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    errors: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    noise: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    snr_db: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    threads: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    load_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    first_call_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    audio_ms: Option<u128>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    dataset: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    n_utt: Option<usize>,
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
    if !matches!(config.llm, syllabix_core::defaults::LlmProvider::Local) {
        return Err(Error::Config {
            field: "pipeline.llm.provider".into(),
            message: "bench measures local component ids only".into(),
        });
    }

    let mut records = Vec::new();

    // One-axis coverage: every supported local STT, LLM, and TTS id in its own
    // child process so host-RSS snapshots cover one model lifetime. Defaults
    // are included; there is no STT×LLM×TTS cross-product.
    for model in SttModel::ALL {
        records.extend(run_named_worker("bench-asr-worker", model.as_str())?);
    }
    for model in V0_LLM_MODELS {
        records.extend(run_named_worker("bench-llm-worker", model)?);
    }
    for model in TtsModel::ALL {
        records.extend(run_named_worker("bench-tts-worker", model.as_str())?);
    }
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_asr_worker(
    out: PathBuf,
    model: String,
    manifest: Option<PathBuf>,
    noise: String,
    snr: f32,
    seed: u64,
    warmup: usize,
    threads: Option<usize>,
    dataset: String,
) -> Result<()> {
    if out.exists() {
        return Err(Error::Config {
            field: "bench.worker.out".into(),
            message: format!("refusing to overwrite {}", out.display()),
        });
    }
    let mut config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    let stt_model = SttModel::parse(&model).ok_or_else(|| Error::Config {
        field: "bench.worker.model".into(),
        message: format!("unsupported STT model {model:?}"),
    })?;
    config.stt_model = stt_model;
    // Moonshine is English-only; force `en` when sweeping that axis so a
    // project yaml with another STT language cannot abort the worker.
    if stt_model.is_moonshine() {
        config.language = "en".into();
    }
    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let memory_before_load = required_rss("before ASR model load")?;
    let load_started = Instant::now();
    let mut stt = build_stt(&cache, &HttpFetcher, &mut progress, &cancel, &config, None)?
        .with_language(&config.language)?;
    if let Some(n) = threads {
        stt = stt.with_threads(i32::try_from(n).unwrap_or(i32::MAX).max(1));
    }
    let load_ms = load_started.elapsed().as_millis();
    let memory_after_load = required_rss("after ASR model load")?;
    let records = if let Some(manifest) = manifest {
        benchmark_stt_manifest(
            &mut stt,
            &config,
            &cancel,
            &fingerprint(),
            &manifest,
            &noise,
            snr,
            seed,
            warmup,
            load_ms,
            dataset,
            memory_before_load,
            memory_after_load,
        )?
    } else {
        benchmark_stt(
            &mut stt,
            &config,
            &cancel,
            &fingerprint(),
            &scenarios()?,
            memory_before_load,
            memory_after_load,
        )?
    };
    write_jsonl(&out, &records)
}

/// Private `bench-llm-worker` entry point. It owns precisely one local GGUF
/// for comparable before-load, after-load, and peak host-RSS measurements.
pub(crate) fn run_llm_worker(out: PathBuf, model: String) -> Result<()> {
    if out.exists() {
        return Err(Error::Config {
            field: "bench.worker.out".into(),
            message: format!("refusing to overwrite {}", out.display()),
        });
    }
    let mut config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    if !matches!(config.llm, syllabix_core::defaults::LlmProvider::Local) {
        return Err(Error::Config {
            field: "pipeline.llm.provider".into(),
            message: "bench measures local component ids only".into(),
        });
    }
    if !is_v0_llm_model(&model) {
        return Err(Error::Config {
            field: "bench.worker.model".into(),
            message: format!("unsupported local LLM model {model:?}"),
        });
    }
    config.llm_model = model;
    // Thinking is a separate posture, not part of the model-id axis. Keep it
    // off so every local GGUF is comparable, including when yaml enables it.
    config.thinking = false;
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
pub(crate) fn run_tts_worker(out: PathBuf, model: String) -> Result<()> {
    if out.exists() {
        return Err(Error::Config {
            field: "bench.worker.out".into(),
            message: format!("refusing to overwrite {}", out.display()),
        });
    }
    let mut config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    let tts_model = TtsModel::parse(&model).ok_or_else(|| Error::Config {
        field: "bench.worker.model".into(),
        message: format!("unsupported TTS model {model:?}"),
    })?;
    config.tts_model = tts_model;
    let scenarios = scenarios()?;
    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let memory_before_load = required_rss("before TTS model load")?;
    let mut tts = build_tts(&cache, &HttpFetcher, &mut progress, &cancel, &config, None)?;
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

fn run_named_worker(subcommand: &str, model: &str) -> Result<Vec<Record>> {
    let temp = std::env::temp_dir().join(format!(
        "syllabix-{subcommand}-{}-{}.jsonl",
        std::process::id(),
        unique_suffix()
    ));
    let output = Command::new(std::env::current_exe()?)
        .args([subcommand, "--out"])
        .arg(&temp)
        .args(["--model", model])
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
    stt: &mut LiveStt,
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
            backend: Some(
                stt.backend_id()
                    .unwrap_or(fingerprint.ggml_backend.as_str())
                    .to_string(),
            ),
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
            wer: None,
            ref_words: None,
            errors: None,
            noise: None,
            snr_db: None,
            threads: None,
            load_ms: None,
            first_call_ms: None,
            audio_ms: None,
            dataset: None,
            n_utt: None,
            passed: ratio >= scenario.min_word_match,
        });
    }
    for record in &mut records {
        record.memory_peak_bytes = Some(memory_peak);
    }
    Ok(records)
}

const NOISE_KINDS: [&str; 5] = ["clean", "white", "pink", "brown", "ambient"];

struct ManifestItem {
    path: PathBuf,
    reference: String,
    dataset: String,
}

fn parse_stt_manifest(text: &str, base: &Path, default_dataset: &str) -> Result<Vec<ManifestItem>> {
    let mut items = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches('\r').trim();
        if line.is_empty() {
            continue;
        }
        let Some((path, rest)) = line.split_once('\t') else {
            return Err(Error::Config {
                field: "bench.manifest".into(),
                message: format!("line {} is not wav_path<TAB>reference text", index + 1),
            });
        };
        let (reference, dataset) = match rest.split_once('\t') {
            Some((reference, dataset)) => (reference, dataset.trim()),
            None => (rest, default_dataset),
        };
        let dataset = if dataset.is_empty() {
            default_dataset
        } else {
            dataset
        };
        let wav = PathBuf::from(path.trim());
        let wav = if wav.is_absolute() {
            wav
        } else {
            base.join(wav)
        };
        items.push(ManifestItem {
            path: wav,
            reference: reference.trim().to_owned(),
            dataset: dataset.to_owned(),
        });
    }
    Ok(items)
}

fn is_unicode_apostrophe(ch: char) -> bool {
    matches!(
        ch,
        '\'' | '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{2032}' | '\u{FF07}' | '\u{02BC}'
    )
}

fn is_hyphen_joiner(ch: char) -> bool {
    matches!(
        ch,
        '-' | '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2212}'
    )
}

fn normalize_wer_text(text: &str) -> Vec<String> {
    let mut words = String::new();
    for ch in text.chars() {
        if is_hyphen_joiner(ch) {
            continue;
        }
        if is_unicode_apostrophe(ch) {
            words.push('\'');
        } else if ch.is_alphanumeric() {
            words.extend(ch.to_lowercase());
        } else {
            words.push(' ');
        }
    }
    words.split_whitespace().map(str::to_owned).collect()
}

fn word_edit_distance(hypothesis: &[String], reference: &[String]) -> usize {
    let (n, m) = (hypothesis.len(), reference.len());
    let mut prev: Vec<usize> = (0..=m).collect();
    for i in 1..=n {
        let mut cur = vec![i; m + 1];
        for j in 1..=m {
            let cost = usize::from(hypothesis[i - 1] != reference[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[m]
}

fn word_error_rate(hypothesis: &str, reference: &str) -> (f64, usize, usize) {
    let hyp = normalize_wer_text(hypothesis);
    let reference = normalize_wer_text(reference);
    let errors = word_edit_distance(&hyp, &reference);
    let ref_words = reference.len();
    let wer = if ref_words == 0 {
        0.0
    } else {
        errors as f64 / ref_words as f64
    };
    (wer, ref_words, errors)
}

fn read_manifest_wav(path: &Path) -> Result<WavPcm> {
    let file = fs::File::open(path)?;
    let wav = read_wav(file)?;
    if wav.format.sample_rate_hz != DEFAULT_SAMPLE_RATE_HZ || wav.format.channels != 1 {
        return Err(Error::Config {
            field: "bench.manifest.wav".into(),
            message: format!(
                "{} is not {DEFAULT_SAMPLE_RATE_HZ} Hz mono (got {} Hz, {} ch)",
                path.display(),
                wav.format.sample_rate_hz,
                wav.format.channels
            ),
        });
    }
    Ok(wav)
}

fn mix_manifest_pcm(
    samples: &[i16],
    noise: &str,
    snr: f32,
    rng: &mut crate::vad_lab::Rng,
) -> Result<Vec<i16>> {
    if noise == "clean" {
        return Ok(samples.to_vec());
    }
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let mixed = crate::vad_lab::noise_sequence(noise, samples.len(), rng)?;
    Ok(crate::vad_lab::mix(samples, &mixed, snr))
}

#[allow(clippy::too_many_arguments)]
fn benchmark_stt_manifest(
    stt: &mut LiveStt,
    config: &AgentConfig,
    cancel: &Cancel,
    fingerprint: &Fingerprint,
    manifest: &Path,
    noise: &str,
    snr: f32,
    seed: u64,
    warmup: usize,
    load_ms: u128,
    default_dataset: String,
    memory_before_load: usize,
    memory_after_load: usize,
) -> Result<Vec<Record>> {
    if !NOISE_KINDS.contains(&noise) {
        return Err(Error::Config {
            field: "bench.noise".into(),
            message: format!("unsupported noise {noise:?}"),
        });
    }
    let text = fs::read_to_string(manifest)?;
    let base = manifest.parent().unwrap_or_else(|| Path::new("."));
    let items = parse_stt_manifest(&text, base, &default_dataset)?;
    if items.is_empty() {
        return Err(Error::Config {
            field: "bench.manifest".into(),
            message: "manifest has no utterances".into(),
        });
    }
    let n_utt = items.len();
    let snr_db = (noise != "clean").then_some(snr);
    let load_clip = |item: &ManifestItem, index: usize| -> Result<(String, Utterance, u128)> {
        let mut rng = crate::vad_lab::Rng(seed.wrapping_add(index as u64));
        let wav = read_manifest_wav(&item.path)?;
        let samples = mix_manifest_pcm(&wav.samples, noise, snr, &mut rng)?;
        let input_audio_ms = pcm_duration_ms(samples.len());
        let utterance = pcm_to_utterance(&samples)?;
        let case = item
            .path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("utterance")
            .to_owned();
        Ok((case, utterance, input_audio_ms))
    };
    let (_, utterance, _) = load_clip(&items[0], 0)?;
    let first_started = Instant::now();
    let _ = stt.transcribe(&utterance, cancel)?;
    let first_call_ms = first_started.elapsed().as_millis();
    drop(utterance);
    for _ in 0..warmup.saturating_sub(1) {
        let (_, utterance, _) = load_clip(&items[0], 0)?;
        let _ = stt.transcribe(&utterance, cancel)?;
    }
    let threads = stt.intra_op_threads().and_then(|n| usize::try_from(n).ok());
    let mut records = Vec::with_capacity(n_utt);
    let mut memory_peak = memory_after_load.max(required_rss("after ASR first call")?);
    for (index, item) in items.iter().enumerate() {
        let (case, utterance, input_audio_ms) = load_clip(item, index)?;
        let started = Instant::now();
        let transcript = stt.transcribe(&utterance, cancel)?;
        let elapsed_ms = started.elapsed().as_millis();
        let (wer, ref_words, errors) = word_error_rate(&transcript.text, &item.reference);
        memory_peak = memory_peak.max(required_rss("during ASR benchmark")?);
        drop(utterance);
        records.push(Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fingerprint.clone(),
            component: "stt".into(),
            model: config.stt_model.as_str().into(),
            backend: Some(
                stt.backend_id()
                    .unwrap_or(fingerprint.ggml_backend.as_str())
                    .to_string(),
            ),
            case,
            elapsed_ms,
            first_output_ms: Some(elapsed_ms),
            output_units: Some(transcript.text.len()),
            prompt_tokens: None,
            generated_tokens: None,
            prompt_tokens_per_second: None,
            generated_tokens_per_second: None,
            word_match: None,
            word_match_threshold: None,
            input_audio_ms: Some(input_audio_ms),
            generated_audio_ms: None,
            real_time_factor: Some(real_time_factor(elapsed_ms, input_audio_ms)),
            transcript: Some(transcript.text),
            response: None,
            memory_before_load_bytes: Some(memory_before_load),
            memory_after_load_bytes: Some(memory_after_load),
            memory_peak_bytes: None,
            wer: Some(wer),
            ref_words: Some(ref_words),
            errors: Some(errors),
            noise: Some(noise.to_owned()),
            snr_db,
            threads,
            load_ms: Some(load_ms),
            first_call_ms: Some(first_call_ms),
            audio_ms: Some(input_audio_ms),
            dataset: Some(item.dataset.clone()),
            n_utt: Some(n_utt),
            passed: true,
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
        let mut capped = false;
        let generate_result = llm.generate(&history, &user, cancel, &mut |token| {
            first_output_ms.get_or_insert_with(|| started.elapsed().as_millis());
            generated_tokens += 1;
            output.push_str(&token.text);
            if generated_tokens >= BENCH_LLM_MAX_GENERATED_TOKENS {
                capped = true;
                // Invalidate this generation only; later cases keep running.
                let _ = cancel.cancel_generation();
            }
            Ok(())
        });
        match generate_result {
            Ok(()) => {}
            Err(Error::Cancelled) if capped && !cancel.is_shutdown() => {}
            Err(err) => return Err(err),
        }
        let elapsed_ms = started.elapsed().as_millis();
        memory_peak = memory_peak.max(required_rss("during LLM benchmark")?);
        let passed = llm_response_passes(scenario, &output);
        records.push(Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fingerprint.clone(),
            component: "llm".into(),
            model: config.llm_model.clone(),
            backend: Some(fingerprint.ggml_backend.clone()),
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
            wer: None,
            ref_words: None,
            errors: None,
            noise: None,
            snr_db: None,
            threads: None,
            load_ms: None,
            first_call_ms: None,
            audio_ms: None,
            dataset: None,
            n_utt: None,
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
    tts: &mut LiveTts,
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
            backend: tts.backend_id().map(str::to_string),
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
            wer: None,
            ref_words: None,
            errors: None,
            noise: None,
            snr_db: None,
            threads: None,
            load_ms: None,
            first_call_ms: None,
            audio_ms: None,
            dataset: None,
            n_utt: None,
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
    let ggml_backend = syllabix_native::ggml_backend_id().to_owned();
    let cpu = cpu_name();
    let ram = ram_bytes();
    let (gpu_name, gpu_vram_bytes) = match ggml_backend.as_str() {
        "vulkan" => (
            syllabix_native::vulkan_device0_name(),
            syllabix_native::vulkan_device0_vram_bytes(),
        ),
        // Apple Silicon: the chip brand is the GPU marketing name; memory is
        // unified with system RAM.
        "metal" => ((cpu != "unknown").then_some(cpu.clone()), ram),
        _ => (None, None),
    };
    Fingerprint {
        os: std::env::consts::OS.into(),
        arch: std::env::consts::ARCH.into(),
        cpu,
        cpu_cores: std::thread::available_parallelism().map_or(0, usize::from),
        ram_bytes: ram,
        binary_version: env!("CARGO_PKG_VERSION").into(),
        git_sha: option_env!("SYLLABIX_GIT_SHA").unwrap_or("unknown").into(),
        ggml_backend,
        gpu_name,
        gpu_vram_bytes,
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
            backend: Some("onnx".into()),
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
            wer: None,
            ref_words: None,
            errors: None,
            noise: None,
            snr_db: None,
            threads: None,
            load_ms: None,
            first_call_ms: None,
            audio_ms: None,
            dataset: None,
            n_utt: None,
            passed: true,
        }
    }

    #[test]
    fn writer_creates_jsonl_and_refuses_overwrite() {
        let path =
            std::env::temp_dir().join(format!("syllabix-bench-{}.jsonl", std::process::id()));
        let _ = fs::remove_file(&path);
        write_jsonl(&path, &[sample_record()]).expect("write");
        let jsonl = fs::read_to_string(&path).unwrap();
        assert_eq!(jsonl.lines().count(), 1);
        assert!(jsonl.contains("\"backend\":\"onnx\""), "{jsonl}");
        assert!(write_jsonl(&path, &[]).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn fingerprint_records_ggml_backend() {
        let fp = fingerprint();
        assert!(
            matches!(fp.ggml_backend.as_str(), "cpu" | "metal" | "vulkan"),
            "{}",
            fp.ggml_backend
        );
        if fp.ggml_backend == "cpu" {
            assert!(fp.gpu_name.is_none());
            assert!(fp.gpu_vram_bytes.is_none());
        }
        let json = serde_json::to_string(&fp).expect("fingerprint json");
        assert!(json.contains("\"ggml_backend\""), "{json}");
    }

    #[test]
    fn stt_and_llm_records_serialize_ggml_backend() {
        let fp = fingerprint();
        let stt = Record {
            schema_version: SCHEMA_VERSION,
            fingerprint: fp.clone(),
            component: "stt".into(),
            model: "whisper-small".into(),
            backend: Some(fp.ggml_backend.clone()),
            case: "jfk".into(),
            elapsed_ms: 1,
            first_output_ms: Some(1),
            output_units: Some(1),
            prompt_tokens: None,
            generated_tokens: None,
            prompt_tokens_per_second: None,
            generated_tokens_per_second: None,
            word_match: Some(1.0),
            word_match_threshold: Some(1.0),
            input_audio_ms: Some(1),
            generated_audio_ms: None,
            real_time_factor: Some(1.0),
            transcript: None,
            response: None,
            memory_before_load_bytes: None,
            memory_after_load_bytes: None,
            memory_peak_bytes: None,
            wer: None,
            ref_words: None,
            errors: None,
            noise: None,
            snr_db: None,
            threads: None,
            load_ms: None,
            first_call_ms: None,
            audio_ms: None,
            dataset: None,
            n_utt: None,
            passed: true,
        };
        let llm = Record {
            component: "llm".into(),
            model: "lfm3-1.7b".into(),
            backend: Some(fp.ggml_backend.clone()),
            case: "short".into(),
            word_match: None,
            word_match_threshold: None,
            input_audio_ms: None,
            generated_audio_ms: None,
            real_time_factor: None,
            prompt_tokens: Some(1),
            generated_tokens: Some(1),
            prompt_tokens_per_second: Some(1.0),
            generated_tokens_per_second: Some(1.0),
            ..stt.clone()
        };
        let stt_json = serde_json::to_string(&stt).expect("stt");
        let llm_json = serde_json::to_string(&llm).expect("llm");
        let expected = format!("\"backend\":\"{}\"", fp.ggml_backend);
        assert!(stt_json.contains(&expected), "{stt_json}");
        assert!(llm_json.contains(&expected), "{llm_json}");
        assert!(stt_json.contains("\"ggml_backend\""), "{stt_json}");
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

    #[test]
    fn one_axis_menu_covers_every_supported_local_model() {
        // Defaults are included; the sweep is one axis at a time, never a
        // cross-product of STT × LLM × TTS.
        assert!(SttModel::ALL.contains(&SttModel::Small));
        assert_eq!(SttModel::ALL.len(), 8);
        assert_eq!(V0_LLM_MODELS[0], "lfm3-1.7b");
        assert_eq!(V0_LLM_MODELS.len(), 8);
        assert!(TtsModel::ALL.contains(&TtsModel::PocketTts));
        assert_eq!(TtsModel::ALL.len(), 4);
        let stt_cases = scenarios()
            .unwrap()
            .iter()
            .filter(|scenario| scenario.component == "stt")
            .count();
        let llm_cases = scenarios()
            .unwrap()
            .iter()
            .filter(|scenario| scenario.component == "llm")
            .count();
        let tts_cases = scenarios()
            .unwrap()
            .iter()
            .filter(|scenario| scenario.component == "tts")
            .count();
        // Fixture corpus sizes; keep these pinned so a silent corpus shrink
        // cannot hide a missing model axis.
        assert_eq!(stt_cases, 4);
        assert_eq!(llm_cases, 5);
        assert_eq!(tts_cases, 3);
        // Derive the expected total from the live menus so adding a model
        // cannot leave a stale hard-coded multiplier behind (as with the
        // Qwen3-ASR STT opt-in that bumped ALL to 8 while this still said 7).
        assert_eq!(
            SttModel::ALL.len() * stt_cases
                + V0_LLM_MODELS.len() * llm_cases
                + TtsModel::ALL.len() * tts_cases,
            SttModel::ALL.len() * 4 + V0_LLM_MODELS.len() * 5 + TtsModel::ALL.len() * 3
        );
    }

    #[test]
    fn bench_wer_counts_insert_delete_substitute_and_empty() {
        let sub = word_error_rate("a x c", "a b c");
        assert_eq!(sub, (1.0 / 3.0, 3, 1));
        let del = word_error_rate("a c", "a b c");
        assert_eq!(del, (1.0 / 3.0, 3, 1));
        let ins = word_error_rate("a b c", "a b");
        assert_eq!(ins, (0.5, 2, 1));
        assert_eq!(word_error_rate("", ""), (0.0, 0, 0));
        assert_eq!(word_error_rate("Hello, world!", "hello world"), (0.0, 2, 0));
        assert_eq!(word_error_rate("don't", "don't"), (0.0, 1, 0));
    }

    #[test]
    fn bench_wer_normalizer_is_shared_across_models() {
        assert_eq!(
            normalize_wer_text("It's well-known, isn\u{2019}t it?"),
            ["it's", "wellknown", "isn't", "it"]
        );
        assert_eq!(
            normalize_wer_text("co\u{2011}op  A\u{2013}B   HELLO"),
            ["coop", "ab", "hello"]
        );
        assert_eq!(normalize_wer_text("  Foo,\nBAR.  "), ["foo", "bar"]);
    }

    #[test]
    fn bench_parses_manifest_tsv_and_skips_blank_lines() {
        let dir = std::env::temp_dir();
        let text = "\nrel/a.wav\thello world\n\n/abs/b.wav\tsecond\r\n";
        let items = parse_stt_manifest(text, &dir, "librispeech-test-clean").expect("parse");
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].path, dir.join("rel/a.wav"));
        assert_eq!(items[0].reference, "hello world");
        assert_eq!(items[0].dataset, "librispeech-test-clean");
        assert_eq!(items[1].path, PathBuf::from("/abs/b.wav"));
        assert_eq!(items[1].reference, "second");
        let tagged = parse_stt_manifest("a.wav\thi\tcustom-set\n", &dir, "default").expect("tag");
        assert_eq!(tagged[0].dataset, "custom-set");
        assert!(parse_stt_manifest("no-tab-here", &dir, "default").is_err());
    }
}
