//! Component-scoped contributor benchmarks. These runners call one provider
//! at a time; no conversational-pipeline timing is emitted.

use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use syllabix_core::audio::record_fixture_to_frames;
use syllabix_core::eval::{self, Fingerprint};
use syllabix_core::{
    build_tts, transcript_words, word_match_ratio, AgentConfig, Cancel, Error, GenerationId,
    HttpFetcher, LlamaLlm, ModelCache, Result, StderrProgress, Stt, SttModel, TokenChunk, Tts,
    TtsModel, TurnId, Utterance, WhisperStt, DEFAULT_SAMPLE_RATE_HZ,
};

const SCHEMA_VERSION: u32 = 3;
const WARMUP_ITERATIONS: u32 = 1;
const MEASURED_ITERATIONS: u32 = 3;
const LLM_PROMPT: &str =
    "In one short sentence, explain why reproducible benchmarks need fixed inputs.";
const TTS_FIXTURES: &[(&str, &str)] = &[
    ("short", "Hello there."),
    ("ordinary", "Syllabix runs speech models locally on your computer."),
    ("long", "A reproducible benchmark fixes its text, model, hardware fingerprint, warm-up policy, and measured iterations so results can be compared honestly."),
    ("spoken_numbers", "The total is one hundred and twenty five dollars and forty two cents."),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Component {
    All,
    Stt,
    Llm,
    Tts,
    TtsAsr,
}
impl Component {
    fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Stt => "stt",
            Self::Llm => "llm",
            Self::Tts => "tts",
            Self::TtsAsr => "tts_asr",
        }
    }
    fn includes(self, component: Self) -> bool {
        self == Self::All || self == component
    }
}

/// Parsed benchmark arguments.
#[derive(Debug, Clone)]
pub(crate) struct BenchArgs {
    pub component: Component,
    pub out: Option<PathBuf>,
    pub stt: Option<String>,
    pub llm: Option<String>,
    pub tts: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Measured,
    Unavailable,
}
#[derive(Debug, Serialize)]
struct Timing {
    warmup_iterations: u32,
    measured_iterations: u32,
    median_ms: u64,
    p95_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_pcm_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    asr_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    real_time_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_tokens_per_second: Option<f64>,
}
#[derive(Debug, Clone, Serialize)]
struct Config {
    stt: String,
    llm: String,
    tts: String,
}
#[derive(Debug, Serialize)]
struct Record {
    schema_version: u32,
    component: &'static str,
    status: Status,
    fixture_id: String,
    config: Config,
    fingerprint: Fingerprint,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing: Option<Timing>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality_pass: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality_detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    unavailable_reason: Option<String>,
}

/// Run selected direct component benchmarks and write schema-v3 JSONL.
pub(crate) fn run(args: BenchArgs) -> Result<()> {
    let config = prepare(&args)?;
    let fingerprint = Fingerprint::collect();
    let mut records = Vec::new();
    if args.component.includes(Component::Stt) {
        records.extend(run_stt_component(&config, &fingerprint));
    }
    if args.component.includes(Component::Llm) {
        records.push(run_llm_component(&config, &fingerprint));
    }
    if args.component.includes(Component::Tts) {
        records.extend(run_tts_component(&config, &fingerprint, false));
    }
    if args.component.includes(Component::TtsAsr) {
        records.extend(run_tts_component(&config, &fingerprint, true));
    }
    let path = args
        .out
        .unwrap_or_else(|| default_output_path(args.component, &fingerprint));
    write_jsonl(&path, &records)?;
    let measured = records
        .iter()
        .filter(|row| matches!(row.status, Status::Measured))
        .count();
    println!(
        "wrote {} ({} component rows; {} measured)",
        path.display(),
        records.len(),
        measured
    );
    Ok(())
}

fn prepare(args: &BenchArgs) -> Result<Config> {
    let mut config = AgentConfig::v0();
    if let Some(id) = args.stt.as_deref() {
        config.stt_model = parse_stt_id(id)?;
    }
    if let Some(id) = args.llm.as_deref() {
        if !syllabix_core::is_v0_llm_model(id) {
            return Err(Error::Config {
                field: "--llm".into(),
                message: format!("unknown id {id:?}"),
            });
        }
        config.llm_model = id.to_string();
        config.thinking = false;
    }
    if let Some(id) = args.tts.as_deref() {
        config.tts_model = parse_tts_id(id)?;
    }
    Ok(Config {
        stt: config.stt_model.as_str().into(),
        llm: config.llm_model,
        tts: config.tts_model.as_str().into(),
    })
}

fn run_stt_component(config: &Config, fingerprint: &Fingerprint) -> Vec<Record> {
    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut stt = match WhisperStt::from_cache(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        parse_stt_id(&config.stt).expect("validated stt"),
    ) {
        Ok(stt) => match stt.with_language("en") {
            Ok(stt) => stt,
            Err(err) => return vec![unavailable(Component::Stt, "all", config, fingerprint, err)],
        },
        Err(err) => return vec![unavailable(Component::Stt, "all", config, fingerprint, err)],
    };
    eval::fixtures::AUDIO_FIXTURES
        .iter()
        .enumerate()
        .map(|(index, fixture)| {
            let wav = eval::fixtures::decode_verified(fixture).expect("embedded fixture valid");
            let utterance = Utterance {
                turn: TurnId(index as u64),
                frames: record_fixture_to_frames(&wav).expect("fixture frames"),
            };
            for _ in 0..WARMUP_ITERATIONS {
                let _ = stt.transcribe(&utterance, &Cancel::new());
            }
            let mut times = Vec::new();
            let mut transcript = String::new();
            for _ in 0..MEASURED_ITERATIONS {
                let start = Instant::now();
                let out = stt
                    .transcribe(&utterance, &Cancel::new())
                    .expect("loaded STT transcribes");
                times.push(ms(start));
                transcript = out.text;
            }
            let expected = transcript_words(fixture.prompt);
            let expected: Vec<&str> = expected.iter().map(String::as_str).collect();
            let ratio = word_match_ratio(&transcript, &expected);
            measured(
                Component::Stt,
                fixture.id,
                config,
                fingerprint,
                timing(times, None, audio_ms(&utterance), None),
                ratio + f64::EPSILON >= 0.8,
                format!("word_match_ratio={ratio:.3}"),
            )
        })
        .collect()
}

fn run_llm_component(config: &Config, fingerprint: &Fingerprint) -> Record {
    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let mut llm = match LlamaLlm::from_cached_model(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        &config.llm,
        false,
    ) {
        Ok(llm) => llm,
        Err(err) => return unavailable(Component::Llm, "fixed_prompt", config, fingerprint, err),
    };
    for _ in 0..WARMUP_ITERATIONS {
        let _ = llm.benchmark_generate(LLM_PROMPT, &Cancel::new());
    }
    let mut wall = Vec::new();
    let mut perf = syllabix_native::LlamaPerf::default();
    for _ in 0..MEASURED_ITERATIONS {
        let start = Instant::now();
        perf = llm
            .benchmark_generate(LLM_PROMPT, &Cancel::new())
            .expect("loaded LLM generates");
        wall.push(ms(start));
    }
    let prompt_tps = (perf.prompt_ms > 0.0 && perf.prompt_tokens > 0)
        .then(|| perf.prompt_tokens as f64 * 1000.0 / perf.prompt_ms);
    let generation_tps = (perf.decode_ms > 0.0 && perf.generated_tokens > 0)
        .then(|| perf.generated_tokens as f64 * 1000.0 / perf.decode_ms);
    measured(
        Component::Llm,
        "fixed_prompt",
        config,
        fingerprint,
        timing(wall, None, None, Some(perf)),
        perf.generated_tokens > 0,
        format!("generated_tokens={}", perf.generated_tokens),
    )
    .with_tps(prompt_tps, generation_tps)
}

fn run_tts_component(config: &Config, fingerprint: &Fingerprint, with_asr: bool) -> Vec<Record> {
    let cancel = Cancel::new();
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let agent = AgentConfig {
        tts_model: parse_tts_id(&config.tts).expect("validated tts"),
        ..AgentConfig::v0()
    };
    let tts = match build_tts(&cache, &HttpFetcher, &mut progress, &cancel, &agent) {
        Ok(tts) => tts,
        Err(err) => {
            return vec![unavailable(
                if with_asr {
                    Component::TtsAsr
                } else {
                    Component::Tts
                },
                "all",
                config,
                fingerprint,
                err,
            )]
        }
    };
    let mut asr = if with_asr {
        match WhisperStt::from_cache(
            &cache,
            &HttpFetcher,
            &mut progress,
            &cancel,
            parse_stt_id(&config.stt).expect("validated stt"),
        ) {
            Ok(value) => match value.with_language("en") {
                Ok(value) => Some(value),
                Err(err) => {
                    return vec![unavailable(
                        Component::TtsAsr,
                        "all",
                        config,
                        fingerprint,
                        err,
                    )]
                }
            },
            Err(err) => {
                return vec![unavailable(
                    Component::TtsAsr,
                    "all",
                    config,
                    fingerprint,
                    err,
                )]
            }
        }
    } else {
        None
    };
    TTS_FIXTURES
        .iter()
        .map(|(id, text)| {
            run_one_tts(
                if with_asr {
                    Component::TtsAsr
                } else {
                    Component::Tts
                },
                id,
                text,
                tts.clone(),
                asr.as_mut(),
                config,
                fingerprint,
            )
        })
        .collect()
}

fn run_one_tts(
    component: Component,
    id: &str,
    text: &str,
    mut tts: impl Tts,
    asr: Option<&mut WhisperStt>,
    config: &Config,
    fingerprint: &Fingerprint,
) -> Record {
    for _ in 0..WARMUP_ITERATIONS {
        let _ = synthesize(&mut tts, text);
    }
    let mut times = Vec::new();
    let mut first = Vec::new();
    let mut pcm = Vec::new();
    for _ in 0..MEASURED_ITERATIONS {
        let (elapsed, first_ms, samples) =
            synthesize(&mut tts, text).expect("loaded TTS synthesizes");
        times.push(elapsed);
        first.push(first_ms);
        pcm = samples;
    }
    let audio = pcm.len() as u64 * 1000 / u64::from(DEFAULT_SAMPLE_RATE_HZ);
    if let Some(stt) = asr {
        let utterance = pcm_utterance(&pcm);
        let start = Instant::now();
        let transcript = stt
            .transcribe(&utterance, &Cancel::new())
            .expect("loaded scorer transcribes");
        let scorer_ms = ms(start);
        let expected = transcript_words(text);
        let expected: Vec<&str> = expected.iter().map(String::as_str).collect();
        let ratio = word_match_ratio(&transcript.text, &expected);
        let mut record = measured(
            component,
            id,
            config,
            fingerprint,
            timing(times, Some(median(first)), Some(audio), None),
            ratio + f64::EPSILON >= 0.8,
            format!("tts_asr_word_match_ratio={ratio:.3}"),
        );
        record.timing.as_mut().expect("measured timing").asr_ms = Some(scorer_ms);
        return record;
    }
    measured(
        component,
        id,
        config,
        fingerprint,
        timing(times, Some(median(first)), Some(audio), None),
        !pcm.is_empty(),
        "non_empty_pcm".into(),
    )
}

fn synthesize(tts: &mut impl Tts, text: &str) -> Result<(u64, u64, Vec<i16>)> {
    let start = Instant::now();
    let mut first = None;
    let mut samples = Vec::new();
    let token = TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index: 0,
        text: text.into(),
        is_last: true,
    };
    tts.synthesize_chunk_into(&token, &Cancel::new(), &mut |audio| {
        first.get_or_insert(ms(start));
        samples.extend(audio.samples);
        Ok(())
    })?;
    Ok((ms(start), first.unwrap_or_else(|| ms(start)), samples))
}
fn pcm_utterance(samples: &[i16]) -> Utterance {
    let mut splitter = eval::FrameSplitter::new();
    let mut frames = splitter.push(samples).expect("valid PCM");
    frames.extend(splitter.flush().expect("flush PCM"));
    Utterance {
        turn: TurnId(0),
        frames,
    }
}
fn measured(
    component: Component,
    fixture_id: impl Into<String>,
    config: &Config,
    fingerprint: &Fingerprint,
    timing: Timing,
    quality_pass: bool,
    quality_detail: String,
) -> Record {
    Record {
        schema_version: SCHEMA_VERSION,
        component: component.name(),
        status: Status::Measured,
        fixture_id: fixture_id.into(),
        config: config.clone(),
        fingerprint: fingerprint.clone(),
        timing: Some(timing),
        quality_pass: Some(quality_pass),
        quality_detail: Some(quality_detail),
        unavailable_reason: None,
    }
}
fn unavailable(
    component: Component,
    fixture: &str,
    config: &Config,
    fingerprint: &Fingerprint,
    err: impl std::fmt::Display,
) -> Record {
    Record {
        schema_version: SCHEMA_VERSION,
        component: component.name(),
        status: Status::Unavailable,
        fixture_id: fixture.into(),
        config: config.clone(),
        fingerprint: fingerprint.clone(),
        timing: None,
        quality_pass: None,
        quality_detail: None,
        unavailable_reason: Some(err.to_string()),
    }
}
impl Record {
    fn with_tps(mut self, prompt: Option<f64>, generation: Option<f64>) -> Self {
        if let Some(timing) = &mut self.timing {
            timing.prompt_tokens_per_second = prompt;
            timing.generation_tokens_per_second = generation;
        }
        self
    }
}
fn timing(
    mut measurements: Vec<u64>,
    first_pcm_ms: Option<u64>,
    audio_ms: Option<u64>,
    perf: Option<syllabix_native::LlamaPerf>,
) -> Timing {
    measurements.sort_unstable();
    let median_ms = median(measurements.clone());
    let p95_ms = measurements[(measurements.len() * 95).div_ceil(100).saturating_sub(1)];
    Timing {
        warmup_iterations: WARMUP_ITERATIONS,
        measured_iterations: MEASURED_ITERATIONS,
        median_ms,
        p95_ms,
        first_pcm_ms,
        audio_ms,
        asr_ms: None,
        real_time_factor: audio_ms
            .filter(|audio| *audio > 0)
            .map(|audio| median_ms as f64 / audio as f64),
        prompt_tokens: perf.map(|value| value.prompt_tokens),
        generated_tokens: perf.map(|value| value.generated_tokens),
        prompt_tokens_per_second: None,
        generation_tokens_per_second: None,
    }
}
fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}
fn ms(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}
fn audio_ms(utterance: &Utterance) -> Option<u64> {
    Some(utterance.pcm().len() as u64 * 1000 / u64::from(DEFAULT_SAMPLE_RATE_HZ))
}
fn write_jsonl(path: &std::path::Path, records: &[Record]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    use std::io::Write;
    for record in records {
        serde_json::to_writer(&mut out, record)
            .map_err(|err| Error::Io(std::io::Error::other(err)))?;
        out.write_all(b"\\n")?;
    }
    out.flush()?;
    Ok(())
}
fn default_output_path(component: Component, fingerprint: &Fingerprint) -> PathBuf {
    PathBuf::from("docs/eval/runs").join(format!(
        "{}-{}-{}-{}.jsonl",
        component.name(),
        fingerprint.os,
        fingerprint.arch,
        fingerprint.git_sha.chars().take(12).collect::<String>()
    ))
}
fn parse_stt_id(id: &str) -> Result<SttModel> {
    SttModel::parse(id).ok_or_else(|| Error::Config {
        field: "--stt".into(),
        message: format!("unknown id {id:?}"),
    })
}
fn parse_tts_id(id: &str) -> Result<TtsModel> {
    TtsModel::parse(id).ok_or_else(|| Error::Config {
        field: "--tts".into(),
        message: format!("unknown id {id:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn component_selection_is_independent() {
        assert!(Component::All.includes(Component::Llm));
        assert!(Component::Tts.includes(Component::Tts));
        assert!(!Component::Stt.includes(Component::Tts));
    }
    #[test]
    fn timing_uses_median_and_p95() {
        let timing = timing(vec![1, 2, 9], None, None, None);
        assert_eq!(timing.median_ms, 2);
        assert_eq!(timing.p95_ms, 9);
    }
}
