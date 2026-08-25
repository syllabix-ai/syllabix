//! `syllabix bench`: the contributor performance harness on the shipped
//! binary (issue #45).
//!
//! Loads the selected on-device stack through the same first-run cache as
//! `run`, then drives the embedded frozen-WAV corpus (Silero → whisper.cpp →
//! llama.cpp → TTS) and writes one JSONL ledger row
//! per scenario. One-axis model swaps (`--stt` / `--llm` / `--tts`) exist for
//! named profiles; there is no online LLM here by design.

use std::path::PathBuf;

use syllabix_core::eval::{self, write_jsonl, BenchOptions, BenchStatus, ConfigIds};
#[cfg(not(coverage))]
use syllabix_core::eval::{run_bench, BenchProviders, BenchRecord};
use syllabix_core::is_v0_llm_model;
#[cfg(not(coverage))]
use syllabix_core::{
    build_tts, HttpFetcher, LiveTts, LlamaLlm, ModelCache, SileroVad, StderrProgress, VadSettings,
    WhisperStt,
};
use syllabix_core::{AgentConfig, Cancel, Error, Result, SttModel, TtsModel};

/// Parsed `syllabix bench` arguments.
#[derive(Debug, Clone, Default)]
pub(crate) struct BenchArgs {
    /// Output JSONL path. Defaults to `docs/eval/runs/<fingerprint>.jsonl`.
    pub out: Option<PathBuf>,
    /// STT yaml id override (one-axis swap).
    pub stt: Option<String>,
    /// LLM yaml id override (one-axis swap).
    pub llm: Option<String>,
    /// TTS yaml id override (one-axis swap).
    pub tts: Option<String>,
}

/// Real launch-stack providers shared across scenarios; VAD comes fresh per
/// scenario from the cache (GRU hangover), the other engines clone their
/// Arc-backed internals.
#[cfg(not(coverage))]
struct RealProviders {
    cache: ModelCache,
    stt: WhisperStt,
    llm: LlamaLlm,
    tts: LiveTts,
}

#[cfg(not(coverage))]
impl BenchProviders for RealProviders {
    type Vad = SileroVad;
    type Stt = WhisperStt;
    type Llm = LlamaLlm;
    type Tts = LiveTts;

    /// Fresh VAD per scenario from the cache (GRU hangover).
    fn fresh_vad(&mut self) -> Result<SileroVad> {
        SileroVad::from_cache(
            &self.cache,
            &HttpFetcher,
            &mut StderrProgress::new(),
            &Cancel::new(),
        )
        .map(|vad| vad.with_settings(VadSettings::v0()))
    }

    fn stt(&mut self) -> Result<WhisperStt> {
        Ok(self.stt.clone())
    }

    fn llm(&mut self) -> Result<LlamaLlm> {
        Ok(self.llm.clone())
    }

    fn tts(&mut self) -> LiveTts {
        self.tts.clone()
    }
}

/// Run the harness end to end: resolve ids → load weights → run scenarios →
/// write JSONL. Prints progress to stderr; the final line names the output.
pub(crate) fn run(args: BenchArgs) -> Result<()> {
    #[cfg(not(coverage))]
    let (config, options) = prepare(&args)?;
    #[cfg(coverage)]
    let (_config, options) = prepare(&args)?;

    #[cfg(coverage)]
    let records = {
        // The coverage build intentionally swaps only the native stages. It
        // still loads the embedded scenarios, runs the complete orchestration
        // and gate pipeline, and writes the same JSONL schema as a release.
        let scenarios = eval::builtin_scenarios();
        eval::run_bench_with_fake_providers(
            scenarios,
            &options,
            &mut |line| eprintln!("bench: {line}"),
            &Cancel::new(),
        )?
    };

    #[cfg(not(coverage))]
    let records = {
        eprintln!(
            "bench: loading {} (first run fetches ~1.6 GB into the model cache)",
            options.profile
        );
        let cancel = Cancel::new();
        match RealProviders::load(&config, &cancel) {
            Ok(mut providers) => run_bench(
                &mut providers,
                eval::builtin_scenarios(),
                &options,
                &mut |line| eprintln!("bench: {line}"),
                &cancel,
            )?,
            Err(err) => {
                let reason = err.to_string();
                eprintln!("bench: N/A {}: {reason}", options.profile);
                eval::builtin_scenarios()
                    .iter()
                    .map(|scenario| BenchRecord::unavailable(&options, scenario, &reason))
                    .collect()
            }
        }
    };

    finish(args.out, &options, records)
}

fn prepare(args: &BenchArgs) -> Result<(AgentConfig, BenchOptions)> {
    let override_count = [args.stt.as_ref(), args.llm.as_ref(), args.tts.as_ref()]
        .into_iter()
        .flatten()
        .count();
    if override_count > 1 {
        return Err(Error::Config {
            field: "bench model axis".into(),
            message: "choose only one of --stt, --llm, or --tts".into(),
        });
    }

    let mut config = AgentConfig::v0();
    if let Some(id) = args.stt.as_deref() {
        config.stt_model = parse_stt_id(id)?;
    }
    if let Some(id) = args.llm.as_deref() {
        if !is_v0_llm_model(id) {
            return Err(Error::Config {
                field: "--llm".into(),
                message: format!(
                    "unknown id {id:?} (allowed: llama-3.2-1b, qwen3.5-0.8b, qwen3.5-2b)"
                ),
            });
        }
        // The ledger measures the launch posture: thinking stays off.
        config.llm_model = id.to_string();
        config.thinking = false;
    }
    if let Some(id) = args.tts.as_deref() {
        config.tts_model = parse_tts_id(id)?;
    }

    let options = BenchOptions {
        mode: eval::MODE_FIXTURE,
        profile: profile_name(&config),
        config: config_ids(&config),
    };

    Ok((config, options))
}

fn finish(
    out: Option<PathBuf>,
    options: &BenchOptions,
    records: Vec<syllabix_core::eval::BenchRecord>,
) -> Result<()> {
    let path = out.unwrap_or_else(|| eval::default_output_path(&options.config));
    write_jsonl(&path, &records)?;

    let measured = records
        .iter()
        .filter(|record| record.status == BenchStatus::Measured)
        .count();
    let unavailable = records.len() - measured;
    let passed = records.iter().filter(|record| record.passed).count();
    println!("wrote {} ({} rows)", path.display(), records.len());
    println!("gates: {passed}/{measured} measured scenarios passed all gates; {unavailable} N/A");
    for record in records
        .iter()
        .filter(|record| record.status == BenchStatus::Measured && !record.passed)
    {
        println!(
            "  gate miss: {} (turn_count {}, keywords {}, not_contains {}, tts_asr {:?})",
            record.scenario_id,
            record.gates.turn_count_pass,
            record.gates.keywords_pass,
            record.gates.not_contains_pass,
            record.gates.tts_asr_ratio
        );
    }
    Ok(())
}

#[cfg(not(coverage))]
impl RealProviders {
    /// Load STT + LLM + TTS once through the shared v0 manifest cache. Only
    /// the selected ids are fetched; the rest of each menu stays untouched.
    fn load(config: &AgentConfig, cancel: &Cancel) -> Result<Self> {
        let cache = ModelCache::v0();
        let mut progress = StderrProgress::new();
        let llm = LlamaLlm::from_cached_model(
            &cache,
            &HttpFetcher,
            &mut progress,
            cancel,
            &config.llm_model,
            config.thinking,
        )?
        .with_system_prompt(config.system_prompt.clone());
        let stt = WhisperStt::from_cache(
            &cache,
            &HttpFetcher,
            &mut progress,
            cancel,
            config.stt_model,
        )?
        .with_language(&config.language)?;
        let tts = build_tts(&cache, &HttpFetcher, &mut progress, cancel, config)?;
        Ok(Self {
            cache,
            stt,
            llm,
            tts,
        })
    }
}

fn parse_stt_id(id: &str) -> Result<SttModel> {
    SttModel::parse(id).ok_or_else(|| Error::Config {
        field: "--stt".into(),
        message: format!(
            "unknown id {id:?} (allowed: {})",
            menu_list(SttModel::ALL.iter().map(|model| model.as_str()))
        ),
    })
}

fn parse_tts_id(id: &str) -> Result<TtsModel> {
    TtsModel::parse(id).ok_or_else(|| Error::Config {
        field: "--tts".into(),
        message: format!(
            "unknown id {id:?} (allowed: {})",
            menu_list(TtsModel::ALL.iter().map(|model| model.as_str()))
        ),
    })
}

fn menu_list<'a>(items: impl Iterator<Item = &'a str>) -> String {
    items.collect::<Vec<_>>().join(", ")
}

fn config_ids(config: &AgentConfig) -> ConfigIds {
    ConfigIds {
        stt: config.stt_model.as_str().to_string(),
        llm: config.llm_model.clone(),
        tts: config.tts_model.as_str().to_string(),
    }
}

fn profile_name(config: &AgentConfig) -> String {
    config_ids(config).profile_name()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programmatic_args_reject_multiple_model_axes() {
        let err = prepare(&BenchArgs {
            stt: Some("medium".into()),
            llm: Some("qwen3.5-0.8b".into()),
            ..BenchArgs::default()
        })
        .expect_err("only one axis is valid");
        assert!(err.to_string().contains("only one"), "{err}");
    }

    #[test]
    fn one_axis_profile_uses_the_selected_id() {
        let (_, options) = prepare(&BenchArgs {
            tts: Some("qwen3-0.6".into()),
            ..BenchArgs::default()
        })
        .expect("valid single-axis config");
        assert_eq!(options.profile, "tts-qwen3-0.6");
    }
}
