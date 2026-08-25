//! Issue #45 internal double-check: the ledger harness over the real launch
//! stack. This is NOT the contribute path (`syllabix bench` is); it keeps the
//! fixture corpus and its gates honest whenever the native bar runs.
//!
//! Runs only when the launch stack is selected (default `cargo test`, or an
//! explicit `SYLLABIX_NATIVE_MODELS` containing small + llama-3.2-1b + kokoro).

use syllabix_core::eval::{run_bench, BenchOptions, BenchProviders};
use syllabix_core::{
    Cancel, HttpFetcher, KokoroTts, LlamaLlm, ModelCache, Result, SileroVad, StderrProgress,
    VadSettings, WhisperStt,
};

use crate::{native, skip_unless_launch_stack};

struct NativeProviders;

impl BenchProviders for NativeProviders {
    type Vad = SileroVad;
    type Stt = WhisperStt;
    type Llm = LlamaLlm;
    type Tts = KokoroTts;

    /// Fresh Silero per scenario: the GRU hangover stalls later turns on a
    /// shared instance (same rule as the six-turn fixture).
    fn fresh_vad(&mut self) -> Result<SileroVad> {
        SileroVad::from_cache(
            &ModelCache::v0(),
            &HttpFetcher,
            &mut StderrProgress::new(),
            &Cancel::new(),
        )
        .map(|vad| vad.with_settings(VadSettings::v0()))
    }

    fn stt(&mut self) -> Result<WhisperStt> {
        Ok(native().stt().clone())
    }

    fn llm(&mut self) -> Result<LlamaLlm> {
        Ok(native().llm().clone())
    }

    fn tts(&mut self) -> KokoroTts {
        native().tts().clone()
    }
}

#[test]
fn eval_scenarios_run_and_score() {
    skip_unless_launch_stack!();

    // Warm every engine once before timing anything.
    {
        let mut models = native();
        let _ = models.stt();
        let _ = models.llm();
        let _ = models.tts();
    }

    let mut providers = NativeProviders;
    let records = run_bench(
        &mut providers,
        syllabix_core::eval::builtin_scenarios(),
        &BenchOptions::default(),
        &mut |line| eprintln!("{line}"),
        &Cancel::new(),
    )
    .expect("native eval run");

    assert_eq!(
        records.len(),
        syllabix_core::eval::builtin_scenarios().len()
    );
    let mut failures = Vec::new();
    for record in &records {
        for turn in &record.turns {
            eprintln!(
                "{}[{}] STT {} TTFT {} TTFB {} total {}",
                record.scenario_id,
                turn.index,
                turn.stt_ms,
                turn.ttft_ms,
                turn.ttfb_ms,
                turn.total_ms
            );
            eprintln!("  user: {}", turn.stt_text);
            eprintln!("  agent: {}", turn.reply);
            eprintln!("  turn gates: {:?}", turn.gates);
        }
        eprintln!(
            "  scenario gates: {:?} tts_asr={:?}",
            record.gates, record.gates.tts_asr_ratio
        );
        if !record.passed {
            failures.push(record.scenario_id.clone());
        }
    }
    assert!(failures.is_empty(), "scenarios failed gates: {failures:?}");
}
