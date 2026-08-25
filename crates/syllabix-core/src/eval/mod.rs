//! The performance harness behind `syllabix bench` (issue #45).
//!
//! It is the issue-#44 fixture loop **inside the process we already ship**:
//! JSONL scenarios → frozen WAV capture → [`run_loop_captured`] (Silero →
//! whisper.cpp → llama.cpp → TTS) → deterministic gates + clocks. One
//! JSONL row per scenario carries the machine fingerprint, per-turn STT /
//! TTFT / TTFB / total timings, and gate verdicts.
//!
//! These clocks are compute-bound by construction (`FixtureCapture` feeds
//! frames as fast as the pipeline drains them; no device pacing), so rows are
//! comparable across runs on the same machine. Live G4 measurement
//! (silence-end → speaker) stays a separate Mac capture that reuses the same
//! row schema with `mode: "live"`.
//!
//! The same runner drives real weights and coverage-bar fakes through the
//! [`BenchProviders`] factory trait, so every working-logic behavior here is
//! unit-tested without models.

pub mod fingerprint;
pub mod fixtures;
pub mod gates;
pub mod report;
pub mod scenario;

use std::sync::{Arc, Mutex};

use crate::cancel::Cancel;
use crate::defaults::BuiltinDefaults;
use crate::error::{Error, Result};
use crate::pipeline::{run_loop_captured, LoopConfig, LoopMode, PipelineStages};
use crate::providers::{AudioSink, Llm, Stt, Tts, Vad};
use crate::speech_text::speak_text_for_tts;
use crate::stt::{transcript_words, word_match_ratio};
use crate::types::{
    AudioFrame, CompletedTurn, SynthesizedAudio, TurnId, Utterance, DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};

pub use crate::audio::{FixtureCapture, FrameSplitter};
pub use fingerprint::Fingerprint;
pub use gates::{ScenarioGateVerdicts, TurnGateVerdicts};
pub use report::{
    default_output_path, write_jsonl, BenchRecord, BenchStatus, ConfigIds, TurnRecord,
    MODE_FIXTURE, MODE_LIVE, SCHEMA_VERSION,
};
pub use scenario::{builtin_scenarios, parse_scenarios, Scenario};

/// Silence pad between synthesized turns: past the hangover so VAD closes
/// each utterance cleanly (same shape as the six-turn native fixture).
const INTER_TURN_SILENCE_FRAMES: usize = crate::vad::END_SILENCE_FRAMES + 2;

/// Silence before the first prompt (~256 ms): mirrors a real capture session
/// and gives Silero's RNN quiet windows to settle before the first onset.
const LEADING_SILENCE_FRAMES: usize = 8;

/// Stage factory so one runner drives both real weights and test fakes.
///
/// Implementations hand out owned stages per scenario: VAD must be fresh
/// (Silero GRU hangover stalls later turns on a shared instance), and the
/// other engines clone their Arc-backed internals cheaply.
pub trait BenchProviders {
    /// VAD implementation handed to the pipeline fresh per scenario.
    type Vad: Vad + 'static;
    /// STT implementation (transcription inside the loop + TTS→ASR scorer).
    type Stt: Stt + 'static;
    /// LLM implementation.
    type Llm: Llm + 'static;
    /// TTS implementation used by the pipeline stage.
    type Tts: Tts + 'static;

    /// A fresh VAD for one scenario.
    fn fresh_vad(&mut self) -> Result<Self::Vad>;
    /// An STT handle. Called once per scenario plus once per TTS→ASR score.
    fn stt(&mut self) -> Result<Self::Stt>;
    /// An LLM handle, consumed by one scenario's pipeline run.
    fn llm(&mut self) -> Result<Self::Llm>;
    /// A TTS handle for the pipeline stage.
    fn tts(&mut self) -> Self::Tts;
}

/// Run the benchmark orchestration with deterministic in-memory providers.
///
/// Coverage builds use this instead of opening devices, downloading weights,
/// or loading native models. The scenario decode, pipeline orchestration,
/// scoring, cancellation, and JSONL shaping remain the production code.
#[cfg(coverage)]
pub fn run_bench_with_fake_providers(
    scenarios: &[Scenario],
    options: &BenchOptions,
    on_event: &mut dyn FnMut(&str),
    cancel: &Cancel,
) -> Result<Vec<BenchRecord>> {
    let mut providers = CoverageFakeProviders;
    run_bench(&mut providers, scenarios, options, on_event, cancel)
}

#[cfg(coverage)]
struct CoverageFakeProviders;

#[cfg(coverage)]
impl BenchProviders for CoverageFakeProviders {
    type Vad = crate::fake::FakeVad;
    type Stt = crate::fake::FakeStt;
    type Llm = crate::fake::FakeLlm;
    type Tts = crate::fake::FakeTts;

    fn fresh_vad(&mut self) -> Result<Self::Vad> {
        Ok(crate::fake::FakeVad::new())
    }

    fn stt(&mut self) -> Result<Self::Stt> {
        Ok(crate::fake::FakeStt)
    }

    fn llm(&mut self) -> Result<Self::Llm> {
        Ok(crate::fake::FakeLlm::new())
    }

    fn tts(&mut self) -> Self::Tts {
        crate::fake::FakeTts
    }
}

/// Options stamped into every row of one bench invocation.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    /// Row mode ([`MODE_FIXTURE`]).
    pub mode: &'static str,
    /// Named profile id (`default`, `stt-medium`, …).
    pub profile: String,
    /// Measured model ids.
    pub config: ConfigIds,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            mode: MODE_FIXTURE,
            profile: ConfigIds::v0().profile_name(),
            config: ConfigIds::v0(),
        }
    }
}

/// Run every scenario and return one ledger row per scenario.
///
/// Fails fast on harness errors (weight load, provider failure, queue-bound
/// violation); gate misses are *recorded*, not raised — the ledger exists to
/// show them. A shutdown `cancel` stops *between* scenarios and returns the
/// rows collected so far; each scenario runs on its own fresh [`Cancel`] so
/// the loop's internal stop-after-N-turns shutdown never poisons later rows.
pub fn run_bench<P: BenchProviders>(
    providers: &mut P,
    scenarios: &[Scenario],
    options: &BenchOptions,
    on_event: &mut dyn FnMut(&str),
    cancel: &Cancel,
) -> Result<Vec<BenchRecord>> {
    let fingerprint = Fingerprint::collect();
    let total = scenarios.len();
    let mut records = Vec::with_capacity(total);
    for (index, scenario) in scenarios.iter().enumerate() {
        if cancel.is_shutdown() {
            break;
        }
        on_event(&format!(
            "[{}/{}] {} ({})",
            index + 1,
            total,
            scenario.id,
            scenario.category
        ));
        let record = run_scenario(providers, scenario, options, &fingerprint)?;
        on_event(&format!(
            "[{}/{}] {} → {}",
            index + 1,
            total,
            scenario.id,
            if record.passed { "pass" } else { "GATE MISS" }
        ));
        records.push(record);
    }
    Ok(records)
}

/// [`AudioSink`] that survives being moved into the pipeline: the harness
/// keeps a handle and reads the agent PCM back after the run joins.
#[derive(Debug, Default, Clone)]
struct SharedCollectingSink(Arc<Mutex<crate::CollectingSink>>);

impl SharedCollectingSink {
    fn collected_samples(&self) -> Vec<i16> {
        let sink = self.0.lock().expect("collecting sink");
        sink.chunks
            .iter()
            .flat_map(|chunk| chunk.samples.iter().copied())
            .collect()
    }

    fn collected_chunk_count(&self) -> usize {
        self.0.lock().expect("collecting sink").chunks.len()
    }
}

impl AudioSink for SharedCollectingSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        self.0.lock().expect("collecting sink").play(audio, cancel)
    }

    fn interrupt(&mut self) {
        self.0.lock().expect("collecting sink").interrupt();
    }
}

fn run_scenario<P: BenchProviders>(
    providers: &mut P,
    scenario: &Scenario,
    options: &BenchOptions,
    fingerprint: &Fingerprint,
) -> Result<BenchRecord> {
    // Fresh cancel per scenario: StopAfterTurns shuts the loop's clone down
    // when the expected turns complete, and that must not leak into the next
    // scenario or the TTS→ASR scorer.
    let cancel = Cancel::new();

    // The same hash-pinned WAV bytes are fed to every profile. A `--tts`
    // override changes only the agent output, never the microphone input.
    let frames = scenario_frames(scenario)?;
    let capture = FixtureCapture::from_frames(frames);

    let vad = providers.fresh_vad()?;
    let stt = providers.stt()?;
    let llm = providers.llm()?;
    let tts = providers.tts();

    let mode = if scenario.silence_only {
        LoopMode::UntilInputEnds
    } else {
        LoopMode::StopAfterTurns(scenario.expected_turns())
    };
    let config = LoopConfig {
        defaults: BuiltinDefaults::v0(),
        mode,
        events: None,
        turn_debug: None,
        barge_in: false,
    };

    let sink = SharedCollectingSink::default();
    let sink_handle = sink.clone();
    let report = run_loop_captured(
        config,
        PipelineStages {
            vad,
            stt,
            llm,
            tts,
            sink,
        },
        capture,
        cancel.clone(),
    )?;

    if report.tasks_still_running != 0 || report.tasks_exited == 0 {
        return Err(Error::Provider {
            provider: "eval",
            message: format!(
                "pipeline did not shut down cleanly (exited {}, still running {})",
                report.tasks_exited, report.tasks_still_running
            ),
        });
    }
    if !report.queues.within_capacity() {
        return Err(Error::Provider {
            provider: "eval",
            message: format!("queue occupancy exceeded a bound: {:?}", report.queues),
        });
    }

    // Per-turn scoring: pair completed turns with corpus prompts in order.
    let mut turn_records = Vec::with_capacity(report.turns.len());
    let mut raw_replies: Vec<String> = Vec::new();
    for (index, turn) in report.turns.iter().enumerate() {
        let Some(prompt) = scenario.turns.get(index) else {
            break;
        };
        let speak_text = speak_text_for_tts(&turn.assistant_text);
        let turn_gates = gates::turn_verdicts(
            prompt,
            &turn.assistant_text,
            &speak_text,
            &turn.user_text,
            scenario.stt_min_match(),
        );
        raw_replies.push(turn.assistant_text.clone());
        turn_records.push(build_turn_record(
            index, prompt, turn, speak_text, turn_gates,
        ));
    }

    // Scenario-level scoring, including the TTS→ASR round-trip over the
    // audio that actually flowed through the loop.
    let replies: Vec<&str> = raw_replies.iter().map(String::as_str).collect();
    let spoken: String = report
        .turns
        .iter()
        .map(|turn| speak_text_for_tts(&turn.assistant_text))
        .collect::<Vec<_>>()
        .join(" ");
    let agent_pcm = sink_handle.collected_samples();
    debug_assert!(sink_handle.collected_chunk_count() > 0 || agent_pcm.is_empty());
    // Fresh cancel for the scorer: the scenario loop may have shut its own
    // clone down after completing the expected turns.
    let asr_ratio = tts_asr_ratio(providers, &agent_pcm, &spoken)?;

    let scenario_gates = gates::scenario_verdicts(
        scenario.expected_turns(),
        report.turns.len(),
        &replies,
        &scenario.expect,
        asr_ratio,
    );

    let passed = turn_records.iter().all(|turn| turn.gates.passed()) && scenario_gates.passed();

    Ok(BenchRecord {
        schema_version: SCHEMA_VERSION,
        mode: options.mode,
        status: BenchStatus::Measured,
        profile: options.profile.clone(),
        config: options.config.clone(),
        fingerprint: fingerprint.clone(),
        scenario_id: scenario.id.clone(),
        category: scenario.category.clone(),
        turns: turn_records,
        gates: scenario_gates,
        skipped_turns: report.skipped_turns,
        passed,
        unavailable_reason: None,
    })
}

fn build_turn_record(
    index: usize,
    prompt: &str,
    turn: &CompletedTurn,
    speak_text: String,
    gates: TurnGateVerdicts,
) -> TurnRecord {
    TurnRecord {
        index,
        prompt: prompt.to_string(),
        stt_text: turn.user_text.clone(),
        reply: turn.assistant_text.clone(),
        speak_text,
        stt_ms: turn.timings.stt.as_millis() as u64,
        ttft_ms: turn.timings.ttft.as_millis() as u64,
        ttfb_ms: turn.timings.ttfb.as_millis() as u64,
        total_ms: turn.timings.total.as_millis() as u64,
        gates,
    }
}

/// Transcribe the agent's own speech back through STT and score it against
/// what was actually spoken ([`TTS_ASR_MIN_WORD_MATCH`] floor).
///
/// `None` when there is nothing to round-trip (silence-only scenarios).
fn tts_asr_ratio<P: BenchProviders>(
    providers: &mut P,
    agent_pcm: &[i16],
    spoken: &str,
) -> Result<Option<f64>> {
    if agent_pcm.is_empty() || spoken.trim().is_empty() {
        return Ok(None);
    }
    let mut splitter = FrameSplitter::new();
    let mut frames = splitter.push(agent_pcm)?;
    frames.extend(splitter.flush()?);
    let utterance = Utterance {
        turn: TurnId(0),
        frames,
    };
    let mut scorer = providers.stt()?;
    let transcript = scorer.transcribe(&utterance, &Cancel::new())?;
    let expected = transcript_words(spoken);
    let refs: Vec<&str> = expected.iter().map(String::as_str).collect();
    Ok(Some(word_match_ratio(&transcript.text, &refs)))
}

fn silence_frame(seq: u64) -> Result<AudioFrame> {
    AudioFrame::new(
        seq,
        DEFAULT_SAMPLE_RATE_HZ,
        DEFAULT_CHANNELS,
        vec![0; FRAME_SAMPLES],
    )
}

/// Decode one frozen scenario recording into v0 frames. Each fixture is
/// canonical 16 kHz mono PCM16; the only added samples are the normal
/// inter-turn silence required to close VAD utterances.
pub fn scenario_frames(scenario: &Scenario) -> Result<Vec<AudioFrame>> {
    let mut out: Vec<AudioFrame> = Vec::new();
    if scenario.silence_only {
        let frame_count = (scenario.silence_seconds * f64::from(DEFAULT_SAMPLE_RATE_HZ)
            / FRAME_SAMPLES as f64)
            .ceil() as usize;
        for index in 0..frame_count {
            out.push(silence_frame(index as u64)?);
        }
        return Ok(out);
    }

    // Leading silence: a real mic never starts mid-word, and Silero's RNN
    // needs a few quiet windows before it scores speech reliably.
    for index in 0..LEADING_SILENCE_FRAMES {
        out.push(silence_frame(index as u64)?);
    }
    for (turn_index, prompt) in scenario.turns.iter().enumerate() {
        let fixture_id = format!("{}_turn_{}", scenario.id, turn_index + 1);
        let fixture = fixtures::audio_fixture(&fixture_id)?;
        if fixture.prompt != prompt {
            return Err(Error::Config {
                field: "eval.scenarios.turns".into(),
                message: format!(
                    "scenario {:?} turn {} does not match fixture {:?}",
                    scenario.id,
                    turn_index + 1,
                    fixture_id
                ),
            });
        }
        let wav = fixtures::decode_verified(fixture)?;
        if wav.format.sample_rate_hz != DEFAULT_SAMPLE_RATE_HZ
            || wav.format.channels != DEFAULT_CHANNELS
        {
            return Err(Error::InvalidAudio {
                message: format!(
                    "audio fixture {:?} must be {} Hz mono",
                    fixture_id, DEFAULT_SAMPLE_RATE_HZ
                ),
            });
        }
        let mut splitter = FrameSplitter::new();
        out.extend(splitter.push(&wav.samples)?);
        out.extend(splitter.flush()?);
        for index in 0..INTER_TURN_SILENCE_FRAMES {
            out.push(silence_frame(index as u64)?);
        }
    }
    // Renumber strictly increasing across the speech + silence mixture.
    for (seq, frame) in out.iter_mut().enumerate() {
        frame.seq = seq as u64;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::scenario::BUILTIN_SCENARIOS_JSONL;
    use crate::fake::{FakeLlm, FakeStt, FakeTts, FakeVad};

    /// Deterministic fakes: energy VAD segments on the silence pads, echo LLM
    /// answers non-blank, FakeStt transcripts are fixed strings.
    struct FakeProviders;

    impl BenchProviders for FakeProviders {
        type Vad = FakeVad;
        type Stt = FakeStt;
        type Llm = FakeLlm;
        type Tts = FakeTts;

        fn fresh_vad(&mut self) -> Result<FakeVad> {
            Ok(FakeVad::new())
        }

        fn stt(&mut self) -> Result<FakeStt> {
            Ok(FakeStt)
        }

        fn llm(&mut self) -> Result<FakeLlm> {
            Ok(FakeLlm::new())
        }

        fn tts(&mut self) -> FakeTts {
            FakeTts
        }
    }

    #[test]
    fn scenario_frames_segment_per_turn_and_silence_only_is_silent() {
        let scenarios = parse_scenarios(BUILTIN_SCENARIOS_JSONL).expect("builtin corpus");
        let scenario = scenarios.iter().find(|s| s.id == "greeting_001").unwrap();
        let frames = scenario_frames(scenario).expect("frames");
        // Every frame valid, strictly increasing seqs, silence pads present.
        assert!(frames.len() > 4);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame.seq as usize, index);
            frame.validate().expect("valid frame");
        }
        let pad = INTER_TURN_SILENCE_FRAMES;
        // Tail is exactly one silence pad; the middle boundary too.
        for frame in &frames[frames.len() - pad..] {
            assert!(frame.samples.iter().all(|sample| *sample == 0));
        }

        let quiet = scenario_frames(scenarios.last().unwrap()).expect("silence frames");
        let expected =
            (2.0 * f64::from(DEFAULT_SAMPLE_RATE_HZ) / FRAME_SAMPLES as f64).ceil() as usize;
        assert_eq!(quiet.len(), expected);
        assert!(quiet
            .iter()
            .all(|frame| frame.samples.iter().all(|s| *s == 0)));
    }

    #[test]
    fn scenario_frames_preserve_the_embedded_wav_bytes() {
        for scenario in builtin_scenarios().iter().filter(|s| !s.silence_only) {
            let frames = scenario_frames(scenario).expect("fixture frames");
            let actual: Vec<i16> = frames
                .iter()
                .flat_map(|frame| frame.samples.iter().copied())
                .collect();
            let mut expected = vec![0; LEADING_SILENCE_FRAMES * FRAME_SAMPLES];
            for turn_index in 0..scenario.expected_turns() {
                let fixture =
                    fixtures::audio_fixture(&format!("{}_turn_{}", scenario.id, turn_index + 1))
                        .unwrap();
                expected.extend_from_slice(&fixtures::decode_verified(fixture).unwrap().samples);
                expected.resize(expected.len().div_ceil(FRAME_SAMPLES) * FRAME_SAMPLES, 0);
                expected.extend(std::iter::repeat_n(
                    0,
                    INTER_TURN_SILENCE_FRAMES * FRAME_SAMPLES,
                ));
            }
            assert_eq!(actual, expected, "{}", scenario.id);
        }
    }

    #[test]
    fn missing_or_mismatched_fixture_fails_fast() {
        let missing = &parse_scenarios(r#"{"id": "missing", "turns": ["hi"]}"#).unwrap()[0];
        assert!(scenario_frames(missing).is_err());

        let mismatched =
            &parse_scenarios(r#"{"id": "greeting_001", "turns": ["not the frozen prompt"]}"#)
                .unwrap()[0];
        let err = scenario_frames(mismatched).unwrap_err();
        assert!(err.to_string().contains("does not match fixture"), "{err}");
    }

    #[test]
    fn full_fake_run_produces_rows_with_verdicts_and_timings() {
        let scenarios = parse_scenarios(BUILTIN_SCENARIOS_JSONL).expect("builtin corpus");
        let mut providers = FakeProviders;
        let options = BenchOptions::default();
        let mut events = Vec::new();
        let records = run_bench(
            &mut providers,
            &scenarios,
            &options,
            &mut |line| events.push(line.to_string()),
            &Cancel::new(),
        )
        .expect("fake bench");

        assert_eq!(records.len(), scenarios.len());
        assert_eq!(events.len(), scenarios.len() * 2);

        // Echo LLM replies never contain think tags or markdown; the fake STT
        // transcripts ("turn-000") do not match prompts, so those two gates
        // fail while everything structural passes — proving verdicts flow.
        for record in &records {
            assert_eq!(record.schema_version, SCHEMA_VERSION);
            assert_eq!(record.mode, MODE_FIXTURE);
            assert_eq!(record.profile, "default");
            assert_eq!(record.fingerprint.os, std::env::consts::OS);
            assert_eq!(record.gates.completed_turns, record.turns.len());
            if record.scenario_id == "silence_001" {
                assert_eq!(record.turns.len(), 0);
                assert!(record.passed, "silence-only must pass: {:?}", record.gates);
            } else {
                assert!(!record.passed, "fake STT cannot meet the 0.8 floor");
                for turn in &record.turns {
                    assert!(turn.gates.non_empty_reply);
                    assert!(turn.gates.no_think_leak);
                    assert!(!turn.gates.stt_pass, "echo transcripts miss prompts");
                }
            }
        }
        // Silence-only row vacuously passes its TTS→ASR gate.
        let silence = records.last().expect("silence row");
        assert_eq!(silence.gates.tts_asr_ratio, None);
        assert_eq!(silence.gates.tts_asr_pass, None);
    }

    #[test]
    fn shutdown_cancel_stops_between_scenarios_without_error() {
        let scenarios =
            parse_scenarios(r#"{"id": "a", "turns": ["hello there friend"]}"#).expect("parse");
        let cancel = Cancel::new();
        cancel.shutdown();
        let mut providers = FakeProviders;
        let records = run_bench(
            &mut providers,
            &scenarios,
            &BenchOptions::default(),
            &mut |_| {},
            &cancel,
        )
        .expect("graceful stop");
        assert!(records.is_empty(), "no scenario may run after shutdown");
    }
}
