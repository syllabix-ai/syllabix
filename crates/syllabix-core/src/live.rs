//! Live `syllabix run`: native mic/speakers plus cached on-device providers.

use std::sync::mpsc::Sender;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::error::Result;
use crate::pipeline::{LoopEvent, LoopReport};
use crate::turn_debug::TurnDebug;

/// Load weights, open default devices, and run until shutdown or capture ends.
///
/// Under `cfg(coverage)` this is a one-turn fake loop so llvm-cov does not
/// load whisper.cpp / llama.cpp / Kokoro weights or open devices.
pub fn run_live(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    barge_in: bool,
) -> Result<LoopReport> {
    run_live_inner(config, cancel, events, turn_debug, barge_in)
}

#[cfg(not(coverage))]
fn run_live_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    barge_in: bool,
) -> Result<LoopReport> {
    use crate::audio::{NativeCapture, NativePlayback};
    use crate::models::{HttpFetcher, ModelCache, StderrProgress};
    use crate::pipeline::{run_loop_captured, LoopConfig, LoopMode, PipelineStages};
    use crate::real::load_real_providers;
    use crate::BuiltinDefaults;

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let (vad, stt, llm, tts) =
        load_real_providers(&cache, &HttpFetcher, &mut progress, &cancel, config)?;
    let (sink, echo_reference) = NativePlayback::open_with_echo()?;
    let mut capture = NativeCapture::open_with_echo(echo_reference)?;
    if turn_debug.is_some() {
        capture.enable_pcm_tap();
    }
    eprintln!(
        "mic: {}  speaker: {}  agent: {}",
        capture.device_name, sink.device_name, config.name
    );
    eprintln!("echo: AEC3 on by default; automatic calibration starts with speaker playback");
    run_loop_captured(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events,
            turn_debug,
            barge_in,
        },
        PipelineStages {
            vad,
            stt,
            llm,
            tts,
            sink,
        },
        capture,
        cancel,
    )
}

#[cfg(coverage)]
fn run_live_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    barge_in: bool,
) -> Result<LoopReport> {
    use crate::fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad};
    use crate::pipeline::{run_loop, LoopConfig, PipelineStages};

    let _ = (
        config.language.as_str(),
        config.llm_model.as_str(),
        config.thinking,
        config.vad_threshold,
        config.vad_min_speech_ms,
        config.vad_end_silence_ms,
        config.vad_preroll_ms,
    );
    run_loop(
        LoopConfig {
            events,
            turn_debug,
            barge_in,
            ..LoopConfig::default()
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        cancel,
    )
}

#[cfg(test)]
mod tests {
    use crate::AgentConfig;

    #[test]
    fn live_entry_uses_v0_language() {
        assert_eq!(AgentConfig::v0().language, "en");
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_run_live_completes_a_fake_turn() {
        let report = super::run_live(&AgentConfig::v0(), crate::Cancel::new(), None, None, false)
            .expect("fake live");
        assert_eq!(report.turns.len(), 1);
        assert_eq!(report.tasks_still_running, 0);
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_run_live_turn_debug_writes_without_devices() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-live-turn-debug-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let debug = crate::TurnDebug::open(&dir).expect("open");
        let report = super::run_live(
            &AgentConfig::v0(),
            crate::Cancel::new(),
            None,
            Some(debug),
            false,
        )
        .expect("fake live debug");
        assert_eq!(report.turns.len(), 1);
        assert!(dir.join("turn-000").join("turn.json").is_file());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
