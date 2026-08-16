//! Live `syllabix run`: native mic/speakers plus cached on-device providers.

use std::sync::mpsc::Sender;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::error::Result;
use crate::pipeline::{LoopEvent, LoopReport};

/// Load weights, open default devices, and run until shutdown or capture ends.
///
/// Under `cfg(coverage)` this is a one-turn fake loop so llvm-cov does not
/// load whisper.cpp / llama.cpp / Kokoro weights or open devices.
pub fn run_live(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
) -> Result<LoopReport> {
    run_live_inner(config, cancel, events)
}

#[cfg(not(coverage))]
fn run_live_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
) -> Result<LoopReport> {
    use crate::audio::{NativeCapture, NativePlayback};
    use crate::models::{HttpFetcher, ModelCache, StderrProgress};
    use crate::pipeline::{run_loop_captured, LoopConfig, LoopMode, PipelineStages};
    use crate::real::load_real_providers;
    use crate::BuiltinDefaults;

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let (vad, stt, llm, tts) = load_real_providers(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        &config.language,
    )?;
    let (sink, echo_reference) = NativePlayback::open_with_echo()?;
    let capture = NativeCapture::open_with_echo(echo_reference)?;
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
) -> Result<LoopReport> {
    use crate::fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad};
    use crate::pipeline::{run_loop, LoopConfig, PipelineStages};

    let _ = config.language.as_str();
    run_loop(
        LoopConfig {
            events,
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
        let report =
            super::run_live(&AgentConfig::v0(), crate::Cancel::new(), None).expect("fake live");
        assert_eq!(report.turns.len(), 1);
        assert_eq!(report.tasks_still_running, 0);
    }
}
