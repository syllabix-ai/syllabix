//! Live `syllabix run`: native mic/speakers plus cached on-device providers.

use std::sync::mpsc::Sender;

use zeroize::Zeroizing;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::error::Result;
use crate::openai::resolve_api_key;
use crate::pipeline::{LoopEvent, LoopReport};
use crate::turn_debug::TurnDebug;

/// Load weights, open default devices, and run until shutdown or capture ends.
///
/// Diagnostics come from yaml (`diagnostics.timestamps` / `diagnostics.audio`
/// / `diagnostics.directory`); disabled by default, which writes nothing.
///
/// A cloud LLM config (`pipeline.llm.provider: online`) must already carry its
/// key in the `SYLLABIX_LLM_API_KEY` environment variable; a missing or empty
/// key fails here — before any device opens or any weight loads. Default
/// keyless runs never read it.
///
/// Under `cfg(coverage)` this is a one-turn fake loop so llvm-cov does not
/// load whisper.cpp / llama.cpp / Kokoro weights or open devices.
pub fn run_live(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    barge_in: bool,
) -> Result<LoopReport> {
    let llm_api_key = match config.llm {
        crate::LlmProvider::Online => Some(resolve_api_key(|name| std::env::var(name).ok())?),
        crate::LlmProvider::Local => None,
    };
    let turn_debug = TurnDebug::from_config(config)?;
    run_live_inner(config, cancel, events, turn_debug, barge_in, llm_api_key)
}

#[cfg(not(coverage))]
fn run_live_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    barge_in: bool,
    llm_api_key: Option<Zeroizing<String>>,
) -> Result<LoopReport> {
    use crate::audio::{NativeCapture, NativePlayback};
    use crate::models::{HttpFetcher, ModelCache, StderrProgress};
    use crate::pipeline::{run_loop_captured, LoopConfig, LoopMode, PipelineStages};
    use crate::real::load_real_providers;
    use crate::turn_debug::PlaybackWatch;
    use crate::BuiltinDefaults;

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let (vad, stt, llm, tts) = load_real_providers(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        config,
        llm_api_key.as_ref(),
    )?;
    let watch = turn_debug
        .as_ref()
        .map(|debug| PlaybackWatch::new(debug.clone()));
    let wants_wavs = turn_debug
        .as_ref()
        .is_some_and(|debug| debug.collects_audio());
    let (sink, echo_reference) = NativePlayback::open_with_echo_and_watch(watch)?;
    let mut capture = NativeCapture::open_with_echo(echo_reference)?;
    if wants_wavs {
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
    llm_api_key: Option<Zeroizing<String>>,
) -> Result<LoopReport> {
    use crate::fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad};
    use crate::pipeline::{run_loop, LoopConfig, PipelineStages};

    let _ = (
        config.language.as_str(),
        config.stt_model.asset_id(),
        config.llm_model.as_str(),
        config.thinking,
        config.system_prompt.as_str(),
        config.vad_threshold,
        config.vad_min_speech_ms,
        config.vad_end_silence_ms,
        config.vad_preroll_ms,
        config.diagnostics_timestamps,
        config.diagnostics_audio,
        llm_api_key.is_some(),
    );
    let watch = turn_debug
        .as_ref()
        .map(|debug| crate::turn_debug::PlaybackWatch::new(debug.clone()));
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
            sink: CollectingSink::with_playback_watch(watch),
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
        let report = super::run_live(&AgentConfig::v0(), crate::Cancel::new(), None, false)
            .expect("fake live");
        assert_eq!(report.turns.len(), 1);
        assert_eq!(report.tasks_still_running, 0);
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_run_live_diagnostics_writes_sidecar_and_wavs_without_devices() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-live-diagnostics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut config = AgentConfig::v0();
        config.diagnostics_timestamps = true;
        config.diagnostics_audio = true;
        config.diagnostics_directory = dir.clone();
        let report = super::run_live(&config, crate::Cancel::new(), None, false)
            .expect("fake live with diagnostics");
        assert_eq!(report.turns.len(), 1);
        let turn_dir = dir.join("turn-000");
        assert!(turn_dir.join("turn.json").is_file());
        for name in ["capture.wav", "clean.wav", "utterance.wav", "tts.wav"] {
            assert!(turn_dir.join(name).is_file(), "{name}");
        }
        let json =
            std::fs::read_to_string(turn_dir.join("turn.json")).expect("diagnostics sidecar");
        assert!(json.contains("\"speech_start_ms\": 0"), "{json}");
        assert!(
            json.contains("\"playback_first_ms\":"),
            "watch-backed fixture sink records playback anchors: {json}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_run_live_disabled_diagnostics_writes_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-live-no-diagnostics-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut config = AgentConfig::v0();
        // Disabled diagnostics must not create the directory at all.
        config.diagnostics_directory = dir.join("never");
        super::run_live(&config, crate::Cancel::new(), None, false).expect("fake live");
        assert!(!dir.exists(), "no files when diagnostics are off");
    }
}
