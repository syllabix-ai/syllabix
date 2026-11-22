//! Live `syllabix run`: native mic/speakers plus cached on-device providers.

use std::sync::mpsc::Sender;

use zeroize::Zeroizing;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::error::Result;
use crate::openai::resolve_api_key;
use crate::pipeline::{LoopEvent, LoopReport, RuntimeControls};
use crate::turn_debug::TurnDebug;

/// Load weights, open default devices, and run until shutdown or capture ends.
///
/// Diagnostics come from yaml (`diagnostics.timestamps` / `diagnostics.audio`
/// / `diagnostics.directory`); disabled by default, which writes nothing.
///
/// A builtin cloud LLM config (`pipeline.llm.provider: openai-compatible`) must already
/// carry its key in the `SYLLABIX_LLM_API_KEY` environment variable; a missing
/// or empty key fails here — before any device opens or any weight loads.
/// With `agent_harness.agent: goose`, this function does not pass that key into
/// the builtin LLM slot, but starting Goose still maps `SYLLABIX_LLM_API_KEY` (or
/// provider-specific passthrough credentials) into the Goose child from the
/// environment when `pipeline.llm` requires it. Online STT
/// (`pipeline.stt.provider: online`) does the same with `SYLLABIX_STT_API_KEY`.
/// Online TTS does the same with `SYLLABIX_TTS_API_KEY`. Default keyless
/// runs never read them.
///
/// Under `cfg(coverage)` this is a one-turn fake loop so llvm-cov does not
/// load whisper.cpp / llama.cpp / Kokoro weights or open devices.
pub fn run_live(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    barge_in: bool,
) -> Result<LoopReport> {
    run_live_with_controls(
        config,
        cancel,
        events,
        RuntimeControls::with_auto_timeout(
            barge_in,
            config.auto_timeout_mic_mute_ms,
            config.auto_timeout_exit_ms,
        ),
    )
}

/// As [`run_live`], with controls that the inline terminal may change while running.
pub fn run_live_with_controls(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    controls: RuntimeControls,
) -> Result<LoopReport> {
    use crate::config::{PipelineMode, S2sReplyWith};
    use crate::s2s::LFM3_AUDIO_2B;

    let lfm3_s2s = config.pipeline_mode == PipelineMode::S2s
        && config.s2s.as_ref().and_then(|s| s.model.as_deref()) == Some(LFM3_AUDIO_2B);
    let s2s_reply_llm = lfm3_s2s
        && config
            .s2s
            .as_ref()
            .is_some_and(|s| s.reply_with == S2sReplyWith::Llm);
    config.ensure_pipeline_mode_supported()?;
    if s2s_reply_llm {
        crate::session_env::install_from_config(config);
        #[cfg(all(
            not(coverage),
            feature = "ggml",
            feature = "lfm3-s2s",
            feature = "device-audio"
        ))]
        {
            let llm_api_key = match (&config.agent, &config.llm) {
                (crate::defaults::AgentBackend::Goose, _) => None,
                (_, llm) if llm.is_openai_compatible() => {
                    Some(crate::openai::resolve_api_key(|name| {
                        std::env::var(name).ok()
                    })?)
                }
                (_, _) => None,
            };
            let turn_debug = TurnDebug::from_config(config)?;
            return run_live_s2s_llm_inner(
                config,
                cancel,
                events,
                turn_debug,
                controls,
                llm_api_key,
            );
        }
        #[cfg(coverage)]
        {
            let turn_debug = TurnDebug::from_config(config)?;
            return run_live_s2s_llm_inner(config, cancel, events, turn_debug, controls, None);
        }
        #[cfg(not(any(
            coverage,
            all(
                not(coverage),
                feature = "ggml",
                feature = "lfm3-s2s",
                feature = "device-audio"
            )
        )))]
        {
            return Err(crate::Error::Config {
                field: "pipeline.mode".into(),
                message: "s2s is not supported yet".into(),
            });
        }
    }
    if lfm3_s2s {
        crate::session_env::install_from_config(config);
        #[cfg(all(
            not(coverage),
            feature = "ggml",
            feature = "lfm3-s2s",
            feature = "device-audio"
        ))]
        {
            let turn_debug = TurnDebug::from_config(config)?;
            return run_live_s2s_inner(config, cancel, events, turn_debug, controls);
        }
        #[cfg(not(all(
            not(coverage),
            feature = "ggml",
            feature = "lfm3-s2s",
            feature = "device-audio"
        )))]
        {
            return Err(crate::Error::Config {
                field: "pipeline.mode".into(),
                message: "s2s is not supported yet".into(),
            });
        }
    }
    crate::session_env::install_from_config(config);
    let llm_api_key = match (&config.agent, &config.llm) {
        (crate::defaults::AgentBackend::Goose, _) => None,
        (_, llm) if llm.is_openai_compatible() => {
            Some(resolve_api_key(|name| std::env::var(name).ok())?)
        }
        (_, _) => None,
    };
    let stt_api_key = match config.stt {
        crate::defaults::SttProvider::Online => {
            #[cfg(feature = "cloud")]
            {
                Some(crate::remote_stt::resolve_api_key(|name| {
                    std::env::var(name).ok()
                })?)
            }
            #[cfg(not(feature = "cloud"))]
            {
                None
            }
        }
        crate::defaults::SttProvider::Local => None,
    };
    let tts_api_key = match config.tts {
        crate::defaults::TtsProvider::Online => {
            #[cfg(feature = "cloud")]
            {
                Some(crate::remote_tts::resolve_api_key(|name| {
                    std::env::var(name).ok()
                })?)
            }
            #[cfg(not(feature = "cloud"))]
            {
                None
            }
        }
        crate::defaults::TtsProvider::Local => None,
    };
    let turn_debug = TurnDebug::from_config(config)?;
    run_live_inner(
        config,
        cancel,
        events,
        turn_debug,
        controls,
        llm_api_key,
        stt_api_key,
        tts_api_key,
    )
}

#[cfg(all(
    not(coverage),
    feature = "ggml",
    feature = "lfm3-s2s",
    feature = "device-audio"
))]
fn run_live_s2s_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    controls: RuntimeControls,
) -> Result<LoopReport> {
    use crate::audio::{NativeCapture, NativePlayback};
    use crate::defaults::BuiltinDefaults;
    use crate::models::{HttpFetcher, ModelCache, StderrProgress};
    use crate::pipeline::{run_s2s_loop_captured, LoopConfig, LoopMode, S2sStages};
    use crate::real::load_s2s_engine;
    use crate::turn_debug::PlaybackWatch;
    use crate::vad::SileroVad;

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let watch = turn_debug
        .as_ref()
        .map(|debug| PlaybackWatch::new(debug.clone()));
    let wants_wavs = turn_debug
        .as_ref()
        .is_some_and(|debug| debug.collects_audio());
    let (sink, echo_reference) = NativePlayback::open_with_echo_and_watch(watch)?;
    let mut capture = NativeCapture::open_with_echo_and_events(echo_reference, events.clone())?;
    let vad = SileroVad::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)?
        .with_settings(config.vad_settings());
    let engine = load_s2s_engine(&cache, &HttpFetcher, &mut progress, &cancel, config)?;
    capture.discard_buffered_audio();
    if wants_wavs {
        capture.enable_pcm_tap();
    }
    eprintln!(
        "mic: {}  speaker: {}  agent: {}  mode: s2s",
        capture.device_name, sink.device_name, config.name
    );
    if let Some(events) = &events {
        let _ = events.send(LoopEvent::Ready);
    }
    run_s2s_loop_captured(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events,
            turn_debug,
            controls,
        },
        S2sStages { vad, engine, sink },
        capture,
        cancel,
    )
}

#[cfg(all(
    not(coverage),
    feature = "ggml",
    feature = "lfm3-s2s",
    feature = "device-audio"
))]
#[allow(clippy::too_many_arguments)]
fn run_live_s2s_llm_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    controls: RuntimeControls,
    llm_api_key: Option<Zeroizing<String>>,
) -> Result<LoopReport> {
    use std::sync::{Arc, Mutex};

    use crate::audio::{NativeCapture, NativePlayback};
    use crate::defaults::BuiltinDefaults;
    use crate::models::{HttpFetcher, ModelCache, StderrProgress};
    use crate::pipeline::{run_loop_captured, LoopConfig, LoopMode, PipelineStages};
    use crate::real::build_llm;
    use crate::real::load_s2s_engine;
    use crate::s2s_speech::{S2sStt, S2sTts};
    use crate::turn_debug::PlaybackWatch;
    use crate::vad::SileroVad;

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let watch = turn_debug
        .as_ref()
        .map(|debug| PlaybackWatch::new(debug.clone()));
    let wants_wavs = turn_debug
        .as_ref()
        .is_some_and(|debug| debug.collects_audio());
    let (sink, echo_reference) = NativePlayback::open_with_echo_and_watch(watch)?;
    let mut capture = NativeCapture::open_with_echo_and_events(echo_reference, events.clone())?;
    let vad = SileroVad::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)?
        .with_settings(config.vad_settings());
    let engine = load_s2s_engine(&cache, &HttpFetcher, &mut progress, &cancel, config)?;
    let llm = build_llm(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        config,
        llm_api_key.as_ref(),
    )?;
    let s2s_cfg = config.s2s.as_ref().expect("s2s block");
    let shared: Arc<Mutex<Box<dyn crate::s2s::S2sEngine>>> = Arc::new(Mutex::new(Box::new(engine)));
    let stt = S2sStt::new(
        Arc::clone(&shared),
        s2s_cfg.language.clone(),
        &config.language,
    );
    let tts = S2sTts::new(shared, s2s_cfg.language.clone(), s2s_cfg.voice.clone());
    capture.discard_buffered_audio();
    if wants_wavs {
        capture.enable_pcm_tap();
    }
    eprintln!(
        "mic: {}  speaker: {}  agent: {}  mode: s2s  reply_with: llm",
        capture.device_name, sink.device_name, config.name
    );
    if let Some(events) = &events {
        let _ = events.send(LoopEvent::Ready);
    }
    run_loop_captured(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events,
            turn_debug,
            controls,
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
#[allow(clippy::too_many_arguments)]
fn run_live_s2s_llm_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    controls: RuntimeControls,
    llm_api_key: Option<Zeroizing<String>>,
) -> Result<LoopReport> {
    use crate::fake::{scripted_frames, CollectingSink, FakeLlm, FakeS2s, FakeVad};
    use crate::pipeline::{run_loop, LoopConfig, PipelineStages};
    use crate::s2s_speech::{S2sStt, S2sTts};

    let _ = llm_api_key;
    let _ = config.s2s.as_ref().and_then(|s| s.language.clone());
    let _ = config.s2s.as_ref().and_then(|s| s.voice.clone());
    let shared = std::sync::Arc::new(std::sync::Mutex::new(
        Box::new(FakeS2s::new()) as Box<dyn crate::s2s::S2sEngine>
    ));
    let stt = S2sStt::new(
        std::sync::Arc::clone(&shared),
        config.s2s.as_ref().and_then(|s| s.language.clone()),
        &config.language,
    );
    let tts = S2sTts::new(
        shared,
        config.s2s.as_ref().and_then(|s| s.language.clone()),
        config.s2s.as_ref().and_then(|s| s.voice.clone()),
    );
    run_loop(
        LoopConfig {
            events,
            turn_debug,
            controls,
            ..LoopConfig::default()
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt,
            llm: FakeLlm::new(),
            tts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        cancel,
    )
}

#[cfg(not(coverage))]
#[allow(clippy::too_many_arguments)]
fn run_live_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    controls: RuntimeControls,
    llm_api_key: Option<Zeroizing<String>>,
    stt_api_key: Option<Zeroizing<String>>,
    tts_api_key: Option<Zeroizing<String>>,
) -> Result<LoopReport> {
    use crate::audio::{NativeCapture, NativePlayback};
    use crate::defaults::BuiltinDefaults;
    use crate::models::{HttpFetcher, ModelCache, StderrProgress};
    use crate::pipeline::{run_loop_captured, LoopConfig, LoopMode, PipelineStages};
    use crate::real::load_real_providers;
    use crate::turn_debug::PlaybackWatch;

    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let watch = turn_debug
        .as_ref()
        .map(|debug| PlaybackWatch::new(debug.clone()));
    let wants_wavs = turn_debug
        .as_ref()
        .is_some_and(|debug| debug.collects_audio());
    // Open both devices before the model load. Some backends refuse a valid
    // stream while the process is busy paging in large weights; an already
    // running stream is left alone, and the devices render silence and record
    // until the pipeline needs them.
    let (sink, echo_reference) = NativePlayback::open_with_echo_and_watch(watch)?;
    let mut capture = NativeCapture::open_with_echo_and_events(echo_reference, events.clone())?;
    let (vad, stt, llm, tts) = load_real_providers(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        config,
        llm_api_key.as_ref(),
        stt_api_key.as_ref(),
        tts_api_key.as_ref(),
    )?;
    // Everything recorded during the load predates the agent being ready.
    capture.discard_buffered_audio();
    if wants_wavs {
        capture.enable_pcm_tap();
    }
    eprintln!(
        "mic: {}  speaker: {}  agent: {}",
        capture.device_name, sink.device_name, config.name
    );
    if let Some(events) = &events {
        let _ = events.send(LoopEvent::Ready);
    }
    run_loop_captured(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events,
            turn_debug,
            controls,
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
#[allow(clippy::too_many_arguments)]
fn run_live_inner(
    config: &AgentConfig,
    cancel: Cancel,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    controls: RuntimeControls,
    llm_api_key: Option<Zeroizing<String>>,
    stt_api_key: Option<Zeroizing<String>>,
    tts_api_key: Option<Zeroizing<String>>,
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
        stt_api_key.is_some(),
        tts_api_key.is_some(),
    );
    let watch = turn_debug
        .as_ref()
        .map(|debug| crate::turn_debug::PlaybackWatch::new(debug.clone()));
    run_loop(
        LoopConfig {
            events,
            turn_debug,
            controls,
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
    fn coverage_run_live_trajectory_writes_the_turn_steps() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-live-trajectory-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut config = AgentConfig::v0();
        config.diagnostics_trajectory = true;
        config.diagnostics_directory = dir.clone();
        super::run_live(&config, crate::Cancel::new(), None, false)
            .expect("fake live with trajectory");
        let body = std::fs::read_to_string(dir.join("turn-000").join("trajectory.jsonl"))
            .expect("trajectory log");
        assert!(body.contains("\"kind\":\"user\""), "{body}");
        assert!(body.contains("\"kind\":\"assistant\""), "{body}");
        assert!(body.contains("\"kind\":\"outcome\""), "{body}");
        assert!(body.contains("\"outcome\":\"completed\""), "{body}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_run_live_lfm3_s2s_fails_without_native_stack() {
        use crate::config::{PipelineMode, S2sConfig, S2sReplyWith};
        use crate::s2s::LFM3_AUDIO_2B;

        let mut config = AgentConfig::v0();
        config.pipeline_mode = PipelineMode::S2s;
        config.s2s = Some(S2sConfig {
            model: Some(LFM3_AUDIO_2B.to_string()),
            ..Default::default()
        });
        let err = super::run_live(&config, crate::Cancel::new(), None, false).unwrap_err();
        assert!(err.to_string().contains("pipeline.mode"), "{err}");
        assert!(err.to_string().contains("not supported yet"), "{err}");

        config.s2s = Some(S2sConfig {
            model: Some(LFM3_AUDIO_2B.to_string()),
            reply_with: S2sReplyWith::Llm,
            ..Default::default()
        });
        let report =
            super::run_live(&config, crate::Cancel::new(), None, false).expect("fake llm s2s live");
        assert!(report.turns.len() >= 1);
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
