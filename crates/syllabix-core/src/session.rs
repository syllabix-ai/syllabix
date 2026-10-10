//! Host-facing conversation session: defaults for every stage, swap before run.

use std::sync::mpsc::Sender;

use zeroize::Zeroizing;

use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::defaults::LlmProvider;
use crate::error::Result;
use crate::openai::resolve_api_key;
use crate::pipeline::{
    run_loop, run_loop_captured, LoopConfig, LoopEvent, LoopMode, LoopReport, PipelineStages,
    RuntimeControls,
};
use crate::providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
use crate::turn_debug::TurnDebug;
use crate::types::AudioFrame;
use crate::BuiltinDefaults;

enum Input {
    Default,
    Capture(Box<dyn AudioCapture>),
    Frames(Vec<AudioFrame>),
}

/// Compose a conversation loop without wiring [`crate::load_real_providers`],
/// [`crate::HttpFetcher`], and [`crate::audio::NativeCapture`] by hand.
///
/// Unset stages use the same defaults as [`crate::run_live`]: Silero, the
/// configured STT/LLM/TTS from the model cache, and the native mic/speakers.
/// Swap any stage — [`Vad`], [`Stt`], [`Llm`], [`Tts`], [`AudioCapture`], or
/// [`AudioSink`] — before [`Session::run`]. An online LLM in yaml still
/// fails fast on a missing `SYLLABIX_LLM_API_KEY`, but only when that slot
/// is left as the default (a custom [`Llm`] does not need the key).
pub struct Session {
    config: AgentConfig,
    events: Option<Sender<LoopEvent>>,
    barge_in: bool,
    controls: Option<RuntimeControls>,
    mode: LoopMode,
    vad: Option<Box<dyn Vad>>,
    stt: Option<Box<dyn Stt>>,
    llm: Option<Box<dyn Llm>>,
    tts: Option<Box<dyn Tts>>,
    sink: Option<Box<dyn AudioSink>>,
    input: Input,
}

impl Session {
    /// Built-in stack ([`AgentConfig::v0()`]), native devices, no barge-in.
    #[must_use]
    pub fn v0() -> Self {
        Self::from_config(&AgentConfig::v0())
    }

    /// Session for `config`. Stages load when [`Session::run`] is called.
    #[must_use]
    pub fn from_config(config: &AgentConfig) -> Self {
        Self {
            config: config.clone(),
            events: None,
            barge_in: false,
            controls: None,
            mode: LoopMode::UntilInputEnds,
            vad: None,
            stt: None,
            llm: None,
            tts: None,
            sink: None,
            input: Input::Default,
        }
    }

    /// Subscribe to [`LoopEvent`] (UI / logs). `None` disables events.
    #[must_use]
    pub fn with_events(mut self, events: Option<Sender<LoopEvent>>) -> Self {
        self.events = events;
        self
    }

    /// Initial barge-in flag used when [`Session::with_controls`] is not set.
    #[must_use]
    pub fn with_barge_in(mut self, barge_in: bool) -> Self {
        self.barge_in = barge_in;
        self
    }

    /// Live barge-in / mic-mute handle. Replaces the barge-in-derived default.
    #[must_use]
    pub fn with_controls(mut self, controls: RuntimeControls) -> Self {
        self.controls = Some(controls);
        self
    }

    /// Stop condition. Default is [`LoopMode::UntilInputEnds`].
    #[must_use]
    pub fn with_mode(mut self, mode: LoopMode) -> Self {
        self.mode = mode;
        self
    }

    /// Replace Silero (or the coverage fake) with `vad`.
    #[must_use]
    pub fn with_vad(mut self, vad: impl Vad + 'static) -> Self {
        self.vad = Some(Box::new(vad));
        self
    }

    /// Replace the configured STT engine with `stt`.
    #[must_use]
    pub fn with_stt(mut self, stt: impl Stt + 'static) -> Self {
        self.stt = Some(Box::new(stt));
        self
    }

    /// Replace the configured LLM with `llm`. Does not fetch a GGUF or require
    /// `SYLLABIX_LLM_API_KEY`.
    #[must_use]
    pub fn with_llm(mut self, llm: impl Llm + 'static) -> Self {
        self.llm = Some(Box::new(llm));
        self
    }

    /// Replace the configured TTS engine with `tts`.
    #[must_use]
    pub fn with_tts(mut self, tts: impl Tts + 'static) -> Self {
        self.tts = Some(Box::new(tts));
        self
    }

    /// Replace native speakers with `sink`.
    #[must_use]
    pub fn with_sink(mut self, sink: impl AudioSink + 'static) -> Self {
        self.sink = Some(Box::new(sink));
        self
    }

    /// Replace the native microphone with `capture`. Last of this and
    /// [`Session::with_frames`] wins.
    #[must_use]
    pub fn with_capture(mut self, capture: impl AudioCapture + 'static) -> Self {
        self.input = Input::Capture(Box::new(capture));
        self
    }

    /// Drive the loop from in-memory PCM instead of a microphone. Last of this
    /// and [`Session::with_capture`] wins.
    #[must_use]
    pub fn with_frames(mut self, frames: impl IntoIterator<Item = AudioFrame>) -> Self {
        self.input = Input::Frames(frames.into_iter().collect());
        self
    }

    /// Load any remaining default stages and run until the loop finishes.
    pub fn run(self, cancel: Cancel) -> Result<LoopReport> {
        let Session {
            config,
            events,
            barge_in,
            controls,
            mode,
            vad,
            stt,
            llm,
            tts,
            sink,
            input,
        } = self;
        let llm_api_key = if llm.is_none() && config.llm == LlmProvider::Online {
            Some(resolve_api_key(|name| std::env::var(name).ok())?)
        } else {
            None
        };
        let turn_debug = TurnDebug::from_config(&config)?;
        let controls = controls.unwrap_or_else(|| {
            RuntimeControls::with_auto_timeout(
                barge_in,
                config.auto_timeout_mic_mute_ms,
                config.auto_timeout_exit_ms,
            )
        });
        let (vad, stt, llm, tts) =
            resolve_ml(&config, &cancel, llm_api_key.as_ref(), vad, stt, llm, tts)?;
        let (input, sink) = resolve_io(&config, &events, turn_debug.as_ref(), input, sink)?;
        if let Some(events) = &events {
            let _ = events.send(LoopEvent::Ready);
        }
        let loop_config = LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode,
            events,
            turn_debug,
            controls,
        };
        let stages = PipelineStages {
            vad,
            stt,
            llm,
            tts,
            sink,
        };
        match input {
            ResolvedInput::Frames(frames) => run_loop(loop_config, stages, frames, cancel),
            ResolvedInput::Capture(capture) => {
                run_loop_captured(loop_config, stages, capture, cancel)
            }
        }
    }
}

enum ResolvedInput {
    Frames(Vec<AudioFrame>),
    Capture(Box<dyn AudioCapture>),
}

type BoxedStages = (Box<dyn Vad>, Box<dyn Stt>, Box<dyn Llm>, Box<dyn Tts>);

#[cfg(not(coverage))]
fn resolve_ml(
    config: &AgentConfig,
    cancel: &Cancel,
    llm_api_key: Option<&Zeroizing<String>>,
    vad: Option<Box<dyn Vad>>,
    stt: Option<Box<dyn Stt>>,
    llm: Option<Box<dyn Llm>>,
    tts: Option<Box<dyn Tts>>,
) -> Result<BoxedStages> {
    use crate::models::{HttpFetcher, ModelCache, StderrProgress};
    use crate::openai::PROVIDER_NAME;
    use crate::real::{build_llm, build_stt, build_tts, LiveLlm};
    use crate::vad::SileroVad;
    use crate::Error;

    let need_cache = vad.is_none() || stt.is_none() || llm.is_none() || tts.is_none();
    let cache = need_cache.then(ModelCache::v0);
    let fetcher = HttpFetcher;
    let mut progress = StderrProgress::new();

    let vad = match vad {
        Some(vad) => vad,
        None => {
            let cache = cache.as_ref().expect("cache when loading default VAD");
            Box::new(
                SileroVad::from_cache(cache, &fetcher, &mut progress, cancel)?
                    .with_settings(config.vad_settings()),
            )
        }
    };
    let stt = match stt {
        Some(stt) => stt,
        None => {
            let cache = cache.as_ref().expect("cache when loading default STT");
            Box::new(
                build_stt(cache, &fetcher, &mut progress, cancel, config)?
                    .with_language(&config.language)?,
            )
        }
    };
    let llm = match llm {
        Some(llm) => llm,
        None => {
            let cache = cache.as_ref().expect("cache when loading default LLM");
            let built = build_llm(cache, &fetcher, &mut progress, cancel, config, llm_api_key)?;
            if let LiveLlm::Cloud(cloud) = &built {
                match cloud.warm_up(cancel) {
                    Ok(()) => eprintln!("cloud: endpoint ready"),
                    Err(Error::Cancelled) => return Err(Error::Cancelled),
                    Err(err) => {
                        tracing::warn!(provider = PROVIDER_NAME, error = %err, "cloud warm-up failed");
                        eprintln!(
                            "cloud: could not verify endpoint; continuing — the first reply may fail"
                        );
                    }
                }
            }
            Box::new(built)
        }
    };
    let tts = match tts {
        Some(tts) => tts,
        None => {
            let cache = cache.as_ref().expect("cache when loading default TTS");
            Box::new(build_tts(cache, &fetcher, &mut progress, cancel, config)?)
        }
    };
    Ok((vad, stt, llm, tts))
}

#[cfg(coverage)]
fn resolve_ml(
    config: &AgentConfig,
    _cancel: &Cancel,
    llm_api_key: Option<&Zeroizing<String>>,
    vad: Option<Box<dyn Vad>>,
    stt: Option<Box<dyn Stt>>,
    llm: Option<Box<dyn Llm>>,
    tts: Option<Box<dyn Tts>>,
) -> Result<BoxedStages> {
    use crate::fake::{FakeLlm, FakeStt, FakeTts, FakeVad};

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
    Ok((
        vad.unwrap_or_else(|| Box::new(FakeVad::new())),
        stt.unwrap_or_else(|| Box::new(FakeStt)),
        llm.unwrap_or_else(|| Box::new(FakeLlm::new())),
        tts.unwrap_or_else(|| Box::new(FakeTts)),
    ))
}

#[cfg(not(coverage))]
fn resolve_io(
    config: &AgentConfig,
    events: &Option<Sender<LoopEvent>>,
    turn_debug: Option<&TurnDebug>,
    input: Input,
    sink: Option<Box<dyn AudioSink>>,
) -> Result<(ResolvedInput, Box<dyn AudioSink>)> {
    use crate::audio::{NativeCapture, NativePlayback};
    use crate::turn_debug::PlaybackWatch;

    let watch = turn_debug.map(|debug| PlaybackWatch::new(debug.clone()));
    let wants_wavs = turn_debug.is_some_and(TurnDebug::collects_audio);
    match (input, sink) {
        (Input::Default, None) => {
            let (sink, echo_reference) = NativePlayback::open_with_echo_and_watch(watch)?;
            let mut capture =
                NativeCapture::open_with_echo_and_events(echo_reference, events.clone())?;
            if wants_wavs {
                capture.enable_pcm_tap();
            }
            eprintln!(
                "mic: {}  speaker: {}  agent: {}",
                capture.device_name, sink.device_name, config.name
            );
            Ok((ResolvedInput::Capture(Box::new(capture)), Box::new(sink)))
        }
        (Input::Default, Some(sink)) => {
            let mut capture = NativeCapture::open()?;
            if wants_wavs {
                capture.enable_pcm_tap();
            }
            eprintln!("mic: {}  agent: {}", capture.device_name, config.name);
            Ok((ResolvedInput::Capture(Box::new(capture)), sink))
        }
        (Input::Capture(capture), None) => {
            let sink = default_playback(watch)?;
            eprintln!("speaker: {}  agent: {}", sink.device_name, config.name);
            Ok((ResolvedInput::Capture(capture), Box::new(sink)))
        }
        (Input::Frames(frames), None) => {
            let sink = default_playback(watch)?;
            eprintln!("speaker: {}  agent: {}", sink.device_name, config.name);
            Ok((ResolvedInput::Frames(frames), Box::new(sink)))
        }
        (Input::Capture(capture), Some(sink)) => Ok((ResolvedInput::Capture(capture), sink)),
        (Input::Frames(frames), Some(sink)) => Ok((ResolvedInput::Frames(frames), sink)),
    }
}

#[cfg(not(coverage))]
fn default_playback(
    watch: Option<crate::turn_debug::PlaybackWatch>,
) -> Result<crate::audio::NativePlayback> {
    use crate::audio::NativePlayback;
    if watch.is_some() {
        Ok(NativePlayback::open_with_echo_and_watch(watch)?.0)
    } else {
        NativePlayback::open()
    }
}

#[cfg(coverage)]
fn resolve_io(
    config: &AgentConfig,
    _events: &Option<Sender<LoopEvent>>,
    turn_debug: Option<&TurnDebug>,
    input: Input,
    sink: Option<Box<dyn AudioSink>>,
) -> Result<(ResolvedInput, Box<dyn AudioSink>)> {
    use crate::fake::{scripted_frames, CollectingSink};
    use crate::turn_debug::PlaybackWatch;

    let _ = config.name.as_str();
    let watch = turn_debug.map(|debug| PlaybackWatch::new(debug.clone()));
    let sink: Box<dyn AudioSink> = match sink {
        Some(sink) => sink,
        None => Box::new(CollectingSink::with_playback_watch(watch)),
    };
    let input = match input {
        Input::Capture(capture) => ResolvedInput::Capture(capture),
        Input::Frames(frames) => ResolvedInput::Frames(frames),
        Input::Default => ResolvedInput::Frames(scripted_frames(1, 2, 1)),
    };
    Ok((input, sink))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad};
    use crate::openai::API_KEY_ENV;
    use crate::Error;

    struct FrameCapture {
        frames: std::vec::IntoIter<AudioFrame>,
    }

    impl AudioCapture for FrameCapture {
        fn name(&self) -> &'static str {
            "fixture"
        }

        fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let frame = self.frames.next();
            if frame.is_some() {
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Ok(frame)
        }
    }

    fn swapped(session: Session) -> Session {
        session
            .with_vad(FakeVad::new())
            .with_stt(FakeStt)
            .with_llm(FakeLlm::new())
            .with_tts(FakeTts)
            .with_sink(CollectingSink::default())
            .with_frames(scripted_frames(1, 2, 1))
    }

    #[test]
    fn swapped_stages_complete_a_turn() {
        let report = swapped(Session::v0())
            .run(Cancel::new())
            .expect("fake session");
        assert_eq!(report.turns.len(), 1);
        assert_eq!(report.tasks_still_running, 0);
    }

    #[test]
    fn from_config_matches_v0_for_the_built_in_stack() {
        let report = swapped(Session::from_config(&AgentConfig::v0()))
            .with_mode(LoopMode::UntilInputEnds)
            .with_barge_in(false)
            .run(Cancel::new())
            .expect("from_config");
        assert_eq!(report.turns.len(), 1);
    }

    #[test]
    fn with_events_emits_ready() {
        let (tx, rx) = std::sync::mpsc::channel();
        swapped(Session::v0())
            .with_events(Some(tx))
            .run(Cancel::new())
            .expect("events");
        let events: Vec<_> = rx.try_iter().collect();
        assert!(
            events.iter().any(|event| matches!(event, LoopEvent::Ready)),
            "{events:?}"
        );
    }

    #[test]
    fn with_controls_keeps_barge_in() {
        let controls = RuntimeControls::new(true);
        assert!(controls.barge_in());
        swapped(Session::v0())
            .with_controls(controls.clone())
            .run(Cancel::new())
            .expect("controls");
        assert!(controls.barge_in());
    }

    #[test]
    fn with_capture_runs_the_loop() {
        let report = Session::v0()
            .with_vad(FakeVad::new())
            .with_stt(FakeStt)
            .with_llm(FakeLlm::new())
            .with_tts(FakeTts)
            .with_sink(CollectingSink::default())
            .with_frames(scripted_frames(2, 2, 1))
            .with_capture(FrameCapture {
                frames: scripted_frames(1, 2, 1).into_iter(),
            })
            .run(Cancel::new())
            .expect("capture");
        assert_eq!(report.turns.len(), 1);
    }

    fn without_api_key<T>(f: impl FnOnce() -> T) -> T {
        let _lock = env_lock().lock().expect("api key lock");
        let prior = std::env::var_os(API_KEY_ENV);
        std::env::remove_var(API_KEY_ENV);
        let out = f();
        match prior {
            Some(value) => std::env::set_var(API_KEY_ENV, value),
            None => std::env::remove_var(API_KEY_ENV),
        }
        out
    }

    fn env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        &LOCK
    }

    #[test]
    fn default_online_llm_fails_fast_without_a_key() {
        let mut config = AgentConfig::v0();
        config.llm = LlmProvider::Online;
        let err = without_api_key(|| {
            Session::from_config(&config)
                .run(Cancel::new())
                .expect_err("missing key")
        });
        assert!(err.to_string().contains(API_KEY_ENV), "{err}");
    }

    #[test]
    fn swapped_llm_skips_the_online_key() {
        let mut config = AgentConfig::v0();
        config.llm = LlmProvider::Online;
        let report = without_api_key(|| {
            swapped(Session::from_config(&config))
                .run(Cancel::new())
                .expect("custom llm needs no key")
        });
        assert_eq!(report.turns.len(), 1);
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_defaults_complete_a_fake_turn() {
        let report = Session::v0().run(Cancel::new()).expect("fake defaults");
        assert_eq!(report.turns.len(), 1);
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_defaults_write_diagnostics_without_devices() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-session-diagnostics-{}-{}",
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
        let report = Session::from_config(&config)
            .run(Cancel::new())
            .expect("fake session diagnostics");
        assert_eq!(report.turns.len(), 1);
        let turn_dir = dir.join("turn-000");
        assert!(turn_dir.join("turn.json").is_file());
        for name in ["capture.wav", "clean.wav", "utterance.wav", "tts.wav"] {
            assert!(turn_dir.join(name).is_file(), "{name}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_partial_overrides_still_fill_defaults() {
        let report = Session::v0()
            .with_llm(FakeLlm::new())
            .with_sink(CollectingSink::default())
            .run(Cancel::new())
            .expect("partial override");
        assert_eq!(report.turns.len(), 1);
    }
}
