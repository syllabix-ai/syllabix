//! S2S `reply_with: llm` cascade routing (fakes only).

use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::Duration;

use syllabix_core::config::{
    AgentConfig, PipelineMode, S2sConfig, S2sReplyWith, STT_API_KEY_ENV, TTS_API_KEY_ENV,
};
use syllabix_core::defaults::{SttProvider, TtsProvider};
use syllabix_core::pipeline::LoopMode;
use syllabix_core::stt::language::LANGUAGE_AUTO;
use syllabix_core::testing::{
    scripted_frames, CollectingSink, FakeLlm, FakeS2s, FakeS2sOp, FakeVad, FAKE_S2S_TRANSCRIPT,
};
use syllabix_core::{Cancel, LoopEvent, Session, TurnId};

fn api_key_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn with_cleared_stt_tts_keys<T>(f: impl FnOnce() -> T) -> T {
    let _lock = api_key_env_lock().lock().expect("stt/tts env lock");
    let prior_stt = std::env::var_os(STT_API_KEY_ENV);
    let prior_tts = std::env::var_os(TTS_API_KEY_ENV);
    std::env::remove_var(STT_API_KEY_ENV);
    std::env::remove_var(TTS_API_KEY_ENV);
    let out = f();
    match prior_stt {
        Some(v) => std::env::set_var(STT_API_KEY_ENV, v),
        None => std::env::remove_var(STT_API_KEY_ENV),
    }
    match prior_tts {
        Some(v) => std::env::set_var(TTS_API_KEY_ENV, v),
        None => std::env::remove_var(TTS_API_KEY_ENV),
    }
    out
}

fn llm_s2s_config() -> AgentConfig {
    let mut config = AgentConfig::v0();
    config.pipeline_mode = PipelineMode::S2s;
    config.s2s = Some(S2sConfig {
        model: Some("lfm3-audio-2b".into()),
        reply_with: S2sReplyWith::Llm,
        ..Default::default()
    });
    config
}

#[test]
fn session_s2s_llm_runs_six_worker_cascade() {
    let config = llm_s2s_config();
    let engine = FakeS2s::new();
    let ops = engine.ops_log();
    let llm = FakeLlm::new();
    let llm_log = llm.call_log();
    let sink = Arc::new(Mutex::new(CollectingSink::default()));
    let sink_view = Arc::clone(&sink);
    let (tx, rx) = mpsc::channel();
    let report = Session::from_config(&config)
        .with_s2s(engine)
        .with_llm(llm)
        .with_vad(FakeVad::new())
        .with_sink(SharedCollectingSink(sink_view))
        .with_frames(scripted_frames(1, 2, 1))
        .with_events(Some(tx))
        .with_mode(LoopMode::StopAfterTurns(1))
        .run(Cancel::new())
        .expect("llm s2s session");
    assert_eq!(report.tasks_exited, 6);
    let events: Vec<LoopEvent> = rx.try_iter().collect();
    let mut saw_user = false;
    let mut user_index = None;
    for (i, event) in events.iter().enumerate() {
        match event {
            LoopEvent::User { text, .. } if text == FAKE_S2S_TRANSCRIPT => {
                assert!(!saw_user, "duplicate User events with the fake transcript");
                assert!(
                    events[..i]
                        .iter()
                        .all(|e| !matches!(e, LoopEvent::Assistant { .. })),
                    "User must appear before any Assistant event"
                );
                saw_user = true;
                user_index = Some(i);
            }
            LoopEvent::Assistant { .. } if !saw_user => {
                panic!("Assistant event before User transcript");
            }
            _ => {}
        }
    }
    assert!(saw_user, "missing User event with fake ASR transcript");
    let assistant: Vec<(String, bool)> = events
        .iter()
        .skip(user_index.unwrap_or(0))
        .filter_map(|e| match e {
            LoopEvent::Assistant { text, is_last, .. } => Some((text.clone(), *is_last)),
            _ => None,
        })
        .collect();
    assert!(!assistant.is_empty());
    let joined: String = assistant.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(joined, format!("echo:{FAKE_S2S_TRANSCRIPT}"));
    assert_eq!(assistant.iter().filter(|(_, last)| *last).count(), 1);
    assert_eq!(llm_log.lock().unwrap().len(), 1);
    assert_eq!(llm_log.lock().unwrap()[0].user_text, FAKE_S2S_TRANSCRIPT);
    let log = ops.lock().unwrap();
    let transcribe_pos = log
        .iter()
        .position(|op| matches!(op, FakeS2sOp::Transcribe { .. }))
        .expect("transcribe op");
    let speak_pos = log
        .iter()
        .position(|op| matches!(op, FakeS2sOp::Speak { .. }))
        .expect("speak op");
    assert!(transcribe_pos < speak_pos);
    assert!(!log.iter().any(|op| matches!(op, FakeS2sOp::Generate(_))));
    assert_eq!(
        sink.lock()
            .unwrap()
            .chunks
            .iter()
            .filter(|c| c.is_last)
            .count(),
        1
    );
}

#[derive(Clone)]
struct SharedCollectingSink(Arc<Mutex<CollectingSink>>);

impl syllabix_core::AudioSink for SharedCollectingSink {
    fn play(
        &mut self,
        audio: syllabix_core::SynthesizedAudio,
        cancel: &Cancel,
    ) -> syllabix_core::Result<()> {
        self.0.lock().unwrap().play(audio, cancel)
    }

    fn finish_turn(&mut self, turn: TurnId, cancel: &Cancel) -> syllabix_core::Result<()> {
        self.0.lock().unwrap().finish_turn(turn, cancel)
    }

    fn interrupt(&mut self) {
        self.0.lock().unwrap().interrupt();
    }
}

#[test]
fn reply_with_s2s_ops_log_only_generate() {
    let mut config = AgentConfig::v0();
    config.pipeline_mode = PipelineMode::S2s;
    config.s2s = Some(S2sConfig {
        model: Some("lfm3-audio-2b".into()),
        ..Default::default()
    });
    let engine = FakeS2s::new();
    let ops = engine.ops_log();
    Session::from_config(&config)
        .with_s2s(engine)
        .with_vad(FakeVad::new())
        .with_sink(CollectingSink::default())
        .with_frames(scripted_frames(1, 2, 1))
        .with_mode(LoopMode::StopAfterTurns(1))
        .run(Cancel::new())
        .expect("voice s2s");
    let log = ops.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert!(matches!(log[0], FakeS2sOp::Generate(TurnId(0))));
}

#[test]
fn default_language_on_transcript_is_auto() {
    use std::sync::{Arc, Mutex};

    use syllabix_core::s2s::S2sEngine;
    use syllabix_core::s2s_speech::S2sStt;
    use syllabix_core::stt::Stt;
    use syllabix_core::Utterance;

    let shared = Arc::new(Mutex::new(Box::new(FakeS2s::new()) as Box<dyn S2sEngine>));
    let mut stt = S2sStt::new(shared, None, LANGUAGE_AUTO);
    let transcript = stt
        .transcribe(
            &Utterance {
                turn: TurnId(0),
                frames: scripted_frames(1, 2, 0),
            },
            &Cancel::new(),
        )
        .expect("transcribe");
    assert_eq!(transcript.language, syllabix_core::stt::STT_LANGUAGE);
}

#[test]
fn language_and_voice_pass_through_ops_log() {
    let mut config = llm_s2s_config();
    config.s2s = Some(S2sConfig {
        model: Some("lfm3-audio-2b".into()),
        reply_with: S2sReplyWith::Llm,
        language: Some("fr".into()),
        voice: Some("male".into()),
        split_sentences: true,
    });
    let engine = FakeS2s::new();
    let ops = engine.ops_log();
    Session::from_config(&config)
        .with_s2s(engine)
        .with_llm(FakeLlm::new())
        .with_vad(FakeVad::new())
        .with_sink(CollectingSink::default())
        .with_frames(scripted_frames(1, 2, 1))
        .with_mode(LoopMode::StopAfterTurns(1))
        .run(Cancel::new())
        .expect("language voice");
    let log = ops.lock().unwrap();
    assert!(log.iter().any(|op| matches!(
        op,
        FakeS2sOp::Transcribe {
            language: Some(lang),
            ..
        } if lang == "fr"
    )));
    assert!(log.iter().any(|op| matches!(
        op,
        FakeS2sOp::Speak {
            language: Some(lang),
            voice: Some(v),
            ..
        } if lang == "fr" && v == "male"
    )));
}

#[test]
fn online_stt_tts_yaml_still_skips_remote_api_keys() {
    let mut config = llm_s2s_config();
    config.stt = SttProvider::Online;
    config.stt_base_url = Some("https://example.invalid/stt".into());
    config.stt_model_id = "whisper-1".into();
    config.tts = TtsProvider::Online;
    config.tts_base_url = Some("https://example.invalid/tts".into());
    config.tts_model_id = "tts-1".into();
    with_cleared_stt_tts_keys(|| {
        Session::from_config(&config)
            .with_s2s(FakeS2s::new())
            .with_llm(FakeLlm::new())
            .with_vad(FakeVad::new())
            .with_sink(CollectingSink::default())
            .with_frames(scripted_frames(1, 2, 1))
            .with_mode(LoopMode::StopAfterTurns(1))
            .run(Cancel::new())
            .expect("online stt/tts yaml without stt/tts keys");
    });
}

struct SlowSink {
    inner: Arc<Mutex<CollectingSink>>,
    delay: Duration,
}

impl SlowSink {
    fn new(delay: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(CollectingSink::default())),
            delay,
        }
    }
}

impl syllabix_core::AudioSink for SlowSink {
    fn play(
        &mut self,
        audio: syllabix_core::SynthesizedAudio,
        cancel: &Cancel,
    ) -> syllabix_core::Result<()> {
        let deadline = std::time::Instant::now() + self.delay;
        while std::time::Instant::now() < deadline {
            if cancel.is_shutdown() || cancel.is_stale(audio.generation) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(5));
        }
        self.inner.lock().unwrap().play(audio, cancel)
    }

    fn finish_turn(&mut self, turn: TurnId, cancel: &Cancel) -> syllabix_core::Result<()> {
        self.inner.lock().unwrap().finish_turn(turn, cancel)
    }

    fn interrupt(&mut self) {
        self.inner.lock().unwrap().interrupt();
    }
}

#[test]
fn barge_in_cancels_slow_speak_and_second_turn_completes() {
    let config = llm_s2s_config();
    let engine = FakeS2s::with_staged_delay(Duration::from_millis(25), Duration::ZERO);
    let ops = engine.ops_log();
    let (tx, rx) = mpsc::channel();
    let report = Session::from_config(&config)
        .with_s2s(engine)
        .with_llm(FakeLlm::new())
        .with_vad(FakeVad::new())
        .with_sink(SlowSink::new(Duration::from_millis(120)))
        .with_frames(scripted_frames(2, 2, 1))
        .with_events(Some(tx))
        .with_mode(LoopMode::UntilInputEnds)
        .with_barge_in(true)
        .run(Cancel::new())
        .expect("barge llm s2s");
    assert!(report.turns.iter().any(|t| t.id.0 == 1));
    let events: Vec<LoopEvent> = rx.try_iter().collect();
    let user_turns: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            LoopEvent::User { turn, text, .. } if text == FAKE_S2S_TRANSCRIPT => Some(*turn),
            _ => None,
        })
        .collect();
    assert!(
        user_turns.iter().any(|t| t.0 >= 1),
        "second speech should produce a User event"
    );
    assert!(
        events.iter().any(|e| matches!(
            e,
            LoopEvent::Assistant { turn, .. } if turn.0 >= 1
        )),
        "second turn should produce assistant tokens"
    );
    let log = ops.lock().unwrap();
    assert!(
        log.iter()
            .filter(|op| matches!(op, FakeS2sOp::Transcribe { .. }))
            .count()
            >= 2,
        "each turn should invoke transcribe"
    );
    let speaks = log
        .iter()
        .filter(|op| matches!(op, FakeS2sOp::Speak { .. }))
        .count();
    assert!(speaks >= 1, "at least one speak (second turn may finish)");
}

#[test]
fn without_barge_in_extra_speech_during_playback_is_discarded() {
    let config = llm_s2s_config();
    let report = Session::from_config(&config)
        .with_s2s(FakeS2s::with_delay(Duration::from_millis(80)))
        .with_llm(FakeLlm::new())
        .with_vad(FakeVad::new())
        .with_sink(SlowSink::new(Duration::from_millis(120)))
        .with_frames(scripted_frames(2, 2, 1))
        .with_mode(LoopMode::UntilInputEnds)
        .with_barge_in(false)
        .run(Cancel::new())
        .expect("no barge");
    assert_eq!(report.turns.len(), 1);
}
