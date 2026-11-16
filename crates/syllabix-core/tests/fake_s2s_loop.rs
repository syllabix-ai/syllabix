//! Turn-based S2S loop fixtures (no mic, no weights).

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use syllabix_core::config::{AgentConfig, PipelineMode};
use syllabix_core::defaults::BuiltinDefaults;
use syllabix_core::pipeline::LoopMode;
use syllabix_core::testing::{
    run_s2s_loop, scripted_frames, CollectingSink, FakeS2s, FakeVad, S2sStages,
};
use syllabix_core::types::Utterance;
use syllabix_core::{
    AudioSink, Cancel, LoopConfig, Result, RuntimeControls, Session, SynthesizedAudio, TurnId,
};

#[derive(Clone)]
struct SharedSink(Arc<Mutex<CollectingSink>>);

impl Default for SharedSink {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(CollectingSink::default())))
    }
}

impl SharedSink {
    fn chunks(&self) -> Vec<SynthesizedAudio> {
        self.0.lock().expect("sink lock").chunks.clone()
    }

    fn interrupted(&self) -> usize {
        self.0.lock().expect("sink lock").interrupted
    }
}

impl AudioSink for SharedSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        self.0.lock().expect("sink lock").play(audio, cancel)
    }

    fn finish_turn(&mut self, turn: TurnId, cancel: &Cancel) -> Result<()> {
        self.0.lock().expect("sink lock").finish_turn(turn, cancel)
    }

    fn interrupt(&mut self) {
        self.0.lock().expect("sink lock").interrupt();
    }
}

fn speech_pcm(frames: &[syllabix_core::AudioFrame]) -> Vec<i16> {
    let speech: Vec<_> = frames
        .iter()
        .filter(|frame| frame.has_energy())
        .cloned()
        .collect();
    Utterance {
        turn: TurnId(0),
        frames: speech,
    }
    .pcm()
}

struct SlowSink {
    inner: SharedSink,
    delay: Duration,
}

impl SlowSink {
    fn new(delay: Duration) -> Self {
        Self {
            inner: SharedSink::default(),
            delay,
        }
    }

    fn shared(&self) -> SharedSink {
        self.inner.clone()
    }
}

impl AudioSink for SlowSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        let deadline = std::time::Instant::now() + self.delay;
        while std::time::Instant::now() < deadline {
            if cancel.is_shutdown() || cancel.is_stale(audio.generation) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(5));
        }
        self.inner.play(audio, cancel)
    }

    fn finish_turn(&mut self, turn: TurnId, cancel: &Cancel) -> Result<()> {
        self.inner.finish_turn(turn, cancel)
    }

    fn interrupt(&mut self) {
        self.inner.interrupt();
    }
}

fn s2s_config() -> LoopConfig {
    LoopConfig {
        defaults: BuiltinDefaults::v0(),
        mode: LoopMode::StopAfterTurns(1),
        events: None,
        turn_debug: None,
        controls: RuntimeControls::new(false),
    }
}

#[test]
fn one_turn_echo_completes_with_four_workers() {
    let frames = scripted_frames(1, 2, 1);
    let expected = speech_pcm(&frames);
    let sink = SharedSink::default();
    let report = run_s2s_loop(
        s2s_config(),
        S2sStages {
            vad: FakeVad::new(),
            engine: FakeS2s::new(),
            sink: sink.clone(),
        },
        frames,
        Cancel::new(),
    )
    .expect("one turn");
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.tasks_exited, 4);
    assert_eq!(report.turns[0].user_text, "");
    let played = sink.chunks();
    assert_eq!(played.len(), 1);
    assert_eq!(played[0].samples, expected);
}

#[test]
fn session_s2s_yaml_skips_cascade_ml() {
    let mut config = AgentConfig::v0();
    config.pipeline_mode = PipelineMode::S2s;
    config.s2s = Some(syllabix_core::config::S2sConfig::default());
    let frames = scripted_frames(1, 2, 1);
    let expected = speech_pcm(&frames);
    let sink = SharedSink::default();
    let report = Session::from_config(&config)
        .with_s2s(FakeS2s::new())
        .with_vad(FakeVad::new())
        .with_sink(sink.clone())
        .with_frames(frames)
        .with_mode(LoopMode::StopAfterTurns(1))
        .run(Cancel::new())
        .expect("session s2s");
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.tasks_exited, 4);
    let played = sink.chunks();
    assert_eq!(played.len(), 1);
    assert_eq!(played[0].samples, expected);
}

#[test]
fn cascade_session_still_uses_six_worker_loop() {
    use syllabix_core::testing::{FakeLlm, FakeStt, FakeTts};

    let report = Session::v0()
        .with_vad(FakeVad::new())
        .with_stt(FakeStt)
        .with_llm(FakeLlm::new())
        .with_tts(FakeTts)
        .with_sink(CollectingSink::default())
        .with_frames(scripted_frames(1, 2, 1))
        .with_mode(LoopMode::StopAfterTurns(1))
        .run(Cancel::new())
        .expect("cascade session");
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.tasks_exited, 6);
}

fn barge_s2s(barge_in: bool) -> (syllabix_core::LoopReport, SharedSink, Vec<TurnId>) {
    let slow = SlowSink::new(Duration::from_millis(30));
    let sink = slow.shared();
    let engine = FakeS2s::with_delay(Duration::from_millis(25));
    let call_log = engine.call_log();
    let report = run_s2s_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(barge_in),
        },
        S2sStages {
            vad: FakeVad::new(),
            engine,
            sink: slow,
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("barge s2s");
    let calls = call_log.lock().expect("call log").clone();
    (report, sink, calls)
}

#[test]
fn without_barge_in_speech_during_playback_is_discarded() {
    let (report, _, calls) = barge_s2s(false);
    assert_eq!(report.turns.len(), 1);
    assert_eq!(calls.len(), 1);
}

#[test]
fn barge_in_cancels_playback_and_completes_second_turn() {
    let (report, sink, calls) = barge_s2s(true);
    assert!(sink.interrupted() >= 1);
    assert!(
        report.turns.iter().any(|t| t.id.0 == 1),
        "second turn must complete"
    );
    assert!(
        calls.len() >= 2,
        "first generate must start before barge-in cancels it, got {calls:?}"
    );
    assert_eq!(calls[0], TurnId(0));
    assert_eq!(calls.last().copied(), Some(TurnId(1)));
}

#[test]
fn s2s_yaml_without_engine_still_not_supported() {
    let mut config = AgentConfig::v0();
    config.pipeline_mode = PipelineMode::S2s;
    config.s2s = Some(syllabix_core::config::S2sConfig::default());
    let err = Session::from_config(&config)
        .with_vad(FakeVad::new())
        .with_sink(CollectingSink::default())
        .with_frames(scripted_frames(1, 2, 1))
        .run(Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("not supported yet"), "{err}");
}

#[cfg(all(feature = "ggml", feature = "device-audio"))]
mod aec_self_reply {
    use std::collections::VecDeque;
    use std::path::Path;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use syllabix_core::audio::{
        f32_to_i16, i16_to_f32, read_wav, record_fixture_to_frames, EchoController, EchoReference,
        FrameSplitter, PcmFormat, SampleRing, AEC_FRAME_SAMPLES,
    };
    use syllabix_core::models::{HttpFetcher, StderrProgress};
    use syllabix_core::testing::run_s2s_loop_captured;
    use syllabix_core::types::FRAME_SAMPLES;
    use syllabix_core::vad::SileroVad;
    use syllabix_core::{AudioCapture, Cancel, Error, LoopConfig, ModelCache, Result};

    const DELAY: usize = AEC_FRAME_SAMPLES * 8;
    const LEAK: f32 = 0.55;

    fn load_speech_frames() -> Vec<syllabix_core::AudioFrame> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vad/speech.wav");
        let wav = read_wav(
            std::fs::File::open(&path)
                .unwrap_or_else(|err| panic!("open {}: {err}", path.display())),
        )
        .expect("speech.wav");
        record_fixture_to_frames(&wav).expect("speech frames")
    }

    struct SpeakerCallbackSink {
        ring: Arc<SampleRing>,
    }

    impl AudioSink for SpeakerCallbackSink {
        fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let pcm = i16_to_f32(&audio.samples);
            let mut offset = 0;
            while offset < pcm.len() {
                if cancel.is_shutdown() {
                    return Err(Error::Cancelled);
                }
                let end = (offset + AEC_FRAME_SAMPLES).min(pcm.len());
                let _ = self.ring.try_push_slice(&pcm[offset..end]);
                offset = end;
                thread::sleep(Duration::from_millis(2));
            }
            Ok(())
        }

        fn interrupt(&mut self) {}
    }

    struct SelfReplyAecCapture {
        echo: EchoController,
        ring: Arc<SampleRing>,
        speech: std::vec::IntoIter<syllabix_core::AudioFrame>,
        tail_blocks: usize,
        history: VecDeque<f32>,
        splitter: FrameSplitter,
        pending: VecDeque<syllabix_core::AudioFrame>,
    }

    impl SelfReplyAecCapture {
        fn ingest_cleaned(&mut self, capture: &[f32]) -> Result<()> {
            let clean = self.echo.process_capture(capture)?;
            self.pending
                .extend(self.splitter.push(&f32_to_i16(&clean))?);
            Ok(())
        }
    }

    impl AudioCapture for SelfReplyAecCapture {
        fn name(&self) -> &'static str {
            "self-reply-aec"
        }

        fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<syllabix_core::AudioFrame>> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            loop {
                if let Some(frame) = self.pending.pop_front() {
                    return Ok(Some(frame));
                }
                if let Some(frame) = self.speech.next() {
                    thread::sleep(Duration::from_millis(25));
                    self.ingest_cleaned(&i16_to_f32(&frame.samples))?;
                    continue;
                }
                if self.tail_blocks == 0 {
                    self.pending.extend(self.splitter.flush()?);
                    return Ok(self.pending.pop_front());
                }
                self.tail_blocks -= 1;
                thread::sleep(Duration::from_millis(25));
                let mut render = vec![0.0f32; AEC_FRAME_SAMPLES];
                self.ring.try_pop_slice(&mut render);
                for sample in render {
                    self.history.push_back(sample * LEAK);
                }
                while self.history.len() < DELAY + AEC_FRAME_SAMPLES {
                    self.history.push_back(0.0);
                }
                let capture: Vec<f32> = self.history.drain(..AEC_FRAME_SAMPLES).collect();
                self.ingest_cleaned(&capture)?;
            }
        }
    }

    #[test]
    fn self_reply_leak_does_not_commit_second_turn() {
        static SILERO_CACHE_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
            std::sync::OnceLock::new();
        let _guard = SILERO_CACHE_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .expect("Silero cache lock");

        let ring = Arc::new(SampleRing::new(AEC_FRAME_SAMPLES * 8));
        let reference = EchoReference::new(Arc::clone(&ring), PcmFormat::v0());
        let mut echo = EchoController::new(reference).expect("echo");
        echo.start_render_pump();

        let engine = FakeS2s::new();
        let call_log = engine.call_log();

        let vad = SileroVad::from_cache(
            &ModelCache::v0(),
            &HttpFetcher,
            &mut StderrProgress::new(),
            &Cancel::new(),
        )
        .expect("Silero");

        let speech_frames = load_speech_frames();
        let tail_blocks = speech_frames.len() / FRAME_SAMPLES + 120;

        let report = run_s2s_loop_captured(
            LoopConfig {
                defaults: BuiltinDefaults::v0(),
                mode: LoopMode::UntilInputEnds,
                events: None,
                turn_debug: None,
                controls: RuntimeControls::new(true),
            },
            S2sStages {
                vad,
                engine,
                sink: SpeakerCallbackSink {
                    ring: Arc::clone(&ring),
                },
            },
            SelfReplyAecCapture {
                echo,
                ring,
                speech: speech_frames.into_iter(),
                tail_blocks,
                history: VecDeque::from(vec![0.0f32; DELAY]),
                splitter: FrameSplitter::new(),
                pending: VecDeque::new(),
            },
            Cancel::new(),
        )
        .expect("self reply");

        assert_eq!(report.turns.len(), 1);
        assert_eq!(call_log.lock().expect("call log").len(), 1);
    }
}
