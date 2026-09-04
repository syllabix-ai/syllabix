//! Merge gate for PR 2: in-memory fake loop, order, bounds, shutdown, cancel.

use std::thread;
use std::time::Duration;

use syllabix_core::{
    audio::{
        device_ring_capacity_samples, i16_to_f32, EchoCalibration, EchoController, EchoReference,
        SampleRing, AEC_FRAME_SAMPLES,
    },
    audio::{read_wav, PcmFormat},
    run_loop, run_loop_captured, scripted_frames, AudioCapture, AudioSink, BuiltinDefaults, Cancel,
    CollectingSink, Error, FailOnceLlm, FailOnceStt, FailOnceTts, FakeLlm, FakeStt, FakeTts,
    FakeVad, LoopConfig, LoopMode, PipelineStages, PlaybackWatch, QueueCaps, Result,
    RuntimeControls, ScriptedStt, SynthesizedAudio, TimelineAnchor, TokenChunk, Transcript, Tts,
    TurnDebug, TurnId, Vad, VadEvent, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
};

fn run_with(
    frames: Vec<syllabix_core::AudioFrame>,
    llm: FakeLlm,
    cancel: Cancel,
    mode: LoopMode,
) -> syllabix_core::LoopReport {
    run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode,
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm,
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        frames,
        cancel,
    )
    .expect("fake loop")
}

#[test]
fn thirty_turns_preserve_order_and_queue_bounds() {
    let n = 30;
    let log_holder = FakeLlm::new();
    let calls = log_holder.call_log();
    let report = run_with(
        scripted_frames(n, 2, 1),
        log_holder,
        Cancel::new(),
        LoopMode::UntilInputEnds,
    );

    assert_eq!(report.turns.len(), n, "every scripted turn must complete");
    assert_eq!(report.tasks_exited, 6);
    assert_eq!(report.tasks_still_running, 0);
    assert!(
        report.queues.within_capacity(),
        "queue occupancy exceeded a bound: {:?}",
        report.queues
    );
    for occupancy in report.queues.all() {
        assert_eq!(
            occupancy.current, 0,
            "{} still held {} items after shutdown",
            occupancy.name, occupancy.current
        );
        assert!(
            occupancy.sent > 0,
            "{} should have carried traffic",
            occupancy.name
        );
        assert_eq!(occupancy.sent, occupancy.received);
        assert!(occupancy.high_water > 0);
        assert!(occupancy.high_water <= occupancy.capacity);
    }

    for (i, turn) in report.turns.iter().enumerate() {
        assert_eq!(turn.id, TurnId(i as u64));
        let expected_user = format!("turn-{i:03}");
        assert_eq!(turn.user_text, expected_user);
        assert_eq!(turn.assistant_text, format!("echo:{expected_user}"));
        assert!(turn.token_count > 0);
        assert_eq!(turn.token_count, turn.audio_chunks);
        if i > 0 {
            assert!(turn.id > report.turns[i - 1].id);
        }
    }

    let calls = calls.lock().expect("call log");
    assert_eq!(calls.len(), n);
    for (i, call) in calls.iter().enumerate() {
        assert_eq!(call.history_len, i);
        assert_eq!(call.user_text, format!("turn-{i:03}"));
        let expected_history: Vec<String> = (0..i).map(|j| format!("turn-{j:03}")).collect();
        assert_eq!(call.history_user_texts, expected_history);
    }

    let caps = QueueCaps::v0();
    assert_eq!(report.queues.frames.capacity, caps.frames);
    assert_eq!(report.queues.utterances.capacity, caps.utterances);
    assert_eq!(report.queues.transcripts.capacity, caps.transcripts);
    assert_eq!(report.queues.tokens.capacity, caps.tokens);
    assert_eq!(report.queues.audio.capacity, caps.audio);
}

#[test]
fn shutdown_joins_every_worker() {
    let cancel = Cancel::new();
    let watcher = cancel.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(15));
        watcher.shutdown();
    });

    let report = run_with(
        scripted_frames(40, 2, 1),
        FakeLlm::with_delay(Duration::from_millis(20)),
        cancel,
        LoopMode::UntilInputEnds,
    );

    assert_eq!(report.tasks_exited, 6);
    assert_eq!(report.tasks_still_running, 0);
    assert!(
        report.queues.within_capacity(),
        "queue occupancy exceeded a bound: {:?}",
        report.queues
    );
    assert!(report.turns.len() < 40);
}

#[test]
fn generation_cancel_drops_assistant_audio_and_keeps_the_next_user_turn() {
    let llm = FakeLlm::with_delay(Duration::from_millis(20));
    let calls = llm.call_log();
    let cancel = Cancel::new();
    let watcher = cancel.clone();
    thread::spawn(move || loop {
        if calls.lock().map(|c| c.len()).unwrap_or(0) >= 1 {
            thread::sleep(Duration::from_millis(30));
            watcher.cancel_generation();
            break;
        }
        thread::sleep(Duration::from_millis(1));
    });

    let report = run_with(
        scripted_frames(2, 2, 1),
        llm,
        cancel,
        LoopMode::UntilInputEnds,
    );

    assert_eq!(report.tasks_exited, 6);
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.queues.within_capacity());

    let completed_ids: Vec<u64> = report.turns.iter().map(|t| t.id.0).collect();
    assert!(
        completed_ids.contains(&1),
        "incoming second user turn must still complete after generation cancel, got {completed_ids:?}"
    );

    if let Some(first) = report.turns.iter().find(|t| t.id.0 == 0) {
        assert_eq!(first.user_text, "turn-000");
        assert_ne!(
            first.assistant_text, "echo:turn-000",
            "cancelled generation must not play a full first reply"
        );
    }

    let second = report
        .turns
        .iter()
        .find(|t| t.id.0 == 1)
        .expect("second turn");
    assert_eq!(second.user_text, "turn-001");
    assert_eq!(second.assistant_text, "echo:turn-001");
}

#[test]
fn stop_after_turns_shuts_the_loop_down() {
    let report = run_with(
        scripted_frames(8, 2, 1),
        FakeLlm::new(),
        Cancel::new(),
        LoopMode::StopAfterTurns(3),
    );
    assert_eq!(report.turns.len(), 3);
    assert_eq!(report.turns[0].id, TurnId(0));
    assert_eq!(report.turns[2].id, TurnId(2));
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.cancelled);
    assert!(report.queues.within_capacity());
}

#[test]
fn thirty_turns_config_helper_matches_launch_defaults() {
    let cfg = LoopConfig::thirty_turns();
    assert_eq!(cfg.mode, LoopMode::StopAfterTurns(30));
    assert_eq!(cfg.defaults.llm_model, "llama-3.2-1b");
}

#[test]
fn six_turns_config_helper_is_the_native_gate() {
    let cfg = LoopConfig::six_turns();
    assert_eq!(cfg.mode, LoopMode::StopAfterTurns(6));
}

#[test]
fn provider_error_in_stt_skips_the_turn_and_keeps_going() {
    let report = run_loop(
        LoopConfig::default(),
        PipelineStages {
            vad: FakeVad::new(),
            stt: FailOnceStt::default(),
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("recoverable stt");
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.skipped_turns >= 1);
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].user_text, "turn-001");
    assert!(report.queues.within_capacity());
}

#[test]
fn provider_error_in_llm_skips_the_turn_and_keeps_going() {
    let report = run_loop(
        LoopConfig::default(),
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FailOnceLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("recoverable llm");
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.skipped_turns >= 1);
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].user_text, "turn-001");
    assert_eq!(report.turns[0].assistant_text, "echo:turn-001");
}

#[test]
fn provider_error_in_tts_skips_the_turn_and_keeps_going() {
    let report = run_loop(
        LoopConfig::default(),
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: FailOnceTts::default(),
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("recoverable tts");
    assert_eq!(report.tasks_still_running, 0);
    assert!(report.skipped_turns >= 1);
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].id, TurnId(1));
}

#[test]
fn empty_and_whitespace_stt_skip_llm_and_keep_later_turns() {
    let llm = FakeLlm::new();
    let calls = llm.call_log();
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: ScriptedStt::new(["", " \n", "hello"]),
            llm,
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(3, 2, 1),
        Cancel::new(),
    )
    .expect("blank then real");
    assert_eq!(report.skipped_turns, 2);
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].user_text, "hello");
    assert_eq!(report.turns[0].assistant_text, "echo:hello");
    let log = calls.lock().expect("llm log");
    assert_eq!(log.len(), 1);
    assert_eq!(log[0].user_text, "hello");
    assert!(log[0].history_user_texts.is_empty());
}

fn unique_debug_dir() -> std::path::PathBuf {
    // pid+nanos collided once under parallel load (two same-process tests in
    // the same clock tick) and one test's WAVs landed in the other's
    // asserted-empty dir. The counter makes collisions impossible.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    std::env::temp_dir().join(format!(
        "syllabix-fake-turn-debug-{}-{}-{seq}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn default_loop_writes_no_turn_debug_files() {
    let dir = unique_debug_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let report = run_with(
        scripted_frames(1, 2, 1),
        FakeLlm::new(),
        Cancel::new(),
        LoopMode::UntilInputEnds,
    );
    assert_eq!(report.turns.len(), 1);
    assert!(std::fs::read_dir(&dir).unwrap().next().is_none());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn turn_debug_one_turn_writes_wavs_and_does_not_change_reply() {
    let dir = unique_debug_dir();
    let debug = TurnDebug::open(&dir).unwrap();
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: Some(debug),
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        Cancel::new(),
    )
    .expect("debug loop");
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].user_text, "turn-000");
    assert_eq!(report.turns[0].assistant_text, "echo:turn-000");

    let turn_dir = dir.join("turn-000");
    for name in [
        "capture.wav",
        "clean.wav",
        "utterance.wav",
        "tts.wav",
        "turn.json",
    ] {
        assert!(turn_dir.join(name).is_file(), "missing {name}");
    }
    let capture = read_wav(std::io::Cursor::new(
        std::fs::read(turn_dir.join("capture.wav")).unwrap(),
    ))
    .unwrap();
    let clean = read_wav(std::io::Cursor::new(
        std::fs::read(turn_dir.join("clean.wav")).unwrap(),
    ))
    .unwrap();
    let utterance = read_wav(std::io::Cursor::new(
        std::fs::read(turn_dir.join("utterance.wav")).unwrap(),
    ))
    .unwrap();
    let tts = read_wav(std::io::Cursor::new(
        std::fs::read(turn_dir.join("tts.wav")).unwrap(),
    ))
    .unwrap();
    assert_eq!(capture.format, PcmFormat::v0());
    assert_eq!(clean.samples, capture.samples);
    assert_eq!(capture.samples.len(), FRAME_SAMPLES * 3);
    assert_eq!(utterance.samples.len(), FRAME_SAMPLES * 2);
    assert!(utterance.samples.iter().all(|s| *s == 1));
    let expected_tts: Vec<i16> = "echo:turn-000".bytes().map(i16::from).collect();
    assert_eq!(tts.samples, expected_tts);
    assert_eq!(tts.format.sample_rate_hz, DEFAULT_SAMPLE_RATE_HZ);
    let sidecar = std::fs::read_to_string(turn_dir.join("turn.json")).unwrap();
    assert!(sidecar.contains("\"outcome\": \"completed\""));
    assert!(sidecar.contains("\"stt_text\": \"turn-000\""));
    assert!(sidecar.contains("\"llm_text\": \"echo:turn-000\""));
    assert!(sidecar.contains("\"tts_speak_text\": \"echo:turn-000\""));
    assert!(sidecar.contains("\"capture_frames\": 3"));
    assert!(sidecar.contains("\"utterance_frames\": 2"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn turn_debug_skipped_stt_still_writes() {
    let dir = unique_debug_dir();
    let debug = TurnDebug::open(&dir).unwrap();
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: Some(debug),
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FailOnceStt::default(),
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("skip debug");
    assert!(report.skipped_turns >= 1);
    let skipped = std::fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
    assert!(skipped.contains("skipped"));
    let completed = std::fs::read_to_string(dir.join("turn-001").join("turn.json")).unwrap();
    assert!(completed.contains("completed"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn turn_debug_empty_stt_is_skipped_without_tts() {
    let dir = unique_debug_dir();
    let debug = TurnDebug::open(&dir).unwrap();
    let llm = FakeLlm::new();
    let calls = llm.call_log();
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: Some(debug),
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: ScriptedStt::new([""]),
            llm,
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        Cancel::new(),
    )
    .expect("empty stt debug");
    assert!(report.turns.is_empty());
    assert!(calls.lock().expect("llm log").is_empty());
    let sidecar = std::fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
    assert!(sidecar.contains("\"outcome\": \"skipped\""));
    assert!(sidecar.contains("\"stt_text\": \"\""));
    assert!(sidecar.contains("\"llm_text\": \"\""));
    assert!(sidecar.contains("\"tts_speak_text\": \"\""));
    let tts = read_wav(std::io::Cursor::new(
        std::fs::read(dir.join("turn-000").join("tts.wav")).unwrap(),
    ))
    .unwrap();
    assert!(tts.samples.is_empty());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn turn_debug_cancel_writes_partial_turn() {
    let dir = unique_debug_dir();
    let debug = TurnDebug::open(&dir).unwrap();
    let llm = FakeLlm::with_delay(Duration::from_millis(20));
    let calls = llm.call_log();
    let cancel = Cancel::new();
    let watcher = cancel.clone();
    thread::spawn(move || loop {
        if calls.lock().map(|c| c.len()).unwrap_or(0) >= 1 {
            thread::sleep(Duration::from_millis(15));
            watcher.cancel_generation();
            break;
        }
        thread::sleep(Duration::from_millis(1));
    });
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: Some(debug),
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm,
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        cancel,
    )
    .expect("cancelled debug");
    assert!(dir.join("turn-000").join("turn.json").is_file());
    let first = std::fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
    assert!(
        first.contains("cancelled") || first.contains("completed"),
        "{first}"
    );
    assert!(
        report.turns.iter().any(|t| t.id.0 == 1),
        "second turn must survive generation cancel"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

struct PacedCapture {
    frames: std::vec::IntoIter<syllabix_core::AudioFrame>,
    delay: Duration,
}

impl AudioCapture for PacedCapture {
    fn name(&self) -> &'static str {
        "paced"
    }

    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<syllabix_core::AudioFrame>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let Some(frame) = self.frames.next() else {
            return Ok(None);
        };
        if !self.delay.is_zero() {
            thread::sleep(self.delay);
        }
        Ok(Some(frame))
    }
}

struct SlowSink {
    delay: Duration,
    playing: std::sync::Arc<std::sync::atomic::AtomicBool>,
    inner: CollectingSink,
}

impl AudioSink for SlowSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        self.playing
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let deadline = std::time::Instant::now() + self.delay;
        while std::time::Instant::now() < deadline {
            if cancel.is_shutdown() {
                self.playing
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                return Err(Error::Cancelled);
            }
            if cancel.is_stale(audio.generation) {
                self.playing
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                self.inner.interrupt();
                return Ok(());
            }
            thread::sleep(Duration::from_millis(5));
        }
        let result = self.inner.play(audio, cancel);
        self.playing
            .store(false, std::sync::atomic::Ordering::SeqCst);
        result
    }

    fn interrupt(&mut self) {
        self.playing
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.inner.interrupt();
    }
}

fn barge_loop(barge_in: bool, llm: FakeLlm) -> syllabix_core::LoopReport {
    run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(barge_in),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm,
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("barge loop")
}

#[test]
fn without_barge_in_speech_during_tts_is_discarded() {
    let report = barge_loop(false, FakeLlm::with_delay(Duration::from_millis(8)));
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].assistant_text, "echo:turn-000");
}

#[test]
fn barge_in_cancels_playback_on_speech_start_and_keeps_the_new_turn() {
    let report = barge_loop(true, FakeLlm::with_delay(Duration::from_millis(25)));
    assert!(
        report.turns.iter().any(|t| t.id.0 == 1),
        "incoming user turn must complete after barge-in, got {:?}",
        report.turns.iter().map(|t| t.id.0).collect::<Vec<_>>()
    );
    if let Some(first) = report.turns.iter().find(|t| t.id.0 == 0) {
        assert_ne!(first.assistant_text, "echo:turn-000");
    }
    let second = report
        .turns
        .iter()
        .find(|t| t.id.0 == 1)
        .expect("second turn");
    assert_eq!(second.user_text, "turn-001");
    assert_eq!(second.assistant_text, "echo:turn-001");
}

#[test]
fn barge_in_turn_debug_writes_interrupted_turn() {
    let dir = unique_debug_dir();
    let debug = TurnDebug::open(&dir).unwrap();
    let report = run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: Some(debug),
            controls: RuntimeControls::new(true),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::with_delay(Duration::from_millis(25)),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("barge debug");
    let first = std::fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
    assert!(
        first.contains("cancelled"),
        "interrupted turn must dump as cancelled: {first}"
    );
    assert!(dir.join("turn-000").join("tts.wav").is_file());
    assert!(
        report.turns.iter().any(|t| t.id.0 == 1),
        "new user utterance must complete"
    );
    let second = std::fs::read_to_string(dir.join("turn-001").join("turn.json")).unwrap();
    assert!(second.contains("completed"), "{second}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn diagnostics_sidecar_carries_the_full_turn_timeline() {
    let dir = unique_debug_dir();
    let debug = TurnDebug::open(&dir).unwrap();
    let watch = PlaybackWatch::new(debug.clone());
    run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: Some(debug.clone()),
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::with_playback_watch(Some(watch)),
        },
        scripted_frames(1, 2, 1),
        Cancel::new(),
    )
    .expect("timeline loop");

    // Sidecar: every anchor present, epoch zeroed, stage order preserved.
    let sidecar = std::fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
    for anchor in [
        "speech_start_ms",
        "speech_end_ms",
        "stt_queued_ms",
        "stt_done_ms",
        "llm_start_ms",
        "llm_first_token_ms",
        "llm_last_token_ms",
        "tts_first_pcm_ms",
        "tts_last_pcm_ms",
        "playback_first_ms",
        "playback_done_ms",
    ] {
        assert!(
            !sidecar.contains(&format!("\"{anchor}\": null")),
            "{anchor} must be captured for a completed turn: {sidecar}"
        );
    }
    assert!(sidecar.contains("\"speech_start_ms\": 0"), "{sidecar}");
    let value_of = |key: &str| -> u128 {
        let marker = format!("\"{key}\": ");
        let start = sidecar
            .find(&marker)
            .unwrap_or_else(|| panic!("{key} in sidecar"))
            + marker.len();
        let rest = &sidecar[start..];
        let end = rest.find([',', '\n']).expect("number ends");
        rest[..end].trim().parse().expect("millis value")
    };
    assert!(value_of("speech_end_ms") <= value_of("stt_queued_ms"));
    assert!(value_of("stt_done_ms") <= value_of("llm_start_ms"));
    assert!(value_of("llm_start_ms") <= value_of("llm_first_token_ms"));
    assert!(value_of("llm_first_token_ms") <= value_of("tts_first_pcm_ms"));
    assert!(value_of("tts_first_pcm_ms") <= value_of("playback_first_ms"));
    assert!(value_of("playback_first_ms") <= value_of("playback_done_ms"));

    // Recorder accessor agrees with the rendered file.
    let anchored = debug.anchored(TurnId(0));
    assert_eq!(anchored[0].0, TimelineAnchor::SpeechStart);
    assert_eq!(
        anchored.last().map(|(a, _)| *a),
        Some(TimelineAnchor::PlaybackDone)
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn interrupted_barge_in_turn_has_a_partial_timeline_without_drain() {
    let dir = unique_debug_dir();
    let debug = TurnDebug::open(&dir).unwrap();
    run_loop(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: Some(debug),
            controls: RuntimeControls::new(true),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::with_delay(Duration::from_millis(25)),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(2, 2, 1),
        Cancel::new(),
    )
    .expect("barge timeline");
    let first = std::fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
    assert!(first.contains("\"outcome\": \"cancelled\""), "{first}");
    // The turn was cancelled mid-generation: no LLM/TTS/playback end anchors.
    assert!(first.contains("\"llm_last_token_ms\": null"), "{first}");
    assert!(first.contains("\"playback_done_ms\": null"), "{first}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn vad_emits_speech_start_during_tts_only_with_barge_in() {
    fn run(barge_in: bool) -> (bool, syllabix_core::LoopReport) {
        let playing = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let overlap = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let overlap_flag = std::sync::Arc::clone(&overlap);
        let playing_flag = std::sync::Arc::clone(&playing);
        struct OverlapVad {
            inner: FakeVad,
            playing: std::sync::Arc<std::sync::atomic::AtomicBool>,
            overlap: std::sync::Arc<std::sync::atomic::AtomicBool>,
        }
        impl Vad for OverlapVad {
            fn name(&self) -> &'static str {
                self.inner.name()
            }
            fn push_frame(&mut self, frame: syllabix_core::AudioFrame) -> Result<Vec<VadEvent>> {
                let events = self.inner.push_frame(frame)?;
                if events
                    .iter()
                    .any(|e| matches!(e, VadEvent::SpeechStart { .. }))
                    && self.playing.load(std::sync::atomic::Ordering::SeqCst)
                {
                    self.overlap
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                }
                Ok(events)
            }
            fn flush(&mut self) -> Result<Vec<VadEvent>> {
                self.inner.flush()
            }
        }
        let report = run_loop_captured(
            LoopConfig {
                defaults: BuiltinDefaults::v0(),
                mode: LoopMode::UntilInputEnds,
                events: None,
                turn_debug: None,
                controls: RuntimeControls::new(barge_in),
            },
            PipelineStages {
                vad: OverlapVad {
                    inner: FakeVad::new(),
                    playing: playing_flag,
                    overlap: overlap_flag,
                },
                stt: FakeStt,
                llm: FakeLlm::new(),
                tts: FakeTts,
                sink: SlowSink {
                    delay: Duration::from_millis(40),
                    playing,
                    inner: CollectingSink::default(),
                },
            },
            PacedCapture {
                frames: scripted_frames(2, 2, 1).into_iter(),
                delay: Duration::from_millis(8),
            },
            Cancel::new(),
        )
        .expect("paced barge");
        (overlap.load(std::sync::atomic::Ordering::SeqCst), report)
    }

    let (overlap_off, report_off) = run(false);
    assert!(
        !overlap_off,
        "default run must not emit SpeechStart while TTS is playing"
    );
    assert_eq!(report_off.turns.len(), 1);

    let (overlap_on, report_on) = run(true);
    assert!(
        overlap_on,
        "--barge-in must allow SpeechStart while TTS is playing"
    );
    assert!(report_on.turns.iter().any(|t| t.id.0 == 1));
}

/// Default mode still captures audio so AEC can run, but its VAD input must be
/// a discard window from SpeechEnd until the speaker has drained. The one-slot
/// frame queue makes the capture/sink handshake deterministic: by the time
/// the sink releases playback, the synthetic assistant-speech frame has been
/// consumed by the paused VAD worker.
#[test]
fn default_mode_never_transcribes_assistant_audio_after_playback_drains() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    struct GatedCapture {
        phase: u8,
        seq: u64,
        playback_started: Arc<AtomicBool>,
        echo_enqueued: Arc<AtomicBool>,
    }

    impl GatedCapture {
        fn frame(&mut self, energy: i16) -> syllabix_core::AudioFrame {
            let frame = syllabix_core::AudioFrame::new(
                self.seq,
                DEFAULT_SAMPLE_RATE_HZ,
                DEFAULT_CHANNELS,
                vec![energy; FRAME_SAMPLES],
            )
            .expect("fixture frame");
            self.seq += 1;
            frame
        }

        fn wait_for(flag: &AtomicBool, label: &str) {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !flag.load(Ordering::SeqCst) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "timed out waiting for {label}"
                );
                thread::yield_now();
            }
        }
    }

    impl AudioCapture for GatedCapture {
        fn name(&self) -> &'static str {
            "gated-capture"
        }

        fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<syllabix_core::AudioFrame>> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let frame = match self.phase {
                0 => self.frame(1_000), // Initial user speech.
                1 => self.frame(0),
                2 => {
                    Self::wait_for(&self.playback_started, "assistant playback");
                    self.frame(2_000) // Synthetic speaker echo while VAD is paused.
                }
                3 => self.frame(0),
                4 => {
                    // Returning here proves the one-slot queue admitted the
                    // echo speech and its trailing silence to the paused VAD.
                    self.echo_enqueued.store(true, Ordering::SeqCst);
                    return Ok(None);
                }
                _ => return Ok(None),
            };
            self.phase += 1;
            Ok(Some(frame))
        }
    }

    struct GatedSink {
        playback_started: Arc<AtomicBool>,
        echo_enqueued: Arc<AtomicBool>,
    }

    impl AudioSink for GatedSink {
        fn play(&mut self, _audio: SynthesizedAudio, _cancel: &Cancel) -> Result<()> {
            self.playback_started.store(true, Ordering::SeqCst);
            GatedCapture::wait_for(&self.echo_enqueued, "suppressed echo frames");
            Ok(())
        }
    }

    struct LabelStt;

    impl syllabix_core::Stt for LabelStt {
        fn name(&self) -> &'static str {
            "label-stt"
        }

        fn transcribe(
            &mut self,
            utterance: &syllabix_core::Utterance,
            _cancel: &Cancel,
        ) -> Result<Transcript> {
            let energy = utterance
                .frames
                .iter()
                .flat_map(|frame| frame.samples.iter())
                .copied()
                .max()
                .unwrap_or_default();
            let text = match energy {
                2_000 => "assistant playback",
                _ => "initial user",
            };
            Ok(Transcript {
                turn: utterance.turn,
                text: text.into(),
                language: "en".into(),
            })
        }
    }

    let playback_started = Arc::new(AtomicBool::new(false));
    let echo_enqueued = Arc::new(AtomicBool::new(false));
    let mut defaults = BuiltinDefaults::v0();
    defaults.queues.frames = 1;
    let report = run_loop_captured(
        LoopConfig {
            defaults,
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: LabelStt,
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: GatedSink {
                playback_started: Arc::clone(&playback_started),
                echo_enqueued: Arc::clone(&echo_enqueued),
            },
        },
        GatedCapture {
            phase: 0,
            seq: 0,
            playback_started: Arc::clone(&playback_started),
            echo_enqueued: Arc::clone(&echo_enqueued),
        },
        Cancel::new(),
    )
    .expect("default-mode loop");

    assert_eq!(
        report.turns.len(),
        1,
        "only the initial user turn may complete"
    );
    assert_eq!(report.turns[0].user_text, "initial user");
    assert!(
        report
            .turns
            .iter()
            .all(|turn| turn.user_text != "assistant playback"),
        "speaker echo must never be replayed into STT: {:?}",
        report.turns
    );
}

struct HoldUntilStaleSink {
    started: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl AudioSink for HoldUntilStaleSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        let first = !self.started.swap(true, std::sync::atomic::Ordering::SeqCst);
        if !first {
            return Ok(());
        }
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if cancel.is_stale(audio.generation) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

#[test]
fn barge_in_stops_a_blocked_previous_tts_play() {
    let started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    struct GateCapture {
        frames: std::vec::IntoIter<syllabix_core::AudioFrame>,
        index: usize,
        gate_after: usize,
        started: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }
    impl AudioCapture for GateCapture {
        fn name(&self) -> &'static str {
            "gate"
        }
        fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<syllabix_core::AudioFrame>> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if self.index == self.gate_after {
                let wait_started = std::time::Instant::now();
                while !self.started.load(std::sync::atomic::Ordering::SeqCst) {
                    if cancel.is_shutdown() {
                        return Err(Error::Cancelled);
                    }
                    assert!(
                        wait_started.elapsed() < Duration::from_secs(2),
                        "TTS play never started"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
            }
            self.index += 1;
            Ok(self.frames.next())
        }
    }
    let report = run_loop_captured(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(true),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: HoldUntilStaleSink {
                started: std::sync::Arc::clone(&started),
            },
        },
        GateCapture {
            frames: scripted_frames(2, 2, 1).into_iter(),
            index: 0,
            gate_after: 3,
            started: std::sync::Arc::clone(&started),
        },
        Cancel::new(),
    )
    .expect("blocked tts barge");
    assert!(
        started.load(std::sync::atomic::Ordering::SeqCst),
        "first TTS chunk must have started playing"
    );
    assert!(
        report.turns.iter().any(|t| t.id.0 == 1),
        "next user turn must still complete, got {:?}",
        report.turns.iter().map(|t| t.id.0).collect::<Vec<_>>()
    );
}

/// Live mic after SpeechEnd: keep producing frames so the VAD queue can fill
/// while default `run` pauses VAD for the assistant reply.
struct LiveMicWithAec {
    echo: EchoController,
    seq: u64,
    speech_left: u32,
    calibration: std::sync::Arc<std::sync::Mutex<EchoCalibration>>,
}

impl AudioCapture for LiveMicWithAec {
    fn name(&self) -> &'static str {
        "aec-mic"
    }

    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<syllabix_core::AudioFrame>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let energy = if self.speech_left > 0 {
            self.speech_left -= 1;
            8_000
        } else {
            0
        };
        let frame = syllabix_core::AudioFrame::new(
            self.seq,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CHANNELS,
            vec![energy; FRAME_SAMPLES],
        )?;
        self.seq += 1;
        let clean = self
            .echo
            .process_capture(&i16_to_f32(&frame.samples))
            .expect("aec capture");
        let _ = clean;
        *self.calibration.lock().expect("calibration") = self.echo.calibration();
        Ok(Some(frame))
    }
}

/// One first-TTS burst longer than the 200 ms speaker-copy ring, paced like a
/// device callback (10 ms blocks). Real agent audio, not silence.
struct FirstReplyTts;

impl Tts for FirstReplyTts {
    fn name(&self) -> &'static str {
        BuiltinDefaults::v0().tts.as_str()
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        if cancel.is_stale(token.generation) {
            return Err(Error::Cancelled);
        }
        if !token.is_last {
            return Ok(Vec::new());
        }
        // 1 s of speech at 16 kHz. Speaker-copy ring is 200 ms.
        Ok(vec![SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index: 0,
            samples: vec![10_000; DEFAULT_SAMPLE_RATE_HZ as usize],
            is_last: true,
        }])
    }
}

struct SpeakerCallbackSink {
    ring: std::sync::Arc<SampleRing>,
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
            // Device-like pacing so a draining capture can empty the copy.
            thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    }

    fn interrupt(&mut self) {}
}

#[test]
fn first_tts_does_not_lose_speaker_copy_while_vad_is_paused() {
    let cap = device_ring_capacity_samples(DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CHANNELS);
    let ring = std::sync::Arc::new(SampleRing::new(cap));
    let reference = EchoReference::new(std::sync::Arc::clone(&ring), PcmFormat::v0());
    let calibration =
        std::sync::Arc::new(std::sync::Mutex::new(EchoCalibration::WaitingForPlayback));
    let mut echo = EchoController::new(reference).expect("echo");
    echo.start_render_pump();
    let report = run_loop_captured(
        LoopConfig {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::StopAfterTurns(1),
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(false),
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: FirstReplyTts,
            sink: SpeakerCallbackSink {
                ring: std::sync::Arc::clone(&ring),
            },
        },
        LiveMicWithAec {
            echo,
            seq: 0,
            speech_left: 3,
            calibration: std::sync::Arc::clone(&calibration),
        },
        Cancel::new(),
    )
    .expect("first tts aec");
    assert_eq!(report.turns.len(), 1, "first reply must complete");
    let status = *calibration.lock().expect("calibration");
    assert_ne!(
        status,
        EchoCalibration::Degraded,
        "first TTS dropped the speaker copy (headphones / reference lost); overruns={}",
        ring.overruns()
    );
    assert_eq!(
        ring.overruns(),
        0,
        "speaker-copy ring overflowed during first TTS"
    );
}
