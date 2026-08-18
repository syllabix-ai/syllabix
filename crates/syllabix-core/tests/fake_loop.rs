//! Merge gate for PR 2: in-memory fake loop, order, bounds, shutdown, cancel.

use std::thread;
use std::time::Duration;

use syllabix_core::{
    audio::{read_wav, PcmFormat},
    run_loop, scripted_frames, BuiltinDefaults, Cancel, CollectingSink, FailOnceLlm, FailOnceStt,
    FailOnceTts, FakeLlm, FakeStt, FakeTts, FakeVad, LoopConfig, LoopMode, PipelineStages,
    QueueCaps, TurnDebug, TurnId, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES,
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

fn unique_debug_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "syllabix-fake-turn-debug-{}-{}",
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
