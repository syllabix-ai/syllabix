//! Merge gate for PR 2: in-memory fake loop, order, bounds, shutdown, cancel.

use std::thread;
use std::time::Duration;

use syllabix_core::{
    run_loop, scripted_frames, BuiltinDefaults, Cancel, CollectingSink, FailOnceLlm, FailOnceStt,
    FailOnceTts, FakeLlm, FakeStt, FakeTts, FakeVad, LoopConfig, LoopMode, PipelineStages,
    QueueCaps, TurnId,
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
