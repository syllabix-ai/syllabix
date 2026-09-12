//! Public behavior contracts that must survive the large-file refactor series.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use syllabix_core::{
    bounded, run_loop, scripted_frames, AgentConfig, Cancel, CollectingSink, FakeLlm, FakeStt,
    FakeTts, FakeVad, Llm, LlmProvider, LoopConfig, LoopEvent, LoopMode, PipelineStages, Result,
    Stt, SttProvider, TokenChunk, ToolTurnEvent, Transcript, Tts, TtsProvider, Vad, VadProvider,
};

fn temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "syllabix-refactor-baseline-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos()
    ))
}

#[test]
fn provider_names_are_stable_at_the_public_boundary() {
    assert_eq!(VadProvider::Silero.as_str(), "silero");
    assert_eq!(SttProvider::Local.as_str(), "local");
    assert_eq!(LlmProvider::Local.as_str(), "local");
    assert_eq!(LlmProvider::Online.as_str(), "online");
    assert_eq!(TtsProvider::Local.as_str(), "local");
    assert_eq!(TtsProvider::Online.as_str(), "online");

    assert_eq!(FakeVad::new().name(), "silero");
    assert_eq!(FakeStt.name(), "local");
    assert_eq!(FakeLlm::new().name(), "local");
    assert_eq!(FakeTts.name(), "local");
}

#[test]
fn cancellation_invalidates_only_the_generation_until_shutdown() {
    let cancel = Cancel::new();
    let worker = cancel.clone();
    let generation_zero = worker.generation();

    let generation_one = cancel.cancel_generation();
    assert!(worker.is_stale(generation_zero));
    assert!(!worker.is_stale(generation_one));
    assert!(!worker.is_shutdown());

    worker.shutdown();
    assert!(cancel.is_shutdown());
    assert!(cancel.is_stale(generation_one));
}

#[test]
fn bounded_queue_preserves_fifo_order_and_accounting() {
    let (sender, receiver, stats) = bounded("baseline-order", 3);
    for value in ["first", "second", "third"] {
        sender.send(value).expect("send within capacity");
    }

    assert_eq!(receiver.recv().expect("first item"), "first");
    assert_eq!(receiver.recv().expect("second item"), "second");
    assert_eq!(receiver.recv().expect("third item"), "third");

    let occupancy = stats.snapshot();
    assert_eq!(occupancy.capacity, 3);
    assert_eq!(occupancy.high_water, 3);
    assert_eq!(occupancy.current, 0);
    assert_eq!(occupancy.sent, 3);
    assert_eq!(occupancy.received, 3);
    assert!(occupancy.within_capacity());
}

#[test]
fn canonical_config_round_trip_preserves_non_default_values() {
    let mut expected = AgentConfig::v0();
    expected.name = "baseline-agent".into();
    expected.vad_threshold = 0.63;
    expected.vad_min_speech_ms = 160;
    expected.vad_end_silence_ms = 480;
    expected.vad_preroll_ms = 240;
    expected.language = "auto".into();
    expected.system_prompt = "Answer in {language}.\nKeep quoted \"text\" intact.".into();
    expected.diagnostics_timestamps = true;
    expected.diagnostics_directory = PathBuf::from("target/baseline-turns");

    let rendered = expected.to_yaml();
    let parsed = AgentConfig::parse_yaml(&rendered).expect("canonical yaml parses");
    assert_eq!(parsed, expected);
    assert_eq!(parsed.to_yaml(), rendered);
}

struct ToolEventLlm {
    inner: FakeLlm,
    events: Vec<ToolTurnEvent>,
}

impl ToolEventLlm {
    fn new() -> Self {
        Self {
            inner: FakeLlm::new(),
            events: vec![
                ToolTurnEvent {
                    kind: "call".into(),
                    name: "shell".into(),
                    call_id: "call-1".into(),
                    arguments: r#"{"argv":["pwd"]}"#.into(),
                    content: String::new(),
                },
                ToolTurnEvent {
                    kind: "result".into(),
                    name: "shell".into(),
                    call_id: "call-1".into(),
                    arguments: r#"{"argv":["pwd"]}"#.into(),
                    content: "workspace".into(),
                },
                ToolTurnEvent {
                    kind: "call".into(),
                    name: "web_fetch".into(),
                    call_id: "call-2".into(),
                    arguments: r#"{"url":"https://example.test"}"#.into(),
                    content: String::new(),
                },
                ToolTurnEvent {
                    kind: "result".into(),
                    name: "web_fetch".into(),
                    call_id: "call-2".into(),
                    arguments: r#"{"url":"https://example.test"}"#.into(),
                    content: "example".into(),
                },
            ],
        }
    }
}

impl Llm for ToolEventLlm {
    fn name(&self) -> &'static str {
        "baseline-tool-llm"
    }

    fn take_tool_events(&mut self) -> Vec<ToolTurnEvent> {
        std::mem::take(&mut self.events)
    }

    fn generate(
        &mut self,
        history: &[syllabix_core::HistoryTurn],
        user: &Transcript,
        cancel: &Cancel,
        on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
    ) -> Result<()> {
        self.inner.generate(history, user, cancel, on_token)
    }
}

#[test]
fn tool_events_keep_provider_order_through_events_and_sidecar() {
    let directory = temp_dir("tool-order");
    let debug = syllabix_core::TurnDebug::open(&directory).expect("open diagnostics");
    let (events_tx, events_rx) = mpsc::channel();

    run_loop(
        LoopConfig {
            mode: LoopMode::StopAfterTurns(1),
            events: Some(events_tx),
            turn_debug: Some(debug),
            ..LoopConfig::default()
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: ToolEventLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        Cancel::new(),
    )
    .expect("one tool-enabled turn");

    let observed: Vec<_> = events_rx
        .try_iter()
        .filter_map(|event| match event {
            LoopEvent::Tool { event, .. } => Some((event.kind, event.call_id)),
            _ => None,
        })
        .collect();
    assert_eq!(
        observed,
        [
            ("call".into(), "call-1".into()),
            ("result".into(), "call-1".into()),
            ("call".into(), "call-2".into()),
            ("result".into(), "call-2".into()),
        ]
    );

    let sidecar = std::fs::read_to_string(directory.join("turn-000/turn.json"))
        .expect("read diagnostics sidecar");
    let json: serde_json::Value = serde_json::from_str(&sidecar).expect("valid sidecar json");
    let sidecar_order: Vec<_> = json["tool_events"]
        .as_array()
        .expect("tool event array")
        .iter()
        .map(|event| {
            (
                event["kind"].as_str().expect("event kind"),
                event["call_id"].as_str().expect("call id"),
            )
        })
        .collect();
    assert_eq!(
        sidecar_order,
        [
            ("call", "call-1"),
            ("result", "call-1"),
            ("call", "call-2"),
            ("result", "call-2"),
        ]
    );

    std::fs::remove_dir_all(directory).expect("remove diagnostics fixture");
}

#[test]
fn completed_turn_keeps_the_final_transcript_verbatim() {
    let (events_tx, events_rx) = mpsc::channel();
    let report = run_loop(
        LoopConfig {
            mode: LoopMode::StopAfterTurns(1),
            events: Some(events_tx),
            ..LoopConfig::default()
        },
        PipelineStages {
            vad: FakeVad::new(),
            stt: syllabix_core::ScriptedStt::new(["  final transcript, exactly.  "]),
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        Cancel::new(),
    )
    .expect("one completed turn");

    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.turns[0].user_text, "  final transcript, exactly.  ");
    let final_user_events: Vec<_> = events_rx
        .try_iter()
        .filter_map(|event| match event {
            LoopEvent::User { text, .. } => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(final_user_events, ["  final transcript, exactly.  "]);
}
