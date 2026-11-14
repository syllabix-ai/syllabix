//! One Riley voice session through the production cascade (#329 phase A2):
//! injected caller audio → VAD → STT → [`crate::llm::RileyLlm`] (prompt +
//! card tools) → TTS → [`crate::voice::RecordingSink`], driven by
//! [`syllabix_core::run_loop_captured`]. The result carries the transcript,
//! the tool trace, and the database before and after the call.

use std::sync::mpsc;
use std::time::Duration;

use serde_json::{json, Value};
use syllabix_core::defaults::BuiltinDefaults;
use syllabix_core::pipeline::LoopMode;
use syllabix_core::{
    run_loop_captured, Cancel, Llm, LoopConfig, LoopEvent, PipelineStages, RuntimeControls, Stt,
    Tts, Vad,
};

use crate::db::CardDb;
use crate::voice::{CallerTurn, InjectedCapture, RecordingSink, Render};

/// Mutating Riley tools (everything except the two lookups).
pub const MUTATING_TOOLS: [&str; 3] = [
    "change_card_status",
    "request_card_replacement",
    "update_card_replacement_status",
];

/// Timing and barge-in knobs for one session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionConfig {
    /// Let caller speech cut the agent's reply (`--barge-in`).
    pub barge_in: bool,
    /// Pace agent playback in real time.
    pub realtime_playback: bool,
    /// Delay per injected 32 ms frame.
    pub frame_pause: Duration,
    /// Silence frames after each caller turn (25 = Silero's 800 ms end silence).
    pub trailing_silence_frames: usize,
    /// How long a caller turn waits for the agent before the call ends.
    pub turn_timeout: Duration,
    /// Hard limit for the whole call.
    pub call_timeout: Duration,
}

impl Default for SessionConfig {
    /// Real-time pacing for live runs.
    fn default() -> Self {
        Self {
            barge_in: false,
            realtime_playback: true,
            frame_pause: Duration::from_millis(32),
            trailing_silence_frames: 25,
            turn_timeout: Duration::from_secs(60),
            call_timeout: Duration::from_secs(600),
        }
    }
}

impl SessionConfig {
    /// Fast pacing for tests with fake stages.
    pub fn fast() -> Self {
        Self {
            barge_in: false,
            realtime_playback: false,
            frame_pause: Duration::from_millis(2),
            trailing_silence_frames: 2,
            turn_timeout: Duration::from_secs(10),
            call_timeout: Duration::from_secs(60),
        }
    }
}

/// Who spoke a transcript line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    /// STT text of injected caller audio.
    Caller,
    /// Riley's reply text (what TTS was asked to say).
    Agent,
}

/// One transcript line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Line {
    /// Pipeline turn id.
    pub turn: u64,
    /// Speaker.
    pub speaker: Speaker,
    /// Text.
    pub text: String,
    /// Agent line whose playback did not finish (barge-in or failure).
    pub cut_off: bool,
}

/// One tool event from `LoopEvent::Tool`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolEvent {
    /// Pipeline turn id.
    pub turn: u64,
    /// `call`, `result`, `rejected`, or `limit`.
    pub kind: String,
    /// Tool name.
    pub name: String,
    /// Provider call id.
    pub call_id: String,
    /// Parsed arguments (`null` when absent).
    pub arguments: Value,
    /// Tool result text (empty for `call`).
    pub content: String,
}

impl ToolEvent {
    /// A `result` whose content is not `{"error": …}`.
    pub fn is_ok_result(&self) -> bool {
        self.kind == "result"
            && serde_json::from_str::<Value>(&self.content)
                .map(|value| value.get("error").is_none())
                .unwrap_or(false)
    }

    /// A successful result of a mutating tool.
    pub fn is_mutation(&self) -> bool {
        self.is_ok_result() && MUTATING_TOOLS.contains(&self.name.as_str())
    }
}

/// Everything A3's checks and judge need from one call.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionReport {
    /// Caller and agent lines in order.
    pub transcript: Vec<Line>,
    /// Tool calls and results in order.
    pub tool_trace: Vec<ToolEvent>,
    /// `CardDb::snapshot` before the call.
    pub initial_db: Value,
    /// `CardDb::snapshot` after the call (`null` if it could not be read).
    pub final_db: Value,
    /// Agent turns that finished playing.
    pub completed_turns: usize,
    /// Turns dropped after a provider error or blank STT.
    pub skipped_turns: usize,
    /// Barge-in playback flushes.
    pub interruptions: usize,
    /// Agent samples played (16 kHz).
    pub agent_samples: usize,
    /// Anything that went wrong. A non-empty list fails the call.
    pub errors: Vec<String>,
}

impl SessionReport {
    /// One JSON object (for A3's JSONL). Contains no secrets.
    pub fn to_json(&self) -> Value {
        let transcript: Vec<Value> = self
            .transcript
            .iter()
            .map(|line| {
                json!({
                    "turn": line.turn,
                    "speaker": match line.speaker { Speaker::Caller => "caller", Speaker::Agent => "agent" },
                    "text": line.text,
                    "cut_off": line.cut_off,
                })
            })
            .collect();
        let tool_trace: Vec<Value> = self
            .tool_trace
            .iter()
            .map(|event| {
                json!({
                    "turn": event.turn,
                    "kind": event.kind,
                    "name": event.name,
                    "call_id": event.call_id,
                    "arguments": event.arguments,
                    "content": event.content,
                })
            })
            .collect();
        json!({
            "transcript": transcript,
            "tool_trace": tool_trace,
            "final_db": self.final_db,
            "completed_turns": self.completed_turns,
            "skipped_turns": self.skipped_turns,
            "interruptions": self.interruptions,
            "agent_samples": self.agent_samples,
            "errors": self.errors,
        })
    }
}

/// VAD, STT, LLM, and TTS for one call. The sink is always [`RecordingSink`].
pub struct Stages<V, S, L, T> {
    /// VAD (`FakeVad` in tests, Silero for live runs).
    pub vad: V,
    /// STT (`RemoteStt` uses only `SYLLABIX_STT_API_KEY`).
    pub stt: S,
    /// Agent stage, normally [`crate::llm::RileyLlm`].
    pub llm: L,
    /// TTS (`RemoteTts` uses only `SYLLABIX_TTS_API_KEY`).
    pub tts: T,
}

/// Run one call. `db` must be a second connection to the schema the LLM's
/// tools use; it is only read (before and after). Never panics on stage
/// failures: they land in [`SessionReport::errors`].
pub fn run_session<V, S, L, T>(
    config: &SessionConfig,
    stages: Stages<V, S, L, T>,
    caller: Vec<CallerTurn>,
    db: &mut CardDb,
) -> SessionReport
where
    V: Vad + 'static,
    S: Stt + 'static,
    L: Llm + 'static,
    T: Tts + 'static,
{
    let mut errors = Vec::new();
    let initial_db = db.snapshot().unwrap_or_else(|err| {
        errors.push(format!(
            "could not read the database before the call: {err}"
        ));
        Value::Null
    });
    // Auto-timeouts off: an eval call must not mute itself.
    let controls = RuntimeControls::with_auto_timeout(config.barge_in, 0, 0);
    let render = Render::new();
    let capture = InjectedCapture::new(
        caller,
        controls.clone(),
        render.clone(),
        config.frame_pause,
        config.trailing_silence_frames,
        config.turn_timeout,
    );
    let sink = RecordingSink::new(render.clone(), config.realtime_playback);
    let (event_tx, event_rx) = mpsc::channel();
    let loop_config = LoopConfig {
        defaults: BuiltinDefaults::v0(),
        mode: LoopMode::UntilInputEnds,
        events: Some(event_tx),
        turn_debug: None,
        controls,
    };
    let cancel = Cancel::new();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let watchdog = {
        let cancel = cancel.clone();
        let limit = config.call_timeout;
        std::thread::spawn(move || {
            let timed_out = matches!(
                done_rx.recv_timeout(limit),
                Err(mpsc::RecvTimeoutError::Timeout)
            );
            if timed_out {
                cancel.shutdown();
            }
            timed_out
        })
    };
    let stages = PipelineStages {
        vad: stages.vad,
        stt: stages.stt,
        llm: stages.llm,
        tts: stages.tts,
        sink,
    };
    let outcome = run_loop_captured(loop_config, stages, capture, cancel);
    let _ = done_tx.send(());
    if watchdog.join().unwrap_or(true) {
        errors.push(format!("call exceeded {:?}", config.call_timeout));
    }
    let events: Vec<LoopEvent> = event_rx.try_iter().collect();
    let (completed_ids, completed_turns, skipped_turns) = match &outcome {
        Ok(report) => (
            report
                .turns
                .iter()
                .map(|turn| turn.id.0)
                .collect::<Vec<_>>(),
            report.turns.len(),
            report.skipped_turns,
        ),
        Err(err) => {
            errors.push(format!("pipeline failed: {err}"));
            (Vec::new(), 0, 0)
        }
    };
    if skipped_turns > 0 {
        errors.push(format!(
            "{skipped_turns} turn(s) skipped after a stage error"
        ));
    }
    let (_, _, interruptions, agent_samples, wait_timed_out) = render.snapshot();
    if wait_timed_out {
        errors.push(format!(
            "caller waited more than {:?} for the agent",
            config.turn_timeout
        ));
    }
    let final_db = db.snapshot().unwrap_or_else(|err| {
        errors.push(format!("could not read the database after the call: {err}"));
        Value::Null
    });
    let (transcript, tool_trace) = collect(&events, &completed_ids);
    SessionReport {
        transcript,
        tool_trace,
        initial_db,
        final_db,
        completed_turns,
        skipped_turns,
        interruptions,
        agent_samples,
        errors,
    }
}

/// Fold loop events into transcript lines and tool events.
pub fn collect(events: &[LoopEvent], completed: &[u64]) -> (Vec<Line>, Vec<ToolEvent>) {
    let mut lines: Vec<Line> = Vec::new();
    let mut tools = Vec::new();
    for event in events {
        match event {
            LoopEvent::User { turn, text, .. } => lines.push(Line {
                turn: turn.0,
                speaker: Speaker::Caller,
                text: text.clone(),
                cut_off: false,
            }),
            LoopEvent::Assistant { turn, text, .. } => match lines.last_mut() {
                Some(line) if line.speaker == Speaker::Agent && line.turn == turn.0 => {
                    line.text.push_str(text);
                }
                _ => lines.push(Line {
                    turn: turn.0,
                    speaker: Speaker::Agent,
                    text: text.clone(),
                    cut_off: !completed.contains(&turn.0),
                }),
            },
            LoopEvent::Tool { turn, event } => tools.push(ToolEvent {
                turn: turn.0,
                kind: event.kind.clone(),
                name: event.name.clone(),
                call_id: event.call_id.clone(),
                arguments: serde_json::from_str(&event.arguments).unwrap_or(Value::Null),
                content: event.content.clone(),
            }),
            _ => {}
        }
    }
    (lines, tools)
}

#[cfg(test)]
mod tests {
    use super::*;
    use syllabix_core::types::{ToolTurnEvent, TurnId};

    #[test]
    fn default_config_paces_in_real_time() {
        let config = SessionConfig::default();
        assert!(!config.barge_in);
        assert!(config.realtime_playback);
        assert_eq!(config.frame_pause, Duration::from_millis(32));
        assert_eq!(config.trailing_silence_frames, 25);
        assert_eq!(config.turn_timeout, Duration::from_secs(60));
        assert_eq!(config.call_timeout, Duration::from_secs(600));
        assert!(!SessionConfig::fast().realtime_playback);
    }

    #[test]
    fn collect_joins_agent_deltas_and_parses_tool_events() {
        let tool = |kind: &str, arguments: &str, content: &str| LoopEvent::Tool {
            turn: TurnId(0),
            event: ToolTurnEvent {
                kind: kind.into(),
                name: "change_card_status".into(),
                call_id: "c1".into(),
                arguments: arguments.into(),
                content: content.into(),
            },
        };
        let events = vec![
            LoopEvent::User {
                turn: TurnId(0),
                text: "freeze it".into(),
                language: "en".into(),
            },
            LoopEvent::Thinking { turn: TurnId(0) },
            LoopEvent::Assistant {
                turn: TurnId(0),
                text: "Do".into(),
                is_last: false,
            },
            LoopEvent::Assistant {
                turn: TurnId(0),
                text: "ne.".into(),
                is_last: true,
            },
            tool("call", "{\"card_id\":\"c\"}", ""),
            tool("result", "{\"card_id\":\"c\"}", "{\"status\":\"frozen\"}"),
            tool("result", "not json", "{\"error\":\"card not found\"}"),
            tool("result", "", "not json"),
        ];
        let (lines, tools) = collect(&events, &[]);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1].text, "Done.");
        assert!(lines[1].cut_off, "turn 0 never completed");
        assert_eq!(tools[0].arguments, json!({"card_id": "c"}));
        assert!(!tools[0].is_ok_result());
        assert!(tools[1].is_mutation());
        assert_eq!(tools[2].arguments, Value::Null);
        assert!(!tools[2].is_ok_result());
        assert!(!tools[3].is_ok_result());
        let report = SessionReport {
            transcript: lines,
            tool_trace: tools,
            initial_db: Value::Null,
            final_db: Value::Null,
            completed_turns: 0,
            skipped_turns: 0,
            interruptions: 0,
            agent_samples: 0,
            errors: vec![],
        };
        let json = report.to_json();
        assert_eq!(json["transcript"][0]["speaker"], "caller");
        assert_eq!(json["transcript"][1]["speaker"], "agent");
        assert_eq!(json["tool_trace"].as_array().unwrap().len(), 4);
    }
}
