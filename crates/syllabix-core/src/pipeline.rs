//! In-memory conversation loop: VAD → STT → LLM → TTS → sink with bounded queues.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::cancel::Cancel;
use crate::defaults::{BuiltinDefaults, QueueCaps};
use crate::error::{Error, Result};
use crate::providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
#[cfg(feature = "device-audio")]
use crate::queue::Occupancy;
use crate::queue::{bounded, BoundedSender, QueueReport};
#[cfg(feature = "device-audio")]
use crate::s2s::{DuplexS2s, JitterBuffer, S2sEvent};
use crate::turn_debug::{TimelineAnchor, TurnDebug};
#[cfg(feature = "device-audio")]
use crate::types::GenerationId;
use crate::types::{
    AudioFrame, CompletedTurn, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript, TurnId,
    TurnTimings, Utterance, VadEvent,
};

const POLL: Duration = Duration::from_millis(5);
/// Empty or whitespace Whisper text must not start LLM/TTS.
pub fn is_blank_stt(text: &str) -> bool {
    text.trim().is_empty()
}

fn tts_backend_id<T: 'static>(tts: &T) -> Option<&str> {
    let provider = tts as &dyn std::any::Any;
    if let Some(qwen) = provider.downcast_ref::<crate::tts::QwenTts>() {
        return qwen.backend_id();
    }
    #[cfg(any(coverage, feature = "ggml"))]
    {
        provider
            .downcast_ref::<crate::real::LiveTts>()
            .and_then(crate::real::LiveTts::backend_id)
    }
    #[cfg(not(any(coverage, feature = "ggml")))]
    {
        let _ = provider;
        None
    }
}

/// How the loop should finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Run until the frame source disconnects and every stage drains.
    UntilInputEnds,
    /// Stop after this many *completed* (played) turns.
    StopAfterTurns(usize),
}

/// Live transcript and latency events for the `run` TUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopEvent {
    /// Native providers and devices are ready; the terminal can begin its live footer.
    Ready,
    /// User STT text for a turn.
    User {
        /// Turn id.
        turn: TurnId,
        /// Transcript.
        text: String,
        /// Effective STT language code (configured or auto-detected).
        language: String,
    },
    /// Provisional STT text for the currently active VAD turn. It is enabled
    /// by the selected streaming STT provider; no separate user setting exists.
    Partial {
        /// Turn id.
        turn: TurnId,
        /// Full replace-in-place provisional text.
        text: String,
    },
    /// Assistant token text (delta).
    Assistant {
        /// Turn id.
        turn: TurnId,
        /// Token piece.
        text: String,
        /// Last token of the generation.
        is_last: bool,
    },
    /// Assistant turn started thinking (SpeechEnd committed). The mic is
    /// muted from here until speaking starts, even with `--barge-in`.
    Thinking {
        /// Turn id.
        turn: TurnId,
    },
    /// Native echo-cancellation state changed. A degraded reference requires
    /// restarting the live session to recreate the full-duplex device path.
    Aec {
        /// Whether AEC is actively processing the speaker reference.
        active: bool,
        /// Whether recovery requires restarting the live session.
        restart_required: bool,
    },
    /// Developer-harness trace evidence. It is never forwarded to TTS.
    Tool {
        /// Turn that caused the call.
        turn: TurnId,
        /// Structured call/result/rejection event.
        event: crate::types::ToolTurnEvent,
    },
    /// Turn finished playback; clocks for the TUI footer.
    Timings {
        /// Turn id.
        turn: TurnId,
        /// STT / TTFT / TTFB / total.
        timings: TurnTimings,
    },
    /// Speaker playback has started or fully drained.
    Playback { playing: bool },
}

/// Turn-control messages sent from VAD to STT on the ordered blocking queue.
/// VAD remains the only endpoint authority: `Start` opens the engine's
/// provisional state and `Finalize` carries the canonical utterance with
/// every preroll/hangover frame, so endpoint timing never depends on
/// partial-decode progress.
#[derive(Clone)]
enum SttControl {
    Start(TurnId),
    Finalize(Utterance),
}

/// One provisional frame sent from VAD to streaming STT on a separate lossy
/// channel. Partial text is advisory only — the `Finalize` utterance stays
/// canonical — so when a partial engine decodes slower than real time these
/// are dropped (`try_send`) instead of stalling VAD, capture, and the mic.
#[derive(Clone)]
struct SttPartial {
    turn: TurnId,
    frame: AudioFrame,
}

/// Result of polling idle auto-timeout thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoTimeoutAction {
    /// No threshold crossed.
    None,
    /// Mic should mute (or just muted).
    MicMute,
    /// Process should exit.
    Exit,
}

/// Monotonic clock for idle auto-timeout.
///
/// Production uses the system clock. Tests inject a manual clock and advance it
/// without sleeping.
#[derive(Debug, Clone)]
pub struct IdleClock(Arc<IdleClockInner>);

#[derive(Debug)]
enum IdleClockInner {
    System,
    Manual { now_ms: AtomicU64 },
}

impl IdleClock {
    /// Wall / monotonic system clock (`Instant::now()`).
    pub fn system() -> Self {
        Self(Arc::new(IdleClockInner::System))
    }

    /// Deterministic clock starting at `start_ms` (tests only).
    pub fn manual(start_ms: u64) -> Self {
        Self(Arc::new(IdleClockInner::Manual {
            now_ms: AtomicU64::new(start_ms),
        }))
    }

    /// Current time in milliseconds on this clock.
    pub fn now_ms(&self) -> u64 {
        match &*self.0 {
            IdleClockInner::System => {
                // Relative origin is fine: only deltas matter for idle.
                static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
                let origin = ORIGIN.get_or_init(Instant::now);
                Instant::now()
                    .saturating_duration_since(*origin)
                    .as_millis() as u64
            }
            IdleClockInner::Manual { now_ms } => now_ms.load(Ordering::SeqCst),
        }
    }

    /// Advance a manual clock. No-op for the system clock.
    pub fn advance(&self, delta: Duration) {
        let IdleClockInner::Manual { now_ms } = &*self.0 else {
            return;
        };
        now_ms.fetch_add(delta.as_millis() as u64, Ordering::SeqCst);
    }

    /// Jump a manual clock to an absolute millisecond. No-op for system.
    pub fn set_ms(&self, ms: u64) {
        let IdleClockInner::Manual { now_ms } = &*self.0 else {
            return;
        };
        now_ms.store(ms, Ordering::SeqCst);
    }
}

/// Runtime controls shared by the inline terminal and the voice loop.
#[derive(Debug, Clone)]
pub struct RuntimeControls(Arc<RuntimeControlsInner>);

#[derive(Debug)]
struct RuntimeControlsInner {
    barge_in: AtomicBool,
    mic_muted: AtomicBool,
    /// Mirrored from the loop's assistant-hold pause so fixture capture can
    /// wait for a listening window between turns.
    vad_paused: AtomicBool,
    /// `Duration::ZERO` disables.
    mic_mute_after: Duration,
    /// `Duration::ZERO` disables.
    exit_after: Duration,
    /// `None` means the idle clock is disarmed (active speech/turn).
    /// Value is [`IdleClock::now_ms`] when armed.
    idle_since_ms: Mutex<Option<u64>>,
    clock: IdleClock,
}

impl RuntimeControls {
    /// New controls with barge-in set from the command-line default and launch
    /// auto-timeout defaults (3 min mic mute / 10 min exit).
    pub fn new(barge_in: bool) -> Self {
        Self::with_auto_timeout(
            barge_in,
            crate::config::DEFAULT_AUTO_TIMEOUT_MIC_MUTE_MS,
            crate::config::DEFAULT_AUTO_TIMEOUT_EXIT_MS,
        )
    }

    /// New controls with explicit auto-timeout thresholds (`0` disables).
    pub fn with_auto_timeout(barge_in: bool, mic_mute_ms: u32, exit_ms: u32) -> Self {
        Self::with_auto_timeout_clock(barge_in, mic_mute_ms, exit_ms, IdleClock::system())
    }

    /// Like [`with_auto_timeout`], with an injectable clock (tests).
    pub fn with_auto_timeout_clock(
        barge_in: bool,
        mic_mute_ms: u32,
        exit_ms: u32,
        clock: IdleClock,
    ) -> Self {
        Self(Arc::new(RuntimeControlsInner {
            barge_in: AtomicBool::new(barge_in),
            mic_muted: AtomicBool::new(false),
            vad_paused: AtomicBool::new(false),
            mic_mute_after: Duration::from_millis(u64::from(mic_mute_ms)),
            exit_after: Duration::from_millis(u64::from(exit_ms)),
            idle_since_ms: Mutex::new(None),
            clock,
        }))
    }

    /// Shared idle clock (advance in tests).
    pub fn clock(&self) -> &IdleClock {
        &self.0.clock
    }

    pub fn barge_in(&self) -> bool {
        self.0.barge_in.load(Ordering::SeqCst)
    }
    pub fn mic_muted(&self) -> bool {
        self.0.mic_muted.load(Ordering::SeqCst)
    }

    /// True when VAD will accept inbound capture (not assistant-held, not muted).
    pub fn capture_open(&self) -> bool {
        !self.0.vad_paused.load(Ordering::SeqCst) && !self.mic_muted()
    }

    pub(crate) fn set_vad_paused(&self, paused: bool) {
        self.0.vad_paused.store(paused, Ordering::SeqCst);
    }

    pub fn toggle_barge_in(&self) -> bool {
        !self.0.barge_in.fetch_xor(true, Ordering::SeqCst)
    }

    /// Clear mic mute and restart the idle clock (listening restored).
    pub fn unmute_mic(&self) {
        self.0.mic_muted.store(false, Ordering::SeqCst);
        self.arm_idle();
    }

    /// Set mic mute and keep the idle clock armed so the exit timer still fires.
    pub fn mute_mic(&self) {
        self.0.mic_muted.store(true, Ordering::SeqCst);
        self.arm_idle();
    }

    /// Toggle manual mic mute. Returns the new muted state.
    pub fn toggle_mic_muted(&self) -> bool {
        if self.mic_muted() {
            self.unmute_mic();
            false
        } else {
            self.mute_mic();
            true
        }
    }

    /// Start or restart the idle clock from now (listening / post-TTS / keypress).
    pub fn arm_idle(&self) {
        *self.0.idle_since_ms.lock().expect("idle_since_ms") = Some(self.0.clock.now_ms());
    }

    /// Stop the idle clock while the user is speaking or a turn is in flight.
    pub fn disarm_idle(&self) {
        *self.0.idle_since_ms.lock().expect("idle_since_ms") = None;
    }

    /// True when the idle clock is armed (listening / post-TTS idle).
    pub fn idle_armed(&self) -> bool {
        self.0
            .idle_since_ms
            .lock()
            .expect("idle_since_ms")
            .is_some()
    }

    /// Reset the idle clock when it is currently armed (any keypress).
    pub fn touch_idle(&self) {
        let mut slot = self.0.idle_since_ms.lock().expect("idle_since_ms");
        if slot.is_some() {
            *slot = Some(self.0.clock.now_ms());
        }
    }

    /// Check idle thresholds. Mic-mute is sticky until [`unmute_mic`]
    /// or [`toggle_mic_muted`].
    pub fn poll_auto_timeout(&self) -> AutoTimeoutAction {
        let since = match *self.0.idle_since_ms.lock().expect("idle_since_ms") {
            Some(since) => since,
            None => return AutoTimeoutAction::None,
        };
        let elapsed = Duration::from_millis(self.0.clock.now_ms().saturating_sub(since));
        // Exit wins when both thresholds are crossed in the same poll.
        if !self.0.exit_after.is_zero() && elapsed >= self.0.exit_after {
            return AutoTimeoutAction::Exit;
        }
        if !self.0.mic_mute_after.is_zero() && elapsed >= self.0.mic_mute_after && !self.mic_muted()
        {
            self.0.mic_muted.store(true, Ordering::SeqCst);
            return AutoTimeoutAction::MicMute;
        }
        AutoTimeoutAction::None
    }
}

/// Configuration for one in-memory run.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Built-in names and queue caps.
    pub defaults: BuiltinDefaults,
    /// Stop condition.
    pub mode: LoopMode,
    /// Optional TUI / log subscriber.
    pub events: Option<Sender<LoopEvent>>,
    /// Opt-in per-turn WAV + sidecar writer. None means default `run` writes nothing.
    pub turn_debug: Option<TurnDebug>,
    /// Mutable terminal controls; the CLI's `--barge-in` sets their initial state.
    pub controls: RuntimeControls,
}

impl LoopConfig {
    /// Configure a 30-turn run using in-memory providers.
    pub fn thirty_turns() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::StopAfterTurns(30),
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(false),
        }
    }

    /// Configure a six-turn run using native providers.
    pub fn six_turns() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::StopAfterTurns(6),
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(false),
        }
    }
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
            events: None,
            turn_debug: None,
            controls: RuntimeControls::new(false),
        }
    }
}

/// Outcome of a pipeline run.
#[derive(Debug, Clone)]
pub struct LoopReport {
    /// Turns that reached playback of a last audio chunk, in turn-id order.
    pub turns: Vec<CompletedTurn>,
    /// Queue occupancy. High-water must stay within caps.
    pub queues: QueueReport,
    /// Provisional partial frames dropped because streaming STT decoded
    /// slower than real time. Dropping (not stalling VAD) is the design; a
    /// nonzero count explains sparse live partials, never endpoint timing.
    pub dropped_stt_partials: usize,
    /// Worker threads that exited (producer + five stages). Always 6 after a clean join.
    pub tasks_exited: usize,
    /// Live worker counter after joins; must be 0.
    pub tasks_still_running: usize,
    /// Whether shutdown was requested before natural completion.
    pub cancelled: bool,
    /// Turns dropped after a recoverable provider error or blank STT.
    pub skipped_turns: usize,
}

struct TurnAcc {
    user_text: Option<String>,
    assistant_text: String,
    token_count: usize,
    audio_chunks: usize,
    text_done: bool,
    audio_done: bool,
    utterance_at: Option<Instant>,
    stt_at: Option<Instant>,
    llm_at: Option<Instant>,
    first_token_at: Option<Instant>,
    first_audio_at: Option<Instant>,
    last_audio_at: Option<Instant>,
}

struct Shared {
    turns: Mutex<BTreeMap<TurnId, TurnAcc>>,
    live_tasks: Arc<AtomicUsize>,
    fail: Mutex<Option<Error>>,
    completed: AtomicUsize,
    skipped: AtomicUsize,
    /// Provisional STT partial frames dropped under backpressure.
    dropped_partials: AtomicUsize,
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    controls: RuntimeControls,
    pause_vad: AtomicBool,
    /// True from SpeechEnd until the first synthesized audio for the turn.
    /// While set the mic stays muted even with `--barge-in` on.
    thinking: AtomicBool,
    flush_playback: AtomicBool,
    assistant_turn: Mutex<Option<TurnId>>,
    interrupted: Mutex<HashSet<TurnId>>,
    /// Sidecar facts captured from the TTS stage before the workers start.
    tts_provider: &'static str,
    tts_model: Option<String>,
    tts_backend: Option<String>,
}

impl Shared {
    fn new(
        events: Option<Sender<LoopEvent>>,
        turn_debug: Option<TurnDebug>,
        controls: RuntimeControls,
        tts_provider: &'static str,
        tts_model: Option<String>,
        tts_backend: Option<String>,
    ) -> Arc<Self> {
        Arc::new(Self {
            turns: Mutex::new(BTreeMap::new()),
            live_tasks: Arc::new(AtomicUsize::new(0)),
            fail: Mutex::new(None),
            completed: AtomicUsize::new(0),
            skipped: AtomicUsize::new(0),
            dropped_partials: AtomicUsize::new(0),
            events,
            turn_debug,
            controls,
            pause_vad: AtomicBool::new(false),
            thinking: AtomicBool::new(false),
            flush_playback: AtomicBool::new(false),
            assistant_turn: Mutex::new(None),
            interrupted: Mutex::new(HashSet::new()),
            tts_provider,
            tts_model,
            tts_backend,
        })
    }

    fn emit(&self, event: LoopEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(event);
        }
    }

    /// Count one provisional partial frame shed under backpressure. VAD
    /// stays unblocked; the canonical `Finalize` utterance is unaffected.
    fn note_partial_drop(&self) {
        self.dropped_partials.fetch_add(1, Ordering::SeqCst);
    }

    fn note_skip(&self, turn: TurnId, cancel: &Cancel) {
        self.release_assistant(turn);
        self.skipped.fetch_add(1, Ordering::SeqCst);
        if let Some(debug) = &self.turn_debug {
            if let Err(err) = debug.skip(turn) {
                self.fail(err, cancel);
            }
        }
    }

    fn set_pause_vad(&self, paused: bool) {
        self.pause_vad.store(paused, Ordering::SeqCst);
        self.controls.set_vad_paused(paused);
    }

    fn mark_assistant(&self, turn: TurnId) {
        *self.assistant_turn.lock().expect("assistant turn") = Some(turn);
        // Thinking mutes the mic unconditionally. Speaking re-applies the
        // barge-in flag when the first audio arrives.
        self.thinking.store(true, Ordering::SeqCst);
        self.set_pause_vad(true);
        self.emit(LoopEvent::Thinking { turn });
    }

    fn clear_thinking(&self) {
        if self.thinking.swap(false, Ordering::SeqCst) {
            let active = self
                .assistant_turn
                .lock()
                .expect("assistant turn")
                .is_some();
            self.set_pause_vad(active && !self.controls.barge_in());
        }
    }

    fn release_assistant(&self, turn: TurnId) {
        let mut slot = self.assistant_turn.lock().expect("assistant turn");
        if *slot == Some(turn) {
            *slot = None;
            self.thinking.store(false, Ordering::SeqCst);
            self.set_pause_vad(false);
            self.controls.arm_idle();
        }
    }

    fn sync_barge_in(&self) {
        if self.thinking.load(Ordering::SeqCst) {
            self.set_pause_vad(true);
            return;
        }
        let active = self
            .assistant_turn
            .lock()
            .expect("assistant turn")
            .is_some();
        self.set_pause_vad(active && !self.controls.barge_in());
    }

    /// Cancel in-flight LLM/TTS when `--barge-in` hears SpeechStart.
    ///
    /// Always bump the generation and flush the speaker ring. Playback can still
    /// hold the previous turn after we released `assistant_turn` (last chunk
    /// queued, ring not empty) or while `play` is blocked feeding a long sentence.
    fn interrupt_assistant(&self, cancel: &Cancel) -> bool {
        if !self.controls.barge_in() {
            return false;
        }
        let turn = {
            let mut slot = self.assistant_turn.lock().expect("assistant turn");
            slot.take()
        };
        self.thinking.store(false, Ordering::SeqCst);
        self.set_pause_vad(false);
        cancel.cancel_generation();
        self.flush_playback.store(true, Ordering::SeqCst);
        if let Some(turn) = turn {
            self.interrupted
                .lock()
                .expect("interrupted turns")
                .insert(turn);
            if let Some(debug) = &self.turn_debug {
                if let Err(err) = debug.interrupt(turn) {
                    self.fail(err, cancel);
                }
            }
        }
        true
    }

    fn take_flush(&self) -> bool {
        self.flush_playback.swap(false, Ordering::SeqCst)
    }

    fn is_interrupted(&self, turn: TurnId) -> bool {
        self.interrupted
            .lock()
            .expect("interrupted turns")
            .contains(&turn)
    }

    fn fail(&self, err: Error, cancel: &Cancel) {
        cancel.shutdown();
        let mut slot = self.fail.lock().expect("fail mutex");
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    fn take_fail(&self) -> Option<Error> {
        self.fail.lock().expect("fail mutex").take()
    }

    fn mark_utterance(&self, turn: TurnId, at: Instant) {
        let mut map = self.turns.lock().expect("turn accumulator");
        let entry = map.entry(turn).or_insert_with(TurnAcc::new);
        if entry.utterance_at.is_none() {
            entry.utterance_at = Some(at);
        }
    }

    fn note_user(&self, turn: TurnId, text: String, language: &str, at: Instant) {
        {
            let mut map = self.turns.lock().expect("turn accumulator");
            let entry = map.entry(turn).or_insert_with(TurnAcc::new);
            entry.user_text = Some(text.clone());
            entry.stt_at = Some(at);
        }
        if let Some(debug) = &self.turn_debug {
            debug.note_stt(turn, &text, language);
        }
        self.emit(LoopEvent::User {
            turn,
            text,
            language: language.to_string(),
        });
    }

    fn mark_llm_start(&self, turn: TurnId, at: Instant) {
        let mut map = self.turns.lock().expect("turn accumulator");
        let entry = map.entry(turn).or_insert_with(TurnAcc::new);
        if entry.llm_at.is_none() {
            entry.llm_at = Some(at);
        }
    }

    fn note_token(&self, chunk: &TokenChunk, at: Instant) {
        {
            let mut map = self.turns.lock().expect("turn accumulator");
            let entry = map.entry(chunk.turn).or_insert_with(TurnAcc::new);
            entry.assistant_text.push_str(&chunk.text);
            entry.token_count += 1;
            if entry.first_token_at.is_none() {
                entry.first_token_at = Some(at);
                if let Some(debug) = &self.turn_debug {
                    debug.note_anchor(chunk.turn, TimelineAnchor::LlmFirstToken, at);
                }
            }
            if chunk.is_last {
                entry.text_done = true;
                if let Some(debug) = &self.turn_debug {
                    debug.note_last_anchor(chunk.turn, TimelineAnchor::LlmLastToken, at);
                    debug.note_llm(chunk.turn, entry.assistant_text.clone());
                }
            }
        }
        self.emit(LoopEvent::Assistant {
            turn: chunk.turn,
            text: chunk.text.clone(),
            is_last: chunk.is_last,
        });
    }

    fn note_audio(&self, chunk: &SynthesizedAudio, at: Instant, cancel: &Cancel) -> usize {
        if let Some(debug) = &self.turn_debug {
            debug.note_tts(
                chunk.turn,
                &chunk.samples,
                self.tts_provider,
                self.tts_model.as_deref(),
                self.tts_backend.as_deref(),
            );
        }
        let timings = {
            let mut map = self.turns.lock().expect("turn accumulator");
            let entry = map.entry(chunk.turn).or_insert_with(TurnAcc::new);
            entry.audio_chunks += 1;
            if entry.first_audio_at.is_none() {
                entry.first_audio_at = Some(at);
            }
            if chunk.is_last {
                entry.audio_done = true;
                entry.last_audio_at = Some(at);
                entry.timings()
            } else {
                None
            }
        };
        // First audio ends thinking: speaking re-applies the barge-in flag.
        self.clear_thinking();
        if let Some(timings) = timings {
            if let Some(debug) = &self.turn_debug {
                if let Err(err) = debug.complete(chunk.turn, timings) {
                    self.fail(err, cancel);
                }
            }
            self.emit(LoopEvent::Timings {
                turn: chunk.turn,
                timings,
            });
            return self.completed.fetch_add(1, Ordering::SeqCst) + 1;
        }
        self.completed.load(Ordering::SeqCst)
    }

    fn finished_turns(&self) -> Vec<CompletedTurn> {
        let map = self.turns.lock().expect("turn accumulator");
        map.iter()
            .filter(|(_, acc)| acc.user_text.is_some() && acc.text_done && acc.audio_done)
            .map(|(id, acc)| CompletedTurn {
                id: *id,
                user_text: acc.user_text.clone().unwrap_or_default(),
                assistant_text: acc.assistant_text.clone(),
                token_count: acc.token_count,
                audio_chunks: acc.audio_chunks,
                timings: acc.timings().unwrap_or_default(),
            })
            .collect()
    }
}

impl TurnAcc {
    fn new() -> Self {
        Self {
            user_text: None,
            assistant_text: String::new(),
            token_count: 0,
            audio_chunks: 0,
            text_done: false,
            audio_done: false,
            utterance_at: None,
            stt_at: None,
            llm_at: None,
            first_token_at: None,
            first_audio_at: None,
            last_audio_at: None,
        }
    }

    fn timings(&self) -> Option<TurnTimings> {
        let t0 = self.utterance_at?;
        let stt_at = self.stt_at.unwrap_or(t0);
        let llm_at = self.llm_at.unwrap_or(stt_at);
        let first_token = self.first_token_at.unwrap_or(llm_at);
        let first_audio = self.first_audio_at.unwrap_or(first_token);
        let last_audio = self.last_audio_at.unwrap_or(first_audio);
        Some(TurnTimings {
            stt: stt_at.saturating_duration_since(t0),
            ttft: first_token.saturating_duration_since(llm_at),
            ttfb: first_audio.saturating_duration_since(llm_at),
            total: last_audio.saturating_duration_since(t0),
        })
    }
}

struct TaskGuard {
    live: Arc<AtomicUsize>,
}

impl TaskGuard {
    fn new(live: &Arc<AtomicUsize>) -> Self {
        live.fetch_add(1, Ordering::SeqCst);
        Self {
            live: Arc::clone(live),
        }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Provider set moved into the worker threads for one run.
pub struct PipelineStages<V, S, L, T, K> {
    /// VAD implementation.
    pub vad: V,
    /// STT implementation.
    pub stt: S,
    /// LLM implementation.
    pub llm: L,
    /// TTS implementation.
    pub tts: T,
    /// Playback or collecting sink.
    pub sink: K,
}

/// Frame iterator presented as [`AudioCapture`] for the capture worker.
struct IterCapture<I> {
    iter: I,
}

impl<I> AudioCapture for IterCapture<I>
where
    I: Iterator<Item = AudioFrame> + Send,
{
    fn name(&self) -> &'static str {
        "fixture"
    }

    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        let frame = self.iter.next();
        // Scripted fixtures should resemble a microphone callback rather
        // than inject a future turn in the same scheduler instant. Real v0
        // frames are 32 ms; this keeps tests quick while leaving enough time
        // for the fake stages to finish a turn before the next begins.
        if frame.is_some() {
            thread::sleep(Duration::from_millis(25));
        }
        Ok(frame)
    }
}

/// Run the cascade on an in-memory frame iterator. Blocks until every worker joins.
pub fn run_loop<V, S, L, T, K, I>(
    config: LoopConfig,
    stages: PipelineStages<V, S, L, T, K>,
    frames: I,
    cancel: Cancel,
) -> Result<LoopReport>
where
    V: Vad + 'static,
    S: Stt + 'static,
    L: Llm + 'static,
    T: Tts + 'static,
    K: AudioSink + 'static,
    I: IntoIterator<Item = AudioFrame> + Send + 'static,
    I::IntoIter: Send + 'static,
{
    run_loop_captured(
        config,
        stages,
        IterCapture {
            iter: frames.into_iter(),
        },
        cancel,
    )
}

/// Run the cascade from an [`AudioCapture`] source (fixture or native mic).
pub fn run_loop_captured<V, S, L, T, K, C>(
    config: LoopConfig,
    stages: PipelineStages<V, S, L, T, K>,
    capture: C,
    cancel: Cancel,
) -> Result<LoopReport>
where
    V: Vad + 'static,
    S: Stt + 'static,
    L: Llm + 'static,
    T: Tts + 'static,
    K: AudioSink + 'static,
    C: AudioCapture + 'static,
{
    let PipelineStages {
        vad,
        stt,
        llm,
        tts,
        mut sink,
    } = stages;
    let tts_provider = tts.name();
    let tts_model = tts.model_id().map(str::to_string);
    let tts_backend = tts_backend_id(&tts).map(str::to_string);
    let caps: QueueCaps = config.defaults.queues;
    let (frame_tx, frame_rx, frame_stats) = bounded("frames", caps.frames);
    // Turn control stays blocking and ordered so VAD remains the single
    // endpoint authority. Provisional partial frames ride a separate lossy
    // channel: a streaming engine that decodes slower than real time (e.g.
    // Moonshine's full-utterance re-decode every ~512ms) must degrade live
    // partial text, never stall VAD and drop mic audio.
    let (utt_tx, utt_rx, utt_stats) = bounded("utterances", caps.utterances);
    let (partial_tx, partial_rx, partial_stats) = bounded("stt_partials", caps.stt_partials);
    let (tr_tx, tr_rx, tr_stats) = bounded("transcripts", caps.transcripts);
    let (tok_tx, tok_rx, tok_stats) = bounded("tokens", caps.tokens);
    let (aud_tx, aud_rx, aud_stats) = bounded("audio", caps.audio);

    let shared = Shared::new(
        config.events.clone(),
        config.turn_debug.clone(),
        config.controls.clone(),
        tts_provider,
        tts_model,
        tts_backend,
    );
    config.controls.arm_idle();
    let mut joins: Vec<JoinHandle<()>> = Vec::new();

    // Capture → VAD
    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        joins.push(spawn("syllabix-capture", live, move || {
            capture_loop(capture, frame_tx, &cancel, &shared)
        }));
    }

    // VAD
    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let utt_tx = utt_tx.clone();
        let partial_tx = partial_tx.clone();
        joins.push(spawn("syllabix-vad", live, move || {
            vad_loop(vad, frame_rx, utt_tx, partial_tx, &cancel, &shared)
        }));
    }
    drop(utt_tx);
    drop(partial_tx);

    // STT
    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let tr_tx = tr_tx.clone();
        joins.push(spawn("syllabix-stt", live, move || {
            stt_loop(stt, utt_rx, partial_rx, tr_tx, &cancel, &shared)
        }));
    }
    drop(tr_tx);

    // LLM
    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let tok_tx = tok_tx.clone();
        joins.push(spawn("syllabix-llm", live, move || {
            llm_loop(llm, tr_rx, tok_tx, &cancel, &shared)
        }));
    }
    drop(tok_tx);

    // TTS
    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let aud_tx = aud_tx.clone();
        joins.push(spawn("syllabix-tts", live, move || {
            tts_loop(tts, tok_rx, aud_tx, &cancel, &shared)
        }));
    }
    drop(aud_tx);

    // Sink + stop-after-N
    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let mode = config.mode;
        joins.push(spawn("syllabix-sink", live, move || {
            sink_loop(&mut sink, aud_rx, mode, &cancel, &shared)
        }));
    }

    let task_count = joins.len();
    let mut panic_msg = None;
    for handle in joins {
        if let Err(payload) = handle.join() {
            panic_msg = Some(panic_message(payload));
            cancel.shutdown();
        }
    }

    let tasks_still_running = shared.live_tasks.load(Ordering::SeqCst);
    if let Some(message) = panic_msg {
        return Err(Error::WorkerPanic { message });
    }
    if let Some(err) = shared.take_fail() {
        match err {
            Error::Cancelled => {}
            other => return Err(other),
        }
    }
    if let Some(debug) = &config.turn_debug {
        debug.finish_open()?;
    }

    Ok(LoopReport {
        turns: shared.finished_turns(),
        queues: QueueReport {
            frames: frame_stats.snapshot(),
            utterances: utt_stats.snapshot(),
            stt_partials: partial_stats.snapshot(),
            transcripts: tr_stats.snapshot(),
            tokens: tok_stats.snapshot(),
            audio: aud_stats.snapshot(),
        },
        dropped_stt_partials: shared.dropped_partials.load(Ordering::SeqCst),
        tasks_exited: task_count,
        tasks_still_running,
        cancelled: cancel.is_shutdown(),
        skipped_turns: shared.skipped.load(Ordering::SeqCst),
    })
}

/// Run capture → AEC3 → [`DuplexS2s::push`], and poll → sink plus AEC3 far-end.
///
/// Mute replaces outgoing samples with silence so the backend still sees a
/// continuous clock. Silero is not run. The loop does not infer a user
/// interruption from capture energy. [`run_loop`] is unchanged.
#[cfg(feature = "device-audio")]
pub fn run_duplex_loop<C, K, D>(
    config: LoopConfig,
    capture: C,
    mut sink: K,
    s2s: D,
    cancel: Cancel,
) -> Result<LoopReport>
where
    C: AudioCapture + 'static,
    K: AudioSink + 'static,
    D: DuplexS2s + 'static,
{
    let s2s = Arc::new(s2s);
    let aec = DuplexAec::new()?;
    let shared = Shared::new(
        config.events.clone(),
        config.turn_debug.clone(),
        config.controls.clone(),
        s2s.name(),
        None,
        None,
    );
    config.controls.arm_idle();
    let mut joins: Vec<JoinHandle<()>> = Vec::new();
    let jitter = Arc::new(Mutex::new(JitterBuffer::new()));

    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let s2s = Arc::clone(&s2s);
        let aec = aec.clone();
        joins.push(spawn("syllabix-s2s-capture", live, move || {
            duplex_capture_loop(capture, s2s, aec, &cancel, &shared);
        }));
    }

    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let s2s = Arc::clone(&s2s);
        let jitter = Arc::clone(&jitter);
        joins.push(spawn("syllabix-s2s-poll", live, move || {
            duplex_poll_loop(s2s, jitter, &cancel, &shared);
        }));
    }

    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let aec = aec.clone();
        let mode = config.mode;
        joins.push(spawn("syllabix-s2s-play", live, move || {
            duplex_play_loop(&mut sink, aec, jitter, mode, &cancel, &shared);
        }));
    }

    let task_count = joins.len();
    let mut panic_msg = None;
    for handle in joins {
        if let Err(payload) = handle.join() {
            panic_msg = Some(panic_message(payload));
            cancel.shutdown();
        }
    }

    let tasks_still_running = shared.live_tasks.load(Ordering::SeqCst);
    if let Some(message) = panic_msg {
        return Err(Error::WorkerPanic { message });
    }
    if let Some(err) = shared.take_fail() {
        match err {
            Error::Cancelled => {}
            other => return Err(other),
        }
    }
    if let Some(debug) = &config.turn_debug {
        debug.finish_open()?;
    }

    Ok(LoopReport {
        turns: Vec::new(),
        queues: idle_queue_report(),
        dropped_stt_partials: 0,
        tasks_exited: task_count,
        tasks_still_running,
        cancelled: cancel.is_shutdown(),
        skipped_turns: 0,
    })
}

#[cfg(feature = "device-audio")]
fn idle_queue_report() -> QueueReport {
    QueueReport {
        frames: idle_occupancy("frames"),
        utterances: idle_occupancy("utterances"),
        stt_partials: idle_occupancy("stt_partials"),
        transcripts: idle_occupancy("transcripts"),
        tokens: idle_occupancy("tokens"),
        audio: idle_occupancy("audio"),
    }
}

#[cfg(feature = "device-audio")]
fn idle_occupancy(name: &'static str) -> Occupancy {
    Occupancy {
        name,
        capacity: 1,
        current: 0,
        high_water: 0,
        sent: 0,
        received: 0,
    }
}

#[cfg(feature = "device-audio")]
#[derive(Clone)]
struct DuplexAec {
    echo: Arc<Mutex<crate::audio::EchoController>>,
    ring: Arc<crate::audio::SampleRing>,
}

#[cfg(feature = "device-audio")]
impl DuplexAec {
    fn new() -> Result<Self> {
        use crate::audio::{
            device_ring_capacity_samples, EchoController, EchoReference, PcmFormat, SampleRing,
        };
        use crate::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ};

        let ring = Arc::new(SampleRing::new(device_ring_capacity_samples(
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CHANNELS,
        )));
        let reference = EchoReference::new(Arc::clone(&ring), PcmFormat::v0());
        let mut echo = EchoController::new(reference)?;
        echo.start_render_pump();
        Ok(Self {
            echo: Arc::new(Mutex::new(echo)),
            ring,
        })
    }

    fn clean_frames(
        &self,
        frame: AudioFrame,
        split: &mut crate::audio::FrameSplitter,
        shared: &Shared,
        echo_status: &mut crate::audio::EchoCalibration,
    ) -> Result<Vec<AudioFrame>> {
        use crate::audio::{f32_to_i16, i16_to_f32, EchoCalibration};

        let clean = self
            .echo
            .lock()
            .expect("duplex echo")
            .process_capture(&i16_to_f32(&frame.samples))?;
        let status = self.echo.lock().expect("duplex echo").calibration();
        if status != *echo_status {
            let (active, restart_required) = match status {
                EchoCalibration::WaitingForPlayback | EchoCalibration::Calibrating => {
                    (false, false)
                }
                EchoCalibration::Active => (true, false),
                EchoCalibration::Degraded => (false, true),
            };
            shared.emit(LoopEvent::Aec {
                active,
                restart_required,
            });
            *echo_status = status;
        }
        split.push(&f32_to_i16(&clean))
    }

    fn feed_far_end(&self, samples: &[i16], cancel: &Cancel) -> Result<()> {
        use crate::audio::i16_to_f32;
        self.ring
            .push_slice_cancellable(&i16_to_f32(samples), cancel, GenerationId(0))
    }
}

#[cfg(feature = "device-audio")]
fn duplex_capture_loop<C, D>(
    mut capture: C,
    s2s: Arc<D>,
    aec: DuplexAec,
    cancel: &Cancel,
    shared: &Shared,
) where
    C: AudioCapture,
    D: DuplexS2s,
{
    use crate::audio::{EchoCalibration, FrameSplitter};

    let mut split = FrameSplitter::new();
    let mut echo_status = EchoCalibration::WaitingForPlayback;
    loop {
        if cancel.is_shutdown() {
            return;
        }
        match shared.controls.poll_auto_timeout() {
            AutoTimeoutAction::Exit => {
                cancel.shutdown();
                return;
            }
            AutoTimeoutAction::MicMute | AutoTimeoutAction::None => {}
        }
        match capture.next_frame(cancel) {
            Ok(Some(frame)) => {
                let outgoing = match aec.clean_frames(frame, &mut split, shared, &mut echo_status) {
                    Ok(frames) => frames,
                    Err(Error::Cancelled) => return,
                    Err(err) => {
                        shared.fail(err, cancel);
                        return;
                    }
                };
                for mut frame in outgoing {
                    if shared.controls.mic_muted() {
                        for sample in &mut frame.samples {
                            *sample = 0;
                        }
                    }
                    ignore_cancel(s2s.push(frame, cancel), shared, cancel);
                    if cancel.is_shutdown() {
                        return;
                    }
                }
            }
            Ok(None) => {
                match split.flush() {
                    Ok(rest) => {
                        for mut frame in rest {
                            if shared.controls.mic_muted() {
                                for sample in &mut frame.samples {
                                    *sample = 0;
                                }
                            }
                            ignore_cancel(s2s.push(frame, cancel), shared, cancel);
                        }
                    }
                    Err(Error::Cancelled) => {}
                    Err(err) => shared.fail(err, cancel),
                }
                cancel.shutdown();
                return;
            }
            Err(Error::Cancelled) => return,
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
        }
    }
}

#[cfg(feature = "device-audio")]
fn duplex_poll_loop<D>(
    s2s: Arc<D>,
    jitter: Arc<Mutex<JitterBuffer>>,
    cancel: &Cancel,
    shared: &Shared,
) where
    D: DuplexS2s,
{
    loop {
        if cancel.is_shutdown() {
            return;
        }
        match s2s.poll(cancel) {
            Ok(S2sEvent::Audio(frame)) => {
                jitter.lock().expect("duplex jitter").push(&frame.samples);
            }
            Ok(S2sEvent::Text(text)) => {
                shared.emit(LoopEvent::Assistant {
                    turn: TurnId(0),
                    text,
                    is_last: false,
                });
            }
            Ok(S2sEvent::Error { message }) => {
                shared.fail(
                    Error::Provider {
                        provider: s2s.name(),
                        message,
                    },
                    cancel,
                );
                return;
            }
            Err(Error::Cancelled) => return,
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
        }
    }
}

#[cfg(feature = "device-audio")]
fn duplex_quantum_duration() -> Duration {
    Duration::from_nanos(
        crate::types::FRAME_SAMPLES as u64 * 1_000_000_000
            / u64::from(crate::types::DEFAULT_SAMPLE_RATE_HZ),
    )
}

#[cfg(feature = "device-audio")]
fn wait_remaining(tick: Duration, started: Instant, cancel: &Cancel) {
    loop {
        if cancel.is_shutdown() {
            return;
        }
        let elapsed = started.elapsed();
        if elapsed >= tick {
            return;
        }
        thread::sleep((tick - elapsed).min(POLL));
    }
}

/// Pull jittered PCM at a playback clock and hand it to the sink.
///
/// Packets stay in [`JitterBuffer`] until this thread pulls a quantum, so a
/// burst is capped at 240 ms here rather than in the device ring. An empty
/// pull is an underrun: playback stops until 90 ms is buffered again.
/// Far-end is fed only after a successful play of those pulled samples.
#[cfg(feature = "device-audio")]
fn duplex_play_loop<K>(
    sink: &mut K,
    aec: DuplexAec,
    jitter: Arc<Mutex<JitterBuffer>>,
    mode: LoopMode,
    cancel: &Cancel,
    shared: &Shared,
) where
    K: AudioSink,
{
    let tick = duplex_quantum_duration();
    let mut index = 0_u32;
    let mut played = 0_usize;
    loop {
        if cancel.is_shutdown() {
            return;
        }
        let samples = jitter
            .lock()
            .expect("duplex jitter")
            .pull(crate::s2s::JITTER_PULL_SAMPLES);
        let Some(samples) = samples else {
            thread::sleep(POLL);
            continue;
        };
        let audio = SynthesizedAudio {
            turn: TurnId(0),
            generation: GenerationId(0),
            index,
            samples: samples.clone(),
            is_last: false,
        };
        index = index.saturating_add(1);
        let started = Instant::now();
        match sink.play(audio, cancel) {
            Ok(()) => {
                match aec.feed_far_end(&samples, cancel) {
                    Ok(()) => {}
                    Err(Error::Cancelled) => {
                        if cancel.is_shutdown() {
                            return;
                        }
                    }
                    Err(err) => {
                        shared.fail(err, cancel);
                        return;
                    }
                }
                shared.emit(LoopEvent::Playback { playing: true });
                played += 1;
                maybe_stop_after(mode, played, cancel);
                wait_remaining(tick, started, cancel);
            }
            Err(Error::Cancelled) => {
                if cancel.is_shutdown() {
                    return;
                }
            }
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
        }
    }
}

fn spawn(
    name: &str,
    live: Arc<AtomicUsize>,
    body: impl FnOnce() + Send + 'static,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let _guard = TaskGuard::new(&live);
            body();
        })
        .unwrap_or_else(|err| panic!("spawn {name}: {err}"))
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

fn ignore_cancel(result: Result<()>, shared: &Shared, cancel: &Cancel) {
    match result {
        Ok(()) => {}
        Err(Error::Cancelled) => {}
        Err(err) => shared.fail(err, cancel),
    }
}

fn capture_loop<C: AudioCapture>(
    mut capture: C,
    tx: BoundedSender<AudioFrame>,
    cancel: &Cancel,
    shared: &Shared,
) {
    loop {
        if cancel.is_shutdown() {
            return;
        }
        match capture.next_frame(cancel) {
            Ok(Some(frame)) => {
                ignore_cancel(tx.send_cancellable(frame, cancel), shared, cancel);
                if cancel.is_shutdown() {
                    return;
                }
            }
            Ok(None) => return,
            Err(Error::Cancelled) => return,
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
        }
    }
}

fn vad_loop<V: Vad>(
    mut vad: V,
    rx: crate::queue::BoundedReceiver<AudioFrame>,
    control: BoundedSender<SttControl>,
    partials: BoundedSender<SttPartial>,
    cancel: &Cancel,
    shared: &Shared,
) {
    let mut active: Option<TurnId> = None;
    let emit = |events: Vec<VadEvent>,
                control: &BoundedSender<SttControl>,
                partials: &BoundedSender<SttPartial>,
                cancel: &Cancel,
                active: &mut Option<TurnId>,
                frame: Option<&AudioFrame>|
     -> Result<()> {
        for event in &events {
            if let VadEvent::SpeechStart { turn } = event {
                shared.controls.disarm_idle();
                shared.interrupt_assistant(cancel);
                *active = Some(*turn);
                if let Some(debug) = &shared.turn_debug {
                    debug.start_turn(*turn);
                    debug.note_anchor(*turn, TimelineAnchor::SpeechStart, Instant::now());
                }
                control.send_cancellable(SttControl::Start(*turn), cancel)?;
            }
        }
        if let (Some(debug), Some(frame), Some(turn)) = (&shared.turn_debug, frame, *active) {
            debug.note_frame(turn, &frame.samples, frame.capture_pcm.as_deref());
        }
        // Stream only frames belonging to an active VAD turn. The canonical
        // final Utterance below still contains preroll/hangover frames.
        // Partial text is advisory: when the streaming engine decodes slower
        // than real time the channel fills and the frame is counted as
        // dropped instead of stalling VAD, capture, and the microphone.
        if let (Some(turn), Some(frame)) = (*active, frame) {
            if partials
                .try_send(SttPartial {
                    turn,
                    frame: frame.clone(),
                })
                .is_err()
            {
                shared.note_partial_drop();
            }
        }
        for event in events {
            if let VadEvent::SpeechEnd { utterance } = event {
                if let Some(debug) = &shared.turn_debug {
                    debug.note_utterance(&utterance);
                    debug.note_anchor(utterance.turn, TimelineAnchor::SpeechEnd, Instant::now());
                }
                shared.mark_assistant(utterance.turn);
                *active = None;
                control.send_cancellable(SttControl::Finalize(utterance), cancel)?;
            }
        }
        Ok(())
    };

    loop {
        if cancel.is_shutdown() {
            return;
        }
        match shared.controls.poll_auto_timeout() {
            AutoTimeoutAction::Exit => {
                cancel.shutdown();
                return;
            }
            AutoTimeoutAction::MicMute | AutoTimeoutAction::None => {}
        }
        shared.sync_barge_in();
        if shared.pause_vad.load(Ordering::SeqCst) || shared.controls.mic_muted() {
            match rx.recv_timeout(POLL) {
                Ok(frame) => {
                    // AEC still receives live capture through `NativeCapture`,
                    // but default mode must not turn assistant playback (or a
                    // user talking over it) into a delayed next utterance.
                    drop(frame);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    // The capture worker can finish while the final speaker
                    // buffer is still draining. Keep the bounded window until
                    // the sink releases VAD, then process it below.
                    thread::sleep(POLL);
                }
            }
            continue;
        }
        let frame = rx.recv_timeout(POLL);
        match frame {
            Ok(frame) => {
                // The STT streaming copy must not depend on diagnostics: the
                // old `turn_debug`-gated tap silently disabled live partials
                // whenever diagnostics were off.
                let streamed = frame.clone();
                match vad.push_frame(frame) {
                    Ok(events) => ignore_cancel(
                        emit(
                            events,
                            &control,
                            &partials,
                            cancel,
                            &mut active,
                            Some(&streamed),
                        ),
                        shared,
                        cancel,
                    ),
                    Err(Error::Cancelled) => return,
                    Err(err) => {
                        shared.fail(err, cancel);
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                match vad.flush() {
                    Ok(events) => ignore_cancel(
                        emit(events, &control, &partials, cancel, &mut active, None),
                        shared,
                        cancel,
                    ),
                    Err(Error::Cancelled) => {}
                    Err(err) => shared.fail(err, cancel),
                }
                return;
            }
        }
    }
}

fn stt_loop<S: Stt>(
    mut stt: S,
    control: crate::queue::BoundedReceiver<SttControl>,
    partials: crate::queue::BoundedReceiver<SttPartial>,
    tx: BoundedSender<Transcript>,
    cancel: &Cancel,
    shared: &Shared,
) {
    // Turn currently owned by the engine. Partials for any other turn are
    // stale (dropped overflow or a turn boundary) and never decoded.
    let mut active: Option<TurnId> = None;
    loop {
        if cancel.is_shutdown() {
            return;
        }
        match control.recv_timeout(POLL) {
            Ok(SttControl::Start(turn)) => {
                drain_partials(&partials);
                active = Some(turn);
                if let Err(err) = stt.start_turn(turn, cancel) {
                    if !matches!(err, Error::Cancelled) {
                        shared.fail(err, cancel);
                    }
                    return;
                }
            }
            Ok(SttControl::Finalize(utterance)) => {
                // A fast turn can finalize before any control poll observes
                // an idle window: decode the latest pending partial first so
                // provisional text still precedes the final transcript.
                if active == Some(utterance.turn)
                    && !shared.is_interrupted(utterance.turn)
                    && stt.supports_partials()
                {
                    if let Some(latest) = drain_to_latest(&partials, utterance.turn) {
                        if !decode_partial(&mut stt, cancel, shared, latest.turn, &latest.frame) {
                            return;
                        }
                    }
                } else {
                    drain_partials(&partials);
                }
                active = None;
                if shared.is_interrupted(utterance.turn) {
                    stt.cancel_turn(utterance.turn);
                    continue;
                }
                let queued_at = Instant::now();
                shared.mark_utterance(utterance.turn, queued_at);
                if let Some(debug) = &shared.turn_debug {
                    debug.note_anchor(utterance.turn, TimelineAnchor::SttQueued, queued_at);
                }
                match stt.transcribe(&utterance, cancel) {
                    Ok(transcript) => {
                        let done_at = Instant::now();
                        if let Some(debug) = &shared.turn_debug {
                            debug.note_anchor(transcript.turn, TimelineAnchor::SttDone, done_at);
                        }
                        if is_blank_stt(&transcript.text) {
                            if let Some(debug) = &shared.turn_debug {
                                debug.note_stt(
                                    transcript.turn,
                                    &transcript.text,
                                    &transcript.language,
                                );
                            }
                            shared.note_skip(transcript.turn, cancel);
                        } else {
                            let language = transcript.language.clone();
                            shared.note_user(
                                transcript.turn,
                                transcript.text.clone(),
                                &language,
                                done_at,
                            );
                            ignore_cancel(tx.send_cancellable(transcript, cancel), shared, cancel);
                        }
                    }
                    Err(Error::Cancelled) => {
                        if cancel.is_shutdown() {
                            return;
                        }
                    }
                    Err(err) if err.is_turn_recoverable() => {
                        shared.note_skip(utterance.turn, cancel);
                    }
                    Err(err) => {
                        shared.fail(err, cancel);
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                // Control idle: at most one provisional decode per poll, and
                // only the latest queued frame — older ones are stale by
                // definition. A `Finalize` that lands during the decode is
                // handled on the next poll; the canonical utterance carries
                // the same audio, so no endpoint timing is lost.
                let Some(latest) = drain_to_latest_for_active(&partials, active) else {
                    continue;
                };
                if !stt.supports_partials() {
                    continue;
                }
                if shared.is_interrupted(latest.turn) {
                    continue;
                }
                if !decode_partial(&mut stt, cancel, shared, latest.turn, &latest.frame) {
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// Drop every queued provisional frame (turn boundary or unsupported engine).
fn drain_partials(rx: &crate::queue::BoundedReceiver<SttPartial>) {
    while rx.try_recv().is_ok() {}
}

/// Drain the partial channel and return the latest frame for `turn`, if any.
/// Frames for other turns are stale and dropped; partial text is advisory, so
/// losing them never affects the canonical `Finalize` utterance.
fn drain_to_latest(
    rx: &crate::queue::BoundedReceiver<SttPartial>,
    turn: TurnId,
) -> Option<SttPartial> {
    let mut latest = None;
    while let Ok(partial) = rx.try_recv() {
        if partial.turn == turn {
            latest = Some(partial);
        }
    }
    latest
}

/// Drain the partial channel and return the latest frame when it belongs to
/// the engine's active turn. Anything else (overflow from a previous turn, or
/// frames for a turn whose `Start` has not been processed yet) is dropped so
/// provisional state can never accumulate across turn boundaries.
fn drain_to_latest_for_active(
    rx: &crate::queue::BoundedReceiver<SttPartial>,
    active: Option<TurnId>,
) -> Option<SttPartial> {
    let Some(active) = active else {
        drain_partials(rx);
        return None;
    };
    drain_to_latest(rx, active)
}

/// Run one provisional decode and emit non-blank text. Returns false when the
/// worker must exit (cancellation or provider failure).
fn decode_partial<S: Stt>(
    stt: &mut S,
    cancel: &Cancel,
    shared: &Shared,
    turn: TurnId,
    frame: &AudioFrame,
) -> bool {
    match stt.push_frame(frame, cancel) {
        Ok(Some(text)) if !is_blank_stt(&text) => {
            let at = Instant::now();
            if let Some(debug) = &shared.turn_debug {
                debug.note_anchor(turn, TimelineAnchor::SttPartial, at);
            }
            shared.emit(LoopEvent::Partial { turn, text });
            true
        }
        Ok(_) => true,
        Err(Error::Cancelled) => false,
        Err(err) => {
            shared.fail(err, cancel);
            false
        }
    }
}

fn llm_loop<L: Llm>(
    mut llm: L,
    rx: crate::queue::BoundedReceiver<Transcript>,
    tx: BoundedSender<TokenChunk>,
    cancel: &Cancel,
    shared: &Shared,
) {
    let mut history: Vec<HistoryTurn> = Vec::new();
    loop {
        match rx.recv_cancellable(cancel) {
            Ok(Some(user)) => {
                if shared.is_interrupted(user.turn) {
                    continue;
                }
                let mut assistant = String::new();
                let llm_started = Instant::now();
                shared.mark_llm_start(user.turn, llm_started);
                if let Some(debug) = &shared.turn_debug {
                    debug.note_anchor(user.turn, TimelineAnchor::LlmStart, llm_started);
                }
                let gen_result = llm.generate(&history, &user, cancel, &mut |chunk| {
                    if cancel.is_stale(chunk.generation) {
                        return Err(Error::Cancelled);
                    }
                    assistant.push_str(&chunk.text);
                    shared.note_token(&chunk, Instant::now());
                    tx.send_cancellable(chunk, cancel)
                });
                let tool_events = llm.take_tool_events();
                for event in &tool_events {
                    shared.emit(LoopEvent::Tool {
                        turn: user.turn,
                        event: event.clone(),
                    });
                }
                // Sidecar provider facts (provider/model/endpoint/request-id)
                // land before any immediate dump (skip) can write the turn.
                if let Some(debug) = &shared.turn_debug {
                    debug.note_llm_meta(user.turn, llm.debug_meta());
                    debug.note_tool_events(user.turn, tool_events);
                }
                match gen_result {
                    Ok(()) => {
                        history.push(HistoryTurn { user, assistant });
                    }
                    Err(Error::Cancelled) => {
                        shared.release_assistant(user.turn);
                        if let Some(debug) = &shared.turn_debug {
                            debug.note_llm(user.turn, assistant);
                        }
                        if cancel.is_shutdown() {
                            return;
                        }
                        // Generation cancel: keep listening so the next user turn is preserved.
                    }
                    Err(err) if err.is_turn_recoverable() => {
                        shared.note_skip(user.turn, cancel);
                    }
                    Err(err) => {
                        shared.fail(err, cancel);
                        return;
                    }
                }
            }
            Ok(None) => return,
            Err(Error::Cancelled) => return,
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
        }
    }
}

fn tts_loop<T: Tts>(
    mut tts: T,
    rx: crate::queue::BoundedReceiver<TokenChunk>,
    tx: BoundedSender<SynthesizedAudio>,
    cancel: &Cancel,
    shared: &Shared,
) {
    loop {
        match rx.recv_cancellable(cancel) {
            Ok(Some(token)) => {
                if shared.is_interrupted(token.turn) || cancel.is_stale(token.generation) {
                    continue;
                }
                match tts.synthesize_chunk_into(&token, cancel, &mut |audio| {
                    if !cancel.is_stale(audio.generation) {
                        if let Some(debug) = &shared.turn_debug {
                            let now = Instant::now();
                            // First write wins; a single-chunk turn sets both.
                            debug.note_anchor(audio.turn, TimelineAnchor::TtsFirstPcm, now);
                            if audio.is_last {
                                debug.note_last_anchor(audio.turn, TimelineAnchor::TtsLastPcm, now);
                            }
                        }
                        ignore_cancel(tx.send_cancellable(audio, cancel), shared, cancel);
                    }
                    Ok(())
                }) {
                    Ok(()) => {}
                    Err(Error::Cancelled) => {
                        shared.release_assistant(token.turn);
                        if cancel.is_shutdown() {
                            return;
                        }
                    }
                    Err(err) if err.is_turn_recoverable() => {
                        shared.note_skip(token.turn, cancel);
                    }
                    Err(err) => {
                        shared.fail(err, cancel);
                        return;
                    }
                }
            }
            Ok(None) => return,
            Err(Error::Cancelled) => return,
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
        }
    }
}

fn sink_loop<K: AudioSink>(
    sink: &mut K,
    rx: crate::queue::BoundedReceiver<SynthesizedAudio>,
    mode: LoopMode,
    cancel: &Cancel,
    shared: &Shared,
) {
    loop {
        if cancel.is_shutdown() {
            return;
        }
        if shared.take_flush() {
            sink.interrupt();
        }
        match rx.recv_timeout(POLL) {
            Ok(audio) => {
                if shared.take_flush() {
                    sink.interrupt();
                }
                if cancel.is_stale(audio.generation) || shared.is_interrupted(audio.turn) {
                    continue;
                }
                let is_last = audio.is_last;
                let turn = audio.turn;
                let generation = audio.generation;
                // Enter speaking before a sink can block on the first audio
                // chunk. With --barge-in this re-opens VAD so a new
                // SpeechStart can cancel the blocked playback.
                shared.clear_thinking();
                match sink.play(audio.clone(), cancel) {
                    Ok(()) => {
                        shared.emit(LoopEvent::Playback { playing: true });
                        if shared.take_flush() {
                            sink.interrupt();
                        }
                        if cancel.is_stale(generation) || shared.is_interrupted(turn) {
                            continue;
                        }
                        let completed = shared.note_audio(&audio, Instant::now(), cancel);
                        if is_last {
                            match sink.finish_turn(turn, cancel) {
                                Ok(()) => {}
                                Err(Error::Cancelled) => {
                                    shared.release_assistant(turn);
                                    if cancel.is_shutdown() {
                                        return;
                                    }
                                    continue;
                                }
                                Err(err) => {
                                    shared.fail(err, cancel);
                                    return;
                                }
                            }
                            shared.release_assistant(turn);
                            shared.emit(LoopEvent::Playback { playing: false });
                            maybe_stop_after(mode, completed, cancel);
                        }
                    }
                    Err(Error::Cancelled) => {
                        shared.release_assistant(turn);
                        if cancel.is_shutdown() {
                            return;
                        }
                    }
                    Err(err) => {
                        shared.fail(err, cancel);
                        return;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn maybe_stop_after(mode: LoopMode, completed: usize, cancel: &Cancel) {
    if let LoopMode::StopAfterTurns(n) = mode {
        if completed >= n {
            cancel.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{
        scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad, ScriptedStt,
    };
    #[cfg(feature = "device-audio")]
    use crate::s2s::{DuplexS2s, S2sEvent};
    use crate::types::TurnId;

    struct PartialStt {
        turn: Option<TurnId>,
        frames: usize,
    }

    impl Stt for PartialStt {
        fn name(&self) -> &'static str {
            "partial-fixture"
        }

        fn transcribe(&mut self, utterance: &Utterance, _cancel: &Cancel) -> Result<Transcript> {
            Ok(Transcript {
                turn: utterance.turn,
                text: "final transcript".into(),
                language: "en".into(),
            })
        }

        fn supports_partials(&self) -> bool {
            true
        }

        fn start_turn(&mut self, turn: TurnId, _cancel: &Cancel) -> Result<()> {
            self.turn = Some(turn);
            self.frames = 0;
            Ok(())
        }

        fn push_frame(&mut self, _frame: &AudioFrame, _cancel: &Cancel) -> Result<Option<String>> {
            self.frames += 1;
            Ok((self.frames == 1).then(|| "partial transcript".into()))
        }
    }

    #[test]
    fn runtime_controls_toggle_independently() {
        let controls = RuntimeControls::new(false);
        assert!(!controls.barge_in());
        assert!(!controls.mic_muted());
        assert!(controls.capture_open());
        assert!(controls.toggle_barge_in());
        assert!(controls.toggle_mic_muted());
        assert!(controls.barge_in());
        assert!(controls.mic_muted());
        assert!(!controls.capture_open());
        assert!(!controls.toggle_mic_muted());
        assert!(!controls.mic_muted());
        assert!(controls.capture_open());
        controls.set_vad_paused(true);
        assert!(!controls.capture_open());
        controls.set_vad_paused(false);
        assert!(controls.capture_open());
    }

    fn controls_with_clock(mic_mute_ms: u32, exit_ms: u32) -> RuntimeControls {
        RuntimeControls::with_auto_timeout_clock(false, mic_mute_ms, exit_ms, IdleClock::manual(0))
    }

    #[test]
    fn auto_timeout_running_exits_after_one_second() {
        // Timer is armed and running; exit deadline is 1s — no wall sleep.
        let controls = controls_with_clock(0, 1_000);
        controls.arm_idle();
        assert!(controls.idle_armed());
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);

        controls.clock().advance(Duration::from_millis(999));
        assert_eq!(
            controls.poll_auto_timeout(),
            AutoTimeoutAction::None,
            "999ms is still under the 1s exit deadline"
        );

        controls.clock().advance(Duration::from_millis(1));
        assert_eq!(
            controls.poll_auto_timeout(),
            AutoTimeoutAction::Exit,
            "exactly 1000ms must exit"
        );
    }

    #[test]
    fn auto_timeout_mutes_then_exits_while_idle() {
        let controls = controls_with_clock(30, 60);
        controls.arm_idle();

        controls.clock().advance(Duration::from_millis(29));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);

        controls.clock().advance(Duration::from_millis(1));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::MicMute);
        assert!(controls.mic_muted());
        // Sticky: later polls stay None until exit.
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);

        controls.clock().advance(Duration::from_millis(29));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        controls.clock().advance(Duration::from_millis(1));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::Exit);

        controls.unmute_mic();
        assert!(!controls.mic_muted());
        assert!(controls.idle_armed());
    }

    #[test]
    fn auto_timeout_exit_wins_when_both_thresholds_crossed_in_one_poll() {
        let controls = controls_with_clock(30, 60);
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(60));
        assert_eq!(
            controls.poll_auto_timeout(),
            AutoTimeoutAction::Exit,
            "skip MicMute when exit is already due"
        );
        // Mic may still be unmuted because Exit short-circuits before setting it.
        assert!(!controls.mic_muted());
    }

    #[test]
    fn auto_timeout_disarmed_during_speech_ignores_elapsed_time() {
        let controls = controls_with_clock(40, 80);
        controls.arm_idle();
        controls.disarm_idle();
        assert!(!controls.idle_armed());

        controls.clock().advance(Duration::from_secs(10));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        assert!(!controls.mic_muted());
    }

    #[test]
    fn auto_timeout_touch_resets_only_when_armed() {
        let controls = controls_with_clock(40, 0);
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(20));
        controls.touch_idle();
        controls.clock().advance(Duration::from_millis(39));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        controls.clock().advance(Duration::from_millis(1));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::MicMute);

        // Touch while disarmed is a no-op (does not re-arm).
        let controls = controls_with_clock(10, 0);
        controls.disarm_idle();
        controls.touch_idle();
        assert!(!controls.idle_armed());
        controls.clock().advance(Duration::from_millis(50));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
    }

    #[test]
    fn auto_timeout_rearm_after_speech_starts_fresh_window() {
        let controls = controls_with_clock(50, 100);
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(40));
        // Speech starts: disarm, then later re-arm after TTS.
        controls.disarm_idle();
        controls.clock().advance(Duration::from_millis(1_000));
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(49));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        controls.clock().advance(Duration::from_millis(1));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::MicMute);
    }

    #[test]
    fn auto_timeout_mic_mute_only_when_exit_disabled() {
        let controls = controls_with_clock(25, 0);
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(25));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::MicMute);
        controls.clock().advance(Duration::from_secs(60));
        assert_eq!(
            controls.poll_auto_timeout(),
            AutoTimeoutAction::None,
            "exit disabled: stay muted forever without Exit"
        );
        assert!(controls.mic_muted());
    }

    #[test]
    fn auto_timeout_exit_only_when_mic_mute_disabled() {
        let controls = controls_with_clock(0, 40);
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(39));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        assert!(!controls.mic_muted());
        controls.clock().advance(Duration::from_millis(1));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::Exit);
        assert!(!controls.mic_muted());
    }

    #[test]
    fn auto_timeout_zero_disables_timers() {
        let controls = controls_with_clock(0, 0);
        controls.arm_idle();
        controls.clock().advance(Duration::from_secs(3_600));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        assert!(!controls.mic_muted());
    }

    #[test]
    fn auto_timeout_unmute_clears_mute_and_restarts_window() {
        let controls = controls_with_clock(20, 100);
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(20));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::MicMute);
        assert!(controls.mic_muted());

        controls.unmute_mic();
        assert!(!controls.mic_muted());
        // Fresh window from unmute: need another full mute period.
        controls.clock().advance(Duration::from_millis(19));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        controls.clock().advance(Duration::from_millis(1));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::MicMute);
    }

    #[test]
    fn auto_timeout_double_arm_resets_elapsed() {
        let controls = controls_with_clock(50, 0);
        controls.arm_idle();
        controls.clock().advance(Duration::from_millis(40));
        controls.arm_idle(); // re-arm as if post-TTS fired again
        controls.clock().advance(Duration::from_millis(40));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        controls.clock().advance(Duration::from_millis(10));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::MicMute);
    }

    #[test]
    fn auto_timeout_poll_before_arm_is_none() {
        let controls = controls_with_clock(1, 1);
        assert!(!controls.idle_armed());
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
        controls.clock().advance(Duration::from_secs(5));
        assert_eq!(controls.poll_auto_timeout(), AutoTimeoutAction::None);
    }

    #[test]
    fn thinking_mutes_vad_even_with_barge_in() {
        let shared = Shared::new(None, None, RuntimeControls::new(true), "local", None, None);
        shared.mark_assistant(TurnId(0));
        assert!(shared.thinking.load(Ordering::SeqCst));
        assert!(shared.pause_vad.load(Ordering::SeqCst));
        assert!(!shared.controls.capture_open());

        let chunk = SynthesizedAudio {
            turn: TurnId(0),
            generation: crate::types::GenerationId(0),
            index: 0,
            samples: vec![1],
            is_last: false,
        };
        shared.note_audio(&chunk, Instant::now(), &Cancel::new());
        assert!(!shared.thinking.load(Ordering::SeqCst));
        assert!(
            !shared.pause_vad.load(Ordering::SeqCst),
            "speaking with --barge-in listens"
        );
        assert!(shared.controls.capture_open());
    }

    fn run_turns(n: usize) -> LoopReport {
        let frames = scripted_frames(n, 2, 1);
        run_loop(
            LoopConfig::default(),
            PipelineStages {
                vad: FakeVad::new(),
                stt: FakeStt,
                llm: FakeLlm::new(),
                tts: FakeTts,
                sink: CollectingSink::default(),
            },
            frames,
            Cancel::new(),
        )
        .expect("loop")
    }

    #[test]
    fn one_turn_echoes_in_order() {
        let report = run_turns(1);
        assert_eq!(report.turns.len(), 1);
        assert_eq!(report.turns[0].id, TurnId(0));
        assert_eq!(report.turns[0].user_text, "turn-000");
        assert_eq!(report.turns[0].assistant_text, "echo:turn-000");
        assert!(report.turns[0].token_count > 0);
        assert_eq!(report.turns[0].token_count, report.turns[0].audio_chunks);
        let timings = report.turns[0].timings;
        assert!(timings.total >= timings.stt);
        assert!(timings.total >= timings.ttft);
        assert!(timings.total >= timings.ttfb);
        assert_eq!(report.tasks_exited, 6);
        assert_eq!(report.tasks_still_running, 0);
        assert_eq!(report.skipped_turns, 0);
        assert!(report.queues.within_capacity());
    }

    #[test]
    fn streaming_stt_emits_partial_before_vad_finalizes_the_turn() {
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let dir = std::env::temp_dir().join(format!(
            "syllabix-partial-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let debug = TurnDebug::open(&dir).unwrap();
        let report = run_loop(
            LoopConfig {
                events: Some(events_tx),
                turn_debug: Some(debug.clone()),
                mode: LoopMode::StopAfterTurns(1),
                ..LoopConfig::default()
            },
            PipelineStages {
                vad: FakeVad::new(),
                stt: PartialStt {
                    turn: None,
                    frames: 0,
                },
                llm: FakeLlm::new(),
                tts: FakeTts,
                sink: CollectingSink::default(),
            },
            scripted_frames(1, 2, 1),
            Cancel::new(),
        )
        .unwrap();
        assert_eq!(report.turns.len(), 1);
        let events: Vec<_> = events_rx.try_iter().collect();
        let partial = events
            .iter()
            .position(|event| matches!(event, LoopEvent::Partial { text, .. } if text == "partial transcript"))
            .expect("partial event");
        let final_user = events
            .iter()
            .position(
                |event| matches!(event, LoopEvent::User { text, .. } if text == "final transcript"),
            )
            .expect("final user event");
        assert!(partial < final_user);
        assert!(debug
            .anchored(TurnId(0))
            .iter()
            .any(|(anchor, _)| *anchor == TimelineAnchor::SttPartial));
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Streaming STT that decodes slower than real time. The first
    /// `push_frame` stalls long enough for VAD to emit the whole turn, which
    /// models Moonshine's full-utterance re-decode falling behind: the
    /// engine, not the test, is the bottleneck.
    struct SlowPartialStt {
        turn: Option<TurnId>,
        push_calls: Arc<AtomicUsize>,
        transcribed_frames: Arc<Mutex<Option<usize>>>,
        stalled: AtomicBool,
        first_delay: Duration,
    }

    impl Stt for SlowPartialStt {
        fn name(&self) -> &'static str {
            "slow-partial-fixture"
        }

        fn transcribe(&mut self, utterance: &Utterance, _cancel: &Cancel) -> Result<Transcript> {
            *self.transcribed_frames.lock().expect("transcribed frames") =
                Some(utterance.frames.len());
            Ok(Transcript {
                turn: utterance.turn,
                text: "slow final".into(),
                language: "en".into(),
            })
        }

        fn supports_partials(&self) -> bool {
            true
        }

        fn start_turn(&mut self, turn: TurnId, _cancel: &Cancel) -> Result<()> {
            self.turn = Some(turn);
            Ok(())
        }

        fn push_frame(&mut self, _frame: &AudioFrame, _cancel: &Cancel) -> Result<Option<String>> {
            self.push_calls.fetch_add(1, Ordering::SeqCst);
            if !self.stalled.swap(true, Ordering::SeqCst) {
                thread::sleep(self.first_delay);
            }
            Ok(None)
        }
    }

    #[test]
    fn slow_partial_stt_drops_partials_instead_of_stalling_vad() {
        // One turn of 40 speech frames (~1s of fixture capture). The first
        // partial decode stalls 3s, so all 40 frames land in the cap-16
        // partial channel while the engine is stuck: the old blocking VAD→STT
        // queue stalled VAD here and, on a live mic, dropped capture audio
        // until VAD endpointed mid-speech regardless of `end_silence_ms`.
        let push_calls = Arc::new(AtomicUsize::new(0));
        let transcribed_frames = Arc::new(Mutex::new(None));
        let report = run_loop(
            LoopConfig {
                mode: LoopMode::StopAfterTurns(1),
                ..LoopConfig::default()
            },
            PipelineStages {
                vad: FakeVad::new(),
                stt: SlowPartialStt {
                    turn: None,
                    push_calls: Arc::clone(&push_calls),
                    transcribed_frames: Arc::clone(&transcribed_frames),
                    stalled: AtomicBool::new(false),
                    first_delay: Duration::from_secs(3),
                },
                llm: FakeLlm::new(),
                tts: FakeTts,
                sink: CollectingSink::default(),
            },
            scripted_frames(1, 40, 2),
            Cancel::new(),
        )
        .expect("loop");
        assert_eq!(report.turns.len(), 1);
        assert_eq!(report.turns[0].user_text, "slow final");
        // Endpoint integrity: FakeVad closes on the first silence frame, so
        // the canonical utterance must hold every speech frame even though
        // most provisional frames were shed under load.
        assert_eq!(
            *transcribed_frames.lock().expect("transcribed frames"),
            Some(40)
        );
        // Load shedding, not stalling: VAD dropped (not blocked on) the
        // frames the slow engine could not consume, and decoded only a
        // fraction of the 40 partial frames.
        assert!(
            report.dropped_stt_partials > 0,
            "slow engine must shed partial frames"
        );
        let calls = push_calls.load(Ordering::SeqCst);
        assert!(
            calls < 40,
            "slow engine must shed load, decoded {calls}/40 partials"
        );
        assert!(report.queues.within_capacity());
    }

    #[test]
    fn loop_emits_user_assistant_and_timing_events() {
        let (tx, rx) = std::sync::mpsc::channel();
        let report = run_loop(
            LoopConfig {
                events: Some(tx),
                ..LoopConfig::default()
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
        .expect("loop");
        assert_eq!(report.turns.len(), 1);
        let events: Vec<LoopEvent> = rx.try_iter().collect();
        assert!(
            events.iter().any(|event| matches!(
                event,
                LoopEvent::User {
                    text,
                    ..
                } if text == "turn-000"
            )),
            "{events:?}"
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, LoopEvent::Assistant { .. })),
            "{events:?}"
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, LoopEvent::Timings { .. })),
            "{events:?}"
        );
    }

    #[test]
    fn six_turns_config_stops_after_six() {
        assert_eq!(LoopConfig::six_turns().mode, LoopMode::StopAfterTurns(6));
    }

    #[test]
    fn shutdown_before_capture_exits_without_turns() {
        let cancel = Cancel::new();
        cancel.shutdown();
        let report = run_loop(
            LoopConfig::default(),
            PipelineStages {
                vad: FakeVad::new(),
                stt: FakeStt,
                llm: FakeLlm::new(),
                tts: FakeTts,
                sink: CollectingSink::default(),
            },
            scripted_frames(3, 2, 1),
            cancel,
        )
        .expect("cancelled loop");
        assert!(report.cancelled);
        assert_eq!(report.tasks_still_running, 0);
        assert_eq!(report.tasks_exited, 6);
        assert!(report.turns.is_empty());
    }

    #[test]
    fn default_loop_config_runs_until_input_ends() {
        assert_eq!(LoopConfig::default().mode, LoopMode::UntilInputEnds);
    }

    #[test]
    fn blank_stt_is_empty_or_whitespace() {
        assert!(is_blank_stt(""));
        assert!(is_blank_stt(" \n\t"));
        assert!(!is_blank_stt("hello"));
        assert!(!is_blank_stt(" a "));
    }

    #[test]
    fn empty_stt_does_not_start_llm_or_tts() {
        let llm = FakeLlm::new();
        let calls = llm.call_log();
        let (tx, rx) = std::sync::mpsc::channel();
        let report = run_loop(
            LoopConfig {
                events: Some(tx),
                ..LoopConfig::default()
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
        .expect("empty stt");
        assert!(
            report.turns.is_empty(),
            "blank STT must not complete a spoken turn"
        );
        assert!(report.skipped_turns >= 1);
        assert!(calls.lock().expect("llm log").is_empty());
        let events: Vec<LoopEvent> = rx.try_iter().collect();
        assert!(
            events.iter().all(|event| !matches!(
                event,
                LoopEvent::User { .. } | LoopEvent::Assistant { .. } | LoopEvent::Timings { .. }
            )),
            "{events:?}"
        );
    }

    #[test]
    fn whitespace_stt_does_not_start_llm_or_tts() {
        let llm = FakeLlm::new();
        let calls = llm.call_log();
        let report = run_loop(
            LoopConfig::default(),
            PipelineStages {
                vad: FakeVad::new(),
                stt: ScriptedStt::new([" \n\t"]),
                llm,
                tts: FakeTts,
                sink: CollectingSink::default(),
            },
            scripted_frames(1, 2, 1),
            Cancel::new(),
        )
        .expect("whitespace stt");
        assert!(report.turns.is_empty());
        assert!(report.skipped_turns >= 1);
        assert!(calls.lock().expect("llm log").is_empty());
    }

    #[test]
    fn final_audio_waits_for_sink_drain_before_releasing_default_vad() {
        use std::sync::atomic::AtomicBool;

        struct DrainSink {
            played_final: Arc<AtomicBool>,
            finished: Arc<AtomicBool>,
        }

        impl crate::providers::AudioSink for DrainSink {
            fn play(&mut self, audio: SynthesizedAudio, _cancel: &Cancel) -> Result<()> {
                if audio.is_last {
                    self.played_final.store(true, Ordering::SeqCst);
                }
                Ok(())
            }

            fn finish_turn(&mut self, _turn: TurnId, _cancel: &Cancel) -> Result<()> {
                assert!(
                    self.played_final.load(Ordering::SeqCst),
                    "the final PCM must enter the sink before its drain wait"
                );
                self.finished.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let played_final = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let report = run_loop(
            LoopConfig {
                mode: LoopMode::StopAfterTurns(1),
                ..LoopConfig::default()
            },
            PipelineStages {
                vad: FakeVad::new(),
                stt: FakeStt,
                llm: FakeLlm::new(),
                tts: FakeTts,
                sink: DrainSink {
                    played_final: Arc::clone(&played_final),
                    finished: Arc::clone(&finished),
                },
            },
            scripted_frames(1, 2, 1),
            Cancel::new(),
        )
        .expect("draining loop");

        assert_eq!(report.turns.len(), 1);
        assert!(finished.load(Ordering::SeqCst));
    }

    /// Wraps [`FakeLlm`] and reports provider facts the way live engines do.
    struct MetaLlm {
        inner: FakeLlm,
        tool_events: Vec<crate::types::ToolTurnEvent>,
    }

    impl MetaLlm {
        fn new() -> Self {
            Self {
                inner: FakeLlm::new(),
                tool_events: vec![crate::types::ToolTurnEvent {
                    kind: "result".into(),
                    name: "web_fetch".into(),
                    call_id: "call-42".into(),
                    arguments: r#"{"url":"https://example.test"}"#.into(),
                    content: "fixture result".into(),
                }],
            }
        }
    }

    impl crate::providers::Llm for MetaLlm {
        fn name(&self) -> &'static str {
            "meta-fake"
        }

        fn debug_meta(&self) -> Option<crate::types::LlmDebugMeta> {
            Some(crate::types::LlmDebugMeta {
                provider: "online".into(),
                model: "gpt-test".into(),
                endpoint: "https://mock.example/v1/chat/completions".into(),
                request_id: "req-42".into(),
            })
        }

        fn take_tool_events(&mut self) -> Vec<crate::types::ToolTurnEvent> {
            std::mem::take(&mut self.tool_events)
        }

        fn generate(
            &mut self,
            history: &[HistoryTurn],
            user: &Transcript,
            cancel: &Cancel,
            on_token: &mut dyn FnMut(TokenChunk) -> Result<()>,
        ) -> Result<()> {
            self.inner.generate(history, user, cancel, on_token)
        }
    }

    #[test]
    fn turn_debug_sidecar_records_llm_provider_facts() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-pipeline-meta-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let debug = TurnDebug::open(&dir).expect("open");
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        run_loop(
            LoopConfig {
                turn_debug: Some(debug),
                mode: LoopMode::StopAfterTurns(1),
                events: Some(events_tx),
                ..LoopConfig::default()
            },
            PipelineStages {
                vad: FakeVad::new(),
                stt: FakeStt,
                llm: MetaLlm::new(),
                tts: FakeTts,
                sink: CollectingSink::default(),
            },
            scripted_frames(1, 2, 1),
            Cancel::new(),
        )
        .expect("loop");
        let json =
            std::fs::read_to_string(dir.join("turn-000").join("turn.json")).expect("sidecar");
        assert!(json.contains("\"llm_provider\": \"online\""), "{json}");
        assert!(json.contains("\"llm_model\": \"gpt-test\""), "{json}");
        assert!(
            json.contains("\"llm_endpoint\": \"https://mock.example/v1/chat/completions\""),
            "{json}"
        );
        assert!(json.contains("\"llm_request_id\": \"req-42\""), "{json}");
        assert!(
            json.contains("\"tool_events\": [{\"kind\":\"result\""),
            "{json}"
        );
        assert!(json.contains("\"call_id\":\"call-42\""), "{json}");
        assert!(events_rx.try_iter().any(|event| matches!(
            event,
            LoopEvent::Tool { event, .. }
                if event.name == "web_fetch" && event.content == "fixture result"
        )));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn completed_turn_timeline_is_complete_and_stage_ordered() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-pipeline-timeline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let debug = TurnDebug::open(&dir).expect("open");
        let watch = crate::turn_debug::PlaybackWatch::new(debug.clone());
        run_loop(
            LoopConfig {
                turn_debug: Some(debug.clone()),
                mode: LoopMode::StopAfterTurns(1),
                ..LoopConfig::default()
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
        .expect("loop");

        let anchored = debug.anchored(TurnId(0));
        let offset = |anchor: TimelineAnchor| {
            anchored
                .iter()
                .find(|(name, _)| *name == anchor)
                .map(|(_, at)| *at)
                .unwrap_or_else(|| panic!("missing anchor {anchor:?}"))
        };
        // Every stage boundary was captured.
        // Final-only STT leaves the streaming-only partial anchor absent.
        assert_eq!(anchored.len(), TimelineAnchor::ALL.len() - 1);
        assert!(anchored
            .iter()
            .all(|(anchor, _)| *anchor != TimelineAnchor::SttPartial));
        assert_eq!(offset(TimelineAnchor::SpeechStart), Duration::ZERO);
        // Stage order; LLM/TTS may legitimately overlap, so only true
        // happens-before edges are asserted.
        assert!(offset(TimelineAnchor::SpeechStart) <= offset(TimelineAnchor::SpeechEnd));
        assert!(offset(TimelineAnchor::SpeechEnd) <= offset(TimelineAnchor::SttQueued));
        assert!(offset(TimelineAnchor::SttQueued) <= offset(TimelineAnchor::SttDone));
        assert!(offset(TimelineAnchor::SttDone) <= offset(TimelineAnchor::LlmStart));
        assert!(offset(TimelineAnchor::LlmStart) <= offset(TimelineAnchor::LlmFirstToken));
        assert!(offset(TimelineAnchor::LlmFirstToken) <= offset(TimelineAnchor::TtsFirstPcm));
        assert!(offset(TimelineAnchor::LlmLastToken) <= offset(TimelineAnchor::TtsLastPcm));
        assert!(offset(TimelineAnchor::TtsFirstPcm) <= offset(TimelineAnchor::PlaybackFirst));
        assert!(offset(TimelineAnchor::PlaybackFirst) <= offset(TimelineAnchor::PlaybackDone));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn blank_stt_turn_stops_the_timeline_after_stt() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-pipeline-blank-timeline-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let debug = TurnDebug::open(&dir).expect("open");
        run_loop(
            LoopConfig {
                turn_debug: Some(debug.clone()),
                mode: LoopMode::StopAfterTurns(1),
                ..LoopConfig::default()
            },
            PipelineStages {
                vad: FakeVad::new(),
                stt: ScriptedStt::new([""]),
                llm: FakeLlm::new(),
                tts: FakeTts,
                sink: CollectingSink::default(),
            },
            scripted_frames(1, 2, 1),
            Cancel::new(),
        )
        .expect("loop");
        let names: Vec<_> = debug.anchored(TurnId(0)).iter().map(|(a, _)| *a).collect();
        assert_eq!(
            names,
            vec![
                TimelineAnchor::SpeechStart,
                TimelineAnchor::SpeechEnd,
                TimelineAnchor::SttQueued,
                TimelineAnchor::SttDone,
            ],
            "empty STT must not open the LLM/TTS/sink stages"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(feature = "device-audio")]
    #[derive(Clone, Default)]
    struct SharedCollectingSink {
        chunks: Arc<Mutex<Vec<SynthesizedAudio>>>,
        interrupted: Arc<AtomicUsize>,
    }

    #[cfg(feature = "device-audio")]
    impl AudioSink for SharedCollectingSink {
        fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            self.chunks.lock().expect("chunks").push(audio);
            Ok(())
        }

        fn interrupt(&mut self) {
            self.interrupted.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[cfg(feature = "device-audio")]
    fn duplex_frames(count: usize, sample: i16) -> Vec<AudioFrame> {
        (0..count)
            .map(|seq| {
                AudioFrame::new(
                    seq as u64,
                    crate::types::DEFAULT_SAMPLE_RATE_HZ,
                    crate::types::DEFAULT_CHANNELS,
                    vec![sample; crate::types::FRAME_SAMPLES],
                )
                .expect("duplex fixture frame")
            })
            .collect()
    }

    #[cfg(feature = "device-audio")]
    struct ImmediateCapture {
        frames: std::vec::IntoIter<AudioFrame>,
    }

    #[cfg(feature = "device-audio")]
    impl AudioCapture for ImmediateCapture {
        fn name(&self) -> &'static str {
            "duplex-fixture"
        }

        fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            Ok(self.frames.next())
        }
    }

    #[cfg(feature = "device-audio")]
    struct ScriptedS2s {
        events: std::sync::Mutex<std::collections::VecDeque<S2sEvent>>,
        pushes: std::sync::Mutex<Vec<AudioFrame>>,
    }

    #[cfg(feature = "device-audio")]
    impl ScriptedS2s {
        fn events(events: impl Into<Vec<S2sEvent>>) -> Self {
            Self {
                events: std::sync::Mutex::new(std::collections::VecDeque::from(events.into())),
                pushes: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn push_event(&self, event: S2sEvent) {
            self.events.lock().expect("events").push_back(event);
        }
    }

    #[cfg(feature = "device-audio")]
    impl DuplexS2s for Arc<ScriptedS2s> {
        fn name(&self) -> &'static str {
            DuplexS2s::name(&**self)
        }

        fn push(&self, frame: AudioFrame, cancel: &Cancel) -> Result<()> {
            DuplexS2s::push(&**self, frame, cancel)
        }

        fn poll(&self, cancel: &Cancel) -> Result<S2sEvent> {
            DuplexS2s::poll(&**self, cancel)
        }
    }

    #[cfg(feature = "device-audio")]
    impl DuplexS2s for ScriptedS2s {
        fn name(&self) -> &'static str {
            "scripted-s2s"
        }

        fn push(&self, frame: AudioFrame, cancel: &Cancel) -> Result<()> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            self.pushes.lock().expect("pushes").push(frame);
            Ok(())
        }

        fn poll(&self, cancel: &Cancel) -> Result<S2sEvent> {
            loop {
                if cancel.is_shutdown() {
                    return Err(Error::Cancelled);
                }
                if let Some(event) = self.events.lock().expect("events").pop_front() {
                    return Ok(event);
                }
                thread::sleep(POLL);
            }
        }
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_loop_echoes_fake_s2s_audio_to_the_sink() {
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let sink = SharedCollectingSink::default();
        let chunks = Arc::clone(&sink.chunks);
        let interrupted = Arc::clone(&sink.interrupted);
        let report = super::run_duplex_loop(
            LoopConfig {
                mode: LoopMode::StopAfterTurns(2),
                events: Some(events_tx),
                ..LoopConfig::default()
            },
            RepeatingCapture {
                sample: 1_200,
                seq: 0,
            },
            sink,
            crate::s2s::FakeS2s::new(Duration::from_millis(5)),
            Cancel::new(),
        )
        .expect("duplex loop");
        assert_eq!(report.tasks_exited, 3);
        assert_eq!(report.tasks_still_running, 0);
        assert!(report.turns.is_empty());
        assert_eq!(report.skipped_turns, 0);
        assert_eq!(interrupted.load(Ordering::SeqCst), 0);
        let played = chunks.lock().expect("chunks").clone();
        assert!(
            played.len() >= 2,
            "FakeS2s should echo capture into the sink, got {}",
            played.len()
        );
        assert!(
            played
                .iter()
                .any(|chunk| chunk.samples.iter().any(|s| *s != 0)),
            "echoed audio should keep capture energy, got {played:?}"
        );
        let events: Vec<_> = events_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .any(|event| matches!(event, LoopEvent::Playback { playing: true })),
            "expected playback, got {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, LoopEvent::Aec { .. })),
            "polled audio must feed the AEC3 far-end, got {events:?}"
        );
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, LoopEvent::User { .. })),
            "duplex must not infer user turns from VAD: {events:?}"
        );
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_mute_pushes_silence() {
        let controls = RuntimeControls::new(false);
        controls.mute_mic();
        let s2s = Arc::new(ScriptedS2s::events(vec![S2sEvent::Text(String::from(
            "hi",
        ))]));
        let report = super::run_duplex_loop(
            LoopConfig {
                mode: LoopMode::UntilInputEnds,
                controls: controls.clone(),
                ..LoopConfig::default()
            },
            ImmediateCapture {
                frames: duplex_frames(4, 9_000).into_iter(),
            },
            CollectingSink::default(),
            Arc::clone(&s2s),
            Cancel::new(),
        )
        .expect("muted duplex");
        assert_eq!(report.tasks_exited, 3);
        let pushed = s2s.pushes.lock().expect("pushes").clone();
        assert!(
            !pushed.is_empty(),
            "mute must still push so the stream keeps a clock"
        );
        assert!(
            pushed
                .iter()
                .all(|frame| frame.samples.iter().all(|s| *s == 0)),
            "muted capture must be silence, got {pushed:?}"
        );
    }

    #[cfg(feature = "device-audio")]
    struct HangUntilCancelCapture;

    #[cfg(feature = "device-audio")]
    impl AudioCapture for HangUntilCancelCapture {
        fn name(&self) -> &'static str {
            "duplex-fixture"
        }

        fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
            loop {
                if cancel.is_shutdown() {
                    return Err(Error::Cancelled);
                }
                thread::sleep(POLL);
            }
        }
    }

    #[cfg(feature = "device-audio")]
    struct RepeatingCapture {
        sample: i16,
        seq: u64,
    }

    #[cfg(feature = "device-audio")]
    impl AudioCapture for RepeatingCapture {
        fn name(&self) -> &'static str {
            "duplex-fixture"
        }

        fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            thread::sleep(POLL);
            let frame = duplex_frames(1, self.sample)
                .into_iter()
                .next()
                .expect("frame");
            let mut frame = frame;
            frame.seq = self.seq;
            self.seq += 1;
            Ok(Some(frame))
        }
    }

    #[cfg(feature = "device-audio")]
    fn played_samples(chunks: &Arc<Mutex<Vec<SynthesizedAudio>>>) -> Vec<i16> {
        chunks
            .lock()
            .expect("chunks")
            .iter()
            .flat_map(|chunk| chunk.samples.iter().copied())
            .collect()
    }

    #[cfg(feature = "device-audio")]
    fn wait_played_len(chunks: &Arc<Mutex<Vec<SynthesizedAudio>>>, min: usize) -> usize {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let n = played_samples(chunks).len();
            if n >= min {
                return n;
            }
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {min} played samples, had {n}"
            );
            thread::sleep(POLL);
        }
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_surfaces_text_as_assistant_events() {
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        super::run_duplex_loop(
            LoopConfig {
                events: Some(events_tx),
                ..LoopConfig::default()
            },
            IterCapture {
                iter: duplex_frames(4, 1).into_iter(),
            },
            CollectingSink::default(),
            ScriptedS2s::events(vec![
                S2sEvent::Text(String::from("hel")),
                S2sEvent::Text(String::from("lo")),
            ]),
            Cancel::new(),
        )
        .expect("duplex text");
        let assistant: Vec<_> = events_rx
            .try_iter()
            .filter_map(|event| match event {
                LoopEvent::Assistant {
                    turn,
                    text,
                    is_last,
                } => Some((turn, text, is_last)),
                _ => None,
            })
            .collect();
        assert_eq!(
            assistant,
            vec![
                (TurnId(0), String::from("hel"), false),
                (TurnId(0), String::from("lo"), false),
            ]
        );
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_poll_error_stops_the_loop() {
        let err = super::run_duplex_loop(
            LoopConfig::default(),
            HangUntilCancelCapture,
            CollectingSink::default(),
            ScriptedS2s::events(vec![S2sEvent::Error {
                message: String::from("boom"),
            }]),
            Cancel::new(),
        )
        .expect_err("s2s error");
        match err {
            Error::Provider { provider, message } => {
                assert_eq!(provider, "scripted-s2s");
                assert_eq!(message, "boom");
            }
            other => panic!("expected provider error, got {other}"),
        }
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_cancel_joins_workers() {
        let cancel = Cancel::new();
        cancel.shutdown();
        let report = super::run_duplex_loop(
            LoopConfig::default(),
            ImmediateCapture {
                frames: duplex_frames(4, 3).into_iter(),
            },
            CollectingSink::default(),
            crate::s2s::FakeS2s::new(Duration::from_secs(30)),
            cancel,
        )
        .expect("cancelled duplex");
        assert!(report.cancelled);
        assert_eq!(report.tasks_exited, 3);
        assert_eq!(report.tasks_still_running, 0);
    }

    #[cfg(feature = "device-audio")]
    fn duplex_pcm_frame(seq: u64, sample: i16, len: usize) -> AudioFrame {
        AudioFrame::new(
            seq,
            crate::types::DEFAULT_SAMPLE_RATE_HZ,
            crate::types::DEFAULT_CHANNELS,
            vec![sample; len],
        )
        .expect("duplex pcm")
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_far_end_feed_blocks_instead_of_overrunning() {
        let aec = super::DuplexAec::new().expect("aec");
        let cancel = Cancel::new();
        let burst = vec![8_000i16; aec.ring.capacity().saturating_mul(2)];
        aec.feed_far_end(&burst, &cancel).expect("feed");
        assert_eq!(
            aec.ring.overruns(),
            0,
            "poll-thread far-end must wait on the ring, not try_push_slice-drop"
        );
    }

    #[cfg(feature = "device-audio")]
    struct FailAfterDelaySink;

    #[cfg(feature = "device-audio")]
    impl AudioSink for FailAfterDelaySink {
        fn play(&mut self, _audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            thread::sleep(Duration::from_millis(20));
            Err(Error::Provider {
                provider: "fail-sink",
                message: "refused playback".into(),
            })
        }
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_does_not_feed_far_end_when_play_fails() {
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let start = crate::s2s::jitter_samples(crate::s2s::JITTER_START_MS);
        let err = super::run_duplex_loop(
            LoopConfig {
                events: Some(events_tx),
                ..LoopConfig::default()
            },
            HangUntilCancelCapture,
            FailAfterDelaySink,
            ScriptedS2s::events(vec![S2sEvent::Audio(duplex_pcm_frame(0, 9_000, start))]),
            Cancel::new(),
        )
        .expect_err("play failed");
        match err {
            Error::Provider { provider, .. } => assert_eq!(provider, "fail-sink"),
            other => panic!("expected fail-sink, got {other}"),
        }
        let events: Vec<_> = events_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, LoopEvent::Aec { .. })),
            "failed play must not leave a ghost far-end, got {events:?}"
        );
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_jitter_holds_until_90ms() {
        let sink = SharedCollectingSink::default();
        let chunks = Arc::clone(&sink.chunks);
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        super::run_duplex_loop(
            LoopConfig {
                events: Some(events_tx),
                ..LoopConfig::default()
            },
            ImmediateCapture {
                frames: duplex_frames(1, 1).into_iter(),
            },
            sink,
            ScriptedS2s::events(vec![S2sEvent::Audio(duplex_pcm_frame(
                0,
                9,
                crate::types::FRAME_SAMPLES,
            ))]),
            Cancel::new(),
        )
        .expect("short duplex");
        assert!(
            chunks.lock().expect("chunks").is_empty(),
            "sink must wait for 90 ms before the first play"
        );
        let events: Vec<_> = events_rx.try_iter().collect();
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, LoopEvent::Aec { .. })),
            "held audio must not feed the far-end, got {events:?}"
        );
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_jitter_plays_once_90ms_is_buffered() {
        let sink = SharedCollectingSink::default();
        let chunks = Arc::clone(&sink.chunks);
        let start = crate::s2s::jitter_samples(crate::s2s::JITTER_START_MS);
        let quantum = crate::s2s::JITTER_PULL_SAMPLES;
        super::run_duplex_loop(
            LoopConfig {
                mode: LoopMode::StopAfterTurns(2),
                ..LoopConfig::default()
            },
            HangUntilCancelCapture,
            sink,
            ScriptedS2s::events(vec![S2sEvent::Audio(duplex_pcm_frame(0, 11, start))]),
            Cancel::new(),
        )
        .expect("started duplex");
        let played = played_samples(&chunks);
        assert_eq!(played.len(), 2 * quantum);
        assert!(played.iter().all(|s| *s == 11));
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_jitter_gap_refills_to_90ms_before_playing_again() {
        let sink = SharedCollectingSink::default();
        let chunks = Arc::clone(&sink.chunks);
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let s2s = Arc::new(ScriptedS2s::events(Vec::new()));
        let cancel = Cancel::new();
        let start = crate::s2s::jitter_samples(crate::s2s::JITTER_START_MS);
        let quantum = crate::s2s::JITTER_PULL_SAMPLES;
        let before_underrun = (start / quantum) * quantum;
        let loop_s2s = Arc::clone(&s2s);
        let loop_cancel = cancel.clone();
        let handle = thread::spawn(move || {
            super::run_duplex_loop(
                LoopConfig {
                    events: Some(events_tx),
                    ..LoopConfig::default()
                },
                RepeatingCapture { sample: 0, seq: 0 },
                sink,
                loop_s2s,
                loop_cancel,
            )
        });

        s2s.push_event(S2sEvent::Audio(duplex_pcm_frame(0, 1, start)));
        wait_played_len(&chunks, before_underrun);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(played_samples(&chunks).len(), before_underrun);

        s2s.push_event(S2sEvent::Audio(duplex_pcm_frame(1, 2, 200)));
        thread::sleep(Duration::from_millis(80));
        let mid = played_samples(&chunks);
        assert_eq!(mid.len(), before_underrun);
        assert!(
            mid.iter().all(|s| *s == 1),
            "gap must not play, got {mid:?}"
        );

        s2s.push_event(S2sEvent::Audio(duplex_pcm_frame(
            2,
            3,
            start - (start % quantum) - 200,
        )));
        wait_played_len(&chunks, before_underrun + quantum);
        cancel.shutdown();
        handle.join().expect("duplex thread").expect("duplex loop");
        drop(events_rx);
    }

    #[cfg(feature = "device-audio")]
    #[test]
    fn duplex_jitter_cap_drops_oldest_before_play() {
        let sink = SharedCollectingSink::default();
        let chunks = Arc::clone(&sink.chunks);
        let (events_tx, events_rx) = std::sync::mpsc::channel();
        let start = crate::s2s::jitter_samples(crate::s2s::JITTER_START_MS);
        let cap = crate::s2s::jitter_samples(crate::s2s::JITTER_CAP_MS);
        let mut pcm = vec![7; cap + 10 - start];
        pcm.extend(vec![8; start]);
        let burst = AudioFrame::new(
            0,
            crate::types::DEFAULT_SAMPLE_RATE_HZ,
            crate::types::DEFAULT_CHANNELS,
            pcm,
        )
        .expect("burst");
        super::run_duplex_loop(
            LoopConfig {
                mode: LoopMode::StopAfterTurns(2),
                events: Some(events_tx),
                ..LoopConfig::default()
            },
            RepeatingCapture { sample: 0, seq: 0 },
            sink,
            ScriptedS2s::events(vec![S2sEvent::Audio(burst)]),
            Cancel::new(),
        )
        .expect("capped duplex");
        let played = played_samples(&chunks);
        assert!(!played.is_empty());
        assert!(
            played.iter().all(|s| *s == 8),
            "cap must drop oldest samples before play, got {played:?}"
        );
        drop(events_rx);
    }
}
