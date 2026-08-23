//! In-memory conversation loop: VAD → STT → LLM → TTS → sink with bounded queues.

use std::collections::{BTreeMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::cancel::Cancel;
use crate::defaults::{BuiltinDefaults, QueueCaps};
use crate::error::{Error, Result};
use crate::providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
use crate::queue::{bounded, BoundedSender, QueueReport};
use crate::turn_debug::{TimelineAnchor, TurnDebug};
use crate::types::{
    AudioFrame, CompletedTurn, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript, TurnId,
    TurnTimings, Utterance, VadEvent,
};

const POLL: Duration = Duration::from_millis(5);

/// Empty or whitespace Whisper text must not start LLM/TTS.
pub fn is_blank_stt(text: &str) -> bool {
    text.trim().is_empty()
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
    /// User STT text for a turn.
    User {
        /// Turn id.
        turn: TurnId,
        /// Transcript.
        text: String,
        /// Effective STT language code (configured or auto-detected).
        language: String,
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
    /// Turn finished playback; clocks for the TUI footer.
    Timings {
        /// Turn id.
        turn: TurnId,
        /// STT / TTFT / TTFB / total.
        timings: TurnTimings,
    },
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
    /// Keep VAD running during TTS and cancel playback on SpeechStart.
    pub barge_in: bool,
}

impl LoopConfig {
    /// 30-turn in-memory fake merge-gate fixture.
    pub fn thirty_turns() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::StopAfterTurns(30),
            events: None,
            turn_debug: None,
            barge_in: false,
        }
    }

    /// Native end-to-end merge-gate fixture (six real provider turns).
    pub fn six_turns() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::StopAfterTurns(6),
            events: None,
            turn_debug: None,
            barge_in: false,
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
            barge_in: false,
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
    events: Option<Sender<LoopEvent>>,
    turn_debug: Option<TurnDebug>,
    barge_in: bool,
    pause_vad: AtomicBool,
    flush_playback: AtomicBool,
    assistant_turn: Mutex<Option<TurnId>>,
    interrupted: Mutex<HashSet<TurnId>>,
    /// Sidecar facts captured from the TTS stage before the workers start.
    tts_provider: &'static str,
    tts_model: Option<String>,
}

impl Shared {
    fn new(
        events: Option<Sender<LoopEvent>>,
        turn_debug: Option<TurnDebug>,
        barge_in: bool,
        tts_provider: &'static str,
        tts_model: Option<String>,
    ) -> Arc<Self> {
        Arc::new(Self {
            turns: Mutex::new(BTreeMap::new()),
            live_tasks: Arc::new(AtomicUsize::new(0)),
            fail: Mutex::new(None),
            completed: AtomicUsize::new(0),
            skipped: AtomicUsize::new(0),
            events,
            turn_debug,
            barge_in,
            pause_vad: AtomicBool::new(false),
            flush_playback: AtomicBool::new(false),
            assistant_turn: Mutex::new(None),
            interrupted: Mutex::new(HashSet::new()),
            tts_provider,
            tts_model,
        })
    }

    fn emit(&self, event: LoopEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(event);
        }
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

    fn mark_assistant(&self, turn: TurnId) {
        *self.assistant_turn.lock().expect("assistant turn") = Some(turn);
        if !self.barge_in {
            self.pause_vad.store(true, Ordering::SeqCst);
        }
    }

    fn release_assistant(&self, turn: TurnId) {
        let mut slot = self.assistant_turn.lock().expect("assistant turn");
        if *slot == Some(turn) {
            *slot = None;
            self.pause_vad.store(false, Ordering::SeqCst);
        }
    }

    /// Cancel in-flight LLM/TTS when `--barge-in` hears SpeechStart.
    ///
    /// Always bump the generation and flush the speaker ring. Playback can still
    /// hold the previous turn after we released `assistant_turn` (last chunk
    /// queued, ring not empty) or while `play` is blocked feeding a long sentence.
    fn interrupt_assistant(&self, cancel: &Cancel) -> bool {
        if !self.barge_in {
            return false;
        }
        let turn = {
            let mut slot = self.assistant_turn.lock().expect("assistant turn");
            slot.take()
        };
        self.pause_vad.store(false, Ordering::SeqCst);
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
        Ok(self.iter.next())
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
    let caps: QueueCaps = config.defaults.queues;
    let (frame_tx, frame_rx, frame_stats) = bounded("frames", caps.frames);
    let (utt_tx, utt_rx, utt_stats) = bounded("utterances", caps.utterances);
    let (tr_tx, tr_rx, tr_stats) = bounded("transcripts", caps.transcripts);
    let (tok_tx, tok_rx, tok_stats) = bounded("tokens", caps.tokens);
    let (aud_tx, aud_rx, aud_stats) = bounded("audio", caps.audio);

    let shared = Shared::new(
        config.events.clone(),
        config.turn_debug.clone(),
        config.barge_in,
        tts_provider,
        tts_model,
    );
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
        joins.push(spawn("syllabix-vad", live, move || {
            vad_loop(vad, frame_rx, utt_tx, &cancel, &shared)
        }));
    }
    drop(utt_tx);

    // STT
    {
        let cancel = cancel.clone();
        let shared = Arc::clone(&shared);
        let live = Arc::clone(&shared.live_tasks);
        let tr_tx = tr_tx.clone();
        joins.push(spawn("syllabix-stt", live, move || {
            stt_loop(stt, utt_rx, tr_tx, &cancel, &shared)
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
            transcripts: tr_stats.snapshot(),
            tokens: tok_stats.snapshot(),
            audio: aud_stats.snapshot(),
        },
        tasks_exited: task_count,
        tasks_still_running,
        cancelled: cancel.is_shutdown(),
        skipped_turns: shared.skipped.load(Ordering::SeqCst),
    })
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
    tx: BoundedSender<Utterance>,
    cancel: &Cancel,
    shared: &Shared,
) {
    let mut active: Option<TurnId> = None;
    let emit = |events: Vec<VadEvent>,
                tx: &BoundedSender<Utterance>,
                cancel: &Cancel,
                active: &mut Option<TurnId>,
                frame: Option<&AudioFrame>|
     -> Result<()> {
        for event in &events {
            if let VadEvent::SpeechStart { turn } = event {
                shared.interrupt_assistant(cancel);
                *active = Some(*turn);
                if let Some(debug) = &shared.turn_debug {
                    debug.start_turn(*turn);
                    debug.note_anchor(*turn, TimelineAnchor::SpeechStart, Instant::now());
                }
            }
        }
        if let (Some(debug), Some(frame), Some(turn)) = (&shared.turn_debug, frame, *active) {
            debug.note_frame(turn, &frame.samples, frame.capture_pcm.as_deref());
        }
        for event in events {
            if let VadEvent::SpeechEnd { utterance } = event {
                if let Some(debug) = &shared.turn_debug {
                    debug.note_utterance(&utterance);
                    debug.note_anchor(utterance.turn, TimelineAnchor::SpeechEnd, Instant::now());
                }
                shared.mark_assistant(utterance.turn);
                *active = None;
                tx.send_cancellable(utterance, cancel)?;
            }
        }
        Ok(())
    };

    loop {
        if cancel.is_shutdown() {
            return;
        }
        if shared.pause_vad.load(Ordering::SeqCst) {
            thread::sleep(POLL);
            continue;
        }
        match rx.recv_timeout(POLL) {
            Ok(frame) => {
                let tap = shared.turn_debug.as_ref().map(|_| frame.clone());
                match vad.push_frame(frame) {
                    Ok(events) => ignore_cancel(
                        emit(events, &tx, cancel, &mut active, tap.as_ref()),
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
                    Ok(events) => {
                        ignore_cancel(emit(events, &tx, cancel, &mut active, None), shared, cancel)
                    }
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
    rx: crate::queue::BoundedReceiver<Utterance>,
    tx: BoundedSender<Transcript>,
    cancel: &Cancel,
    shared: &Shared,
) {
    loop {
        match rx.recv_cancellable(cancel) {
            Ok(Some(utterance)) => {
                if shared.is_interrupted(utterance.turn) {
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
            Ok(None) => return,
            Err(Error::Cancelled) => return,
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
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
                // Sidecar provider facts (provider/model/endpoint/request-id)
                // land before any immediate dump (skip) can write the turn.
                if let Some(debug) = &shared.turn_debug {
                    debug.note_llm_meta(user.turn, llm.debug_meta());
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
                match sink.play(audio.clone(), cancel) {
                    Ok(()) => {
                        if shared.take_flush() {
                            sink.interrupt();
                        }
                        if cancel.is_stale(generation) || shared.is_interrupted(turn) {
                            continue;
                        }
                        let completed = shared.note_audio(&audio, Instant::now(), cancel);
                        if is_last {
                            shared.release_assistant(turn);
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
    use crate::types::TurnId;

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

    /// Wraps [`FakeLlm`] and reports provider facts the way live engines do.
    struct MetaLlm {
        inner: FakeLlm,
    }

    impl MetaLlm {
        fn new() -> Self {
            Self {
                inner: FakeLlm::new(),
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
        run_loop(
            LoopConfig {
                turn_debug: Some(debug),
                mode: LoopMode::StopAfterTurns(1),
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
        assert_eq!(anchored.len(), TimelineAnchor::ALL.len());
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
}
