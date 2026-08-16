//! In-memory conversation loop: VAD → STT → LLM → TTS → sink with bounded queues.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::cancel::Cancel;
use crate::defaults::{BuiltinDefaults, QueueCaps};
use crate::error::{Error, Result};
use crate::providers::{AudioCapture, AudioSink, Llm, Stt, Tts, Vad};
use crate::queue::{bounded, BoundedSender, QueueReport};
use crate::types::{
    AudioFrame, CompletedTurn, HistoryTurn, SynthesizedAudio, TokenChunk, Transcript, TurnId,
    Utterance, VadEvent,
};

const POLL: Duration = Duration::from_millis(5);

/// How the loop should finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Run until the frame source disconnects and every stage drains.
    UntilInputEnds,
    /// Stop after this many *completed* (played) turns.
    StopAfterTurns(usize),
}

/// Configuration for one in-memory run.
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Built-in names and queue caps.
    pub defaults: BuiltinDefaults,
    /// Stop condition.
    pub mode: LoopMode,
}

impl LoopConfig {
    /// 30-turn in-memory fake merge-gate fixture.
    pub fn thirty_turns() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::StopAfterTurns(30),
        }
    }

    /// Native end-to-end merge-gate fixture (six real provider turns).
    pub fn six_turns() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::StopAfterTurns(6),
        }
    }
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            defaults: BuiltinDefaults::v0(),
            mode: LoopMode::UntilInputEnds,
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
    /// Turns dropped after a recoverable provider error.
    pub skipped_turns: usize,
}

struct TurnAcc {
    user_text: Option<String>,
    assistant_text: String,
    token_count: usize,
    audio_chunks: usize,
    text_done: bool,
    audio_done: bool,
}

struct Shared {
    turns: Mutex<BTreeMap<TurnId, TurnAcc>>,
    live_tasks: Arc<AtomicUsize>,
    fail: Mutex<Option<Error>>,
    completed: AtomicUsize,
    skipped: AtomicUsize,
}

impl Shared {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            turns: Mutex::new(BTreeMap::new()),
            live_tasks: Arc::new(AtomicUsize::new(0)),
            fail: Mutex::new(None),
            completed: AtomicUsize::new(0),
            skipped: AtomicUsize::new(0),
        })
    }

    fn note_skip(&self) {
        self.skipped.fetch_add(1, Ordering::SeqCst);
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

    fn note_user(&self, turn: TurnId, text: String) {
        let mut map = self.turns.lock().expect("turn accumulator");
        let entry = map.entry(turn).or_insert_with(TurnAcc::new);
        entry.user_text = Some(text);
    }

    fn note_token(&self, chunk: &TokenChunk) {
        let mut map = self.turns.lock().expect("turn accumulator");
        let entry = map.entry(chunk.turn).or_insert_with(TurnAcc::new);
        entry.assistant_text.push_str(&chunk.text);
        entry.token_count += 1;
        if chunk.is_last {
            entry.text_done = true;
        }
    }

    fn note_audio(&self, chunk: &SynthesizedAudio) -> usize {
        let mut map = self.turns.lock().expect("turn accumulator");
        let entry = map.entry(chunk.turn).or_insert_with(TurnAcc::new);
        entry.audio_chunks += 1;
        if chunk.is_last {
            entry.audio_done = true;
            drop(map);
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
        }
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
    let caps: QueueCaps = config.defaults.queues;
    let (frame_tx, frame_rx, frame_stats) = bounded("frames", caps.frames);
    let (utt_tx, utt_rx, utt_stats) = bounded("utterances", caps.utterances);
    let (tr_tx, tr_rx, tr_stats) = bounded("transcripts", caps.transcripts);
    let (tok_tx, tok_rx, tok_stats) = bounded("tokens", caps.tokens);
    let (aud_tx, aud_rx, aud_stats) = bounded("audio", caps.audio);

    let shared = Shared::new();
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
    let emit =
        |events: Vec<VadEvent>, tx: &BoundedSender<Utterance>, cancel: &Cancel| -> Result<()> {
            for event in events {
                if let VadEvent::SpeechEnd { utterance } = event {
                    tx.send_cancellable(utterance, cancel)?;
                }
            }
            Ok(())
        };

    loop {
        if cancel.is_shutdown() {
            return;
        }
        match rx.recv_timeout(POLL) {
            Ok(frame) => match vad.push_frame(frame) {
                Ok(events) => ignore_cancel(emit(events, &tx, cancel), shared, cancel),
                Err(Error::Cancelled) => return,
                Err(err) => {
                    shared.fail(err, cancel);
                    return;
                }
            },
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => {
                match vad.flush() {
                    Ok(events) => ignore_cancel(emit(events, &tx, cancel), shared, cancel),
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
            Ok(Some(utterance)) => match stt.transcribe(&utterance, cancel) {
                Ok(transcript) => {
                    shared.note_user(transcript.turn, transcript.text.clone());
                    ignore_cancel(tx.send_cancellable(transcript, cancel), shared, cancel);
                }
                Err(Error::Cancelled) => {
                    if cancel.is_shutdown() {
                        return;
                    }
                }
                Err(err) if err.is_turn_recoverable() => {
                    shared.note_skip();
                }
                Err(err) => {
                    shared.fail(err, cancel);
                    return;
                }
            },
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
                let mut assistant = String::new();
                let gen_result = llm.generate(&history, &user, cancel, &mut |chunk| {
                    if cancel.is_stale(chunk.generation) {
                        return Err(Error::Cancelled);
                    }
                    assistant.push_str(&chunk.text);
                    shared.note_token(&chunk);
                    tx.send_cancellable(chunk, cancel)
                });
                match gen_result {
                    Ok(()) => {
                        history.push(HistoryTurn { user, assistant });
                    }
                    Err(Error::Cancelled) => {
                        if cancel.is_shutdown() {
                            return;
                        }
                        // Generation cancel: keep listening so the next user turn is preserved.
                    }
                    Err(err) if err.is_turn_recoverable() => {
                        shared.note_skip();
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
                if cancel.is_stale(token.generation) {
                    continue;
                }
                match tts.synthesize_chunk(&token, cancel) {
                    Ok(chunks) => {
                        for audio in chunks {
                            if cancel.is_stale(audio.generation) {
                                continue;
                            }
                            ignore_cancel(tx.send_cancellable(audio, cancel), shared, cancel);
                        }
                    }
                    Err(Error::Cancelled) => {
                        if cancel.is_shutdown() {
                            return;
                        }
                    }
                    Err(err) if err.is_turn_recoverable() => {
                        shared.note_skip();
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
        match rx.recv_cancellable(cancel) {
            Ok(Some(audio)) => {
                if cancel.is_stale(audio.generation) {
                    continue;
                }
                let is_last = audio.is_last;
                match sink.play(audio.clone(), cancel) {
                    Ok(()) => {
                        let completed = shared.note_audio(&audio);
                        if is_last {
                            maybe_stop_after(mode, completed, cancel);
                        }
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
            Ok(None) => return,
            Err(Error::Cancelled) => return,
            Err(err) => {
                shared.fail(err, cancel);
                return;
            }
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
    use crate::fake::{scripted_frames, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad};
    use crate::types::TurnId;

    fn run_turns(n: usize) -> LoopReport {
        let frames = scripted_frames(n, 2, 1);
        run_loop(
            LoopConfig {
                defaults: BuiltinDefaults::v0(),
                mode: LoopMode::UntilInputEnds,
            },
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
        assert_eq!(report.tasks_exited, 6);
        assert_eq!(report.tasks_still_running, 0);
        assert_eq!(report.skipped_turns, 0);
        assert!(report.queues.within_capacity());
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
            LoopConfig {
                defaults: BuiltinDefaults::v0(),
                mode: LoopMode::UntilInputEnds,
            },
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
}
