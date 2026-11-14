//! Injected caller audio and a recording speaker for Riley voice sessions
//! (#329 phase A2). No microphone or speaker is opened.
//!
//! [`InjectedCapture`] feeds 16 kHz mono caller turns into the pipeline as
//! [`AudioFrame`]s. Before each turn it waits for the agent: a normal turn
//! waits until the previous reply finished playing and the loop is
//! listening again; a barge-in turn starts as soon as the reply begins
//! playing. [`RecordingSink`] collects agent audio, optionally paced in real
//! time so barge-in can cut a reply short.
//!
//! Turns come from a [`CallerScript`]: a fixed list ([`FixedCaller`]) or,
//! in phase A3, the simulated caller in [`crate::caller`], which decides
//! each line after hearing the agent.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use syllabix_core::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES};
use syllabix_core::{
    AudioCapture, AudioFrame, AudioSink, Cancel, Error, Result, RuntimeControls, SynthesizedAudio,
    TurnId,
};

use crate::session::{Line, LiveLog};

/// One caller utterance: 16 kHz mono PCM16.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallerTurn {
    /// Speech samples. Trailing silence is added by [`InjectedCapture`].
    pub audio: Vec<i16>,
    /// Start while the agent's previous reply is still playing.
    pub barge_in: bool,
}

/// Source of caller turns for [`InjectedCapture`].
pub trait CallerScript: Send {
    /// `None` when the caller has nothing more to say; otherwise whether the
    /// next turn barges in.
    fn peek(&self) -> Option<bool>;

    /// Next turn, given the call so far. `Ok(None)` hangs up; `Err` ends the
    /// call and is reported as a session error.
    fn next_turn(&mut self, transcript: &[Line])
        -> std::result::Result<Option<CallerTurn>, String>;
}

/// A fixed list of turns (A2 fixtures and tests).
pub struct FixedCaller(VecDeque<CallerTurn>);

impl FixedCaller {
    /// Play `turns` in order.
    pub fn new(turns: Vec<CallerTurn>) -> Self {
        Self(turns.into())
    }
}

impl CallerScript for FixedCaller {
    fn peek(&self) -> Option<bool> {
        self.0.front().map(|turn| turn.barge_in)
    }

    fn next_turn(
        &mut self,
        _transcript: &[Line],
    ) -> std::result::Result<Option<CallerTurn>, String> {
        Ok(self.0.pop_front())
    }
}

/// Playback facts shared by the sink and the capture.
#[derive(Debug, Default)]
pub struct RenderState {
    /// Agent turns whose first chunk reached the speaker.
    pub started: usize,
    /// Agent turns whose last chunk finished playing.
    pub finished: usize,
    /// Times the loop flushed playback for barge-in.
    pub interrupted: usize,
    /// Agent samples played (after barge-in cuts).
    pub samples: usize,
    /// Set when a caller turn gave up waiting for the agent.
    pub wait_timed_out: bool,
    /// Set when the caller script failed.
    pub caller_error: Option<String>,
    current: Option<TurnId>,
}

/// [`RenderState`] behind a mutex plus a wake-up for waiting capture.
#[derive(Debug, Clone, Default)]
pub struct Render(Arc<(Mutex<RenderState>, Condvar)>);

impl Render {
    /// Fresh state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Copy of the counters.
    pub fn snapshot(&self) -> (usize, usize, usize, usize, bool) {
        let state = self.0 .0.lock().expect("render state");
        (
            state.started,
            state.finished,
            state.interrupted,
            state.samples,
            state.wait_timed_out,
        )
    }

    /// Why the caller script ended the call, if it failed.
    pub fn caller_error(&self) -> Option<String> {
        self.0 .0.lock().expect("render state").caller_error.clone()
    }

    fn update(&self, change: impl FnOnce(&mut RenderState)) {
        let (lock, wake) = &*self.0;
        change(&mut lock.lock().expect("render state"));
        wake.notify_all();
    }

    /// Wait until `ready` holds, `cancel` shuts down, or `timeout` passes.
    fn wait(
        &self,
        timeout: Duration,
        cancel: &Cancel,
        ready: impl Fn(&RenderState) -> bool,
    ) -> bool {
        let (lock, wake) = &*self.0;
        let deadline = Instant::now() + timeout;
        let mut state = lock.lock().expect("render state");
        loop {
            if ready(&state) {
                return true;
            }
            let now = Instant::now();
            if cancel.is_shutdown() || now >= deadline {
                return false;
            }
            let slice = (deadline - now).min(Duration::from_millis(10));
            state = wake.wait_timeout(state, slice).expect("render state").0;
        }
    }
}

/// Fixture microphone for one session.
pub struct InjectedCapture {
    script: Box<dyn CallerScript>,
    live: Option<LiveLog>,
    frames: VecDeque<AudioFrame>,
    controls: RuntimeControls,
    render: Render,
    frame_pause: Duration,
    trailing_silence_frames: usize,
    wait_timeout: Duration,
    seq: u64,
    sent: usize,
    /// `(started, finished)` when the previous caller turn was queued.
    mark: (usize, usize),
}

impl InjectedCapture {
    /// `frame_pause` is the delay per 32 ms frame (32 ms = real time).
    pub fn new(
        turns: Vec<CallerTurn>,
        controls: RuntimeControls,
        render: Render,
        frame_pause: Duration,
        trailing_silence_frames: usize,
        wait_timeout: Duration,
    ) -> Self {
        Self::with_script(
            Box::new(FixedCaller::new(turns)),
            None,
            controls,
            render,
            frame_pause,
            trailing_silence_frames,
            wait_timeout,
        )
    }

    /// Like [`InjectedCapture::new`], but turns come from `script`, which
    /// sees the call so far through `live`.
    pub fn with_script(
        script: Box<dyn CallerScript>,
        live: Option<LiveLog>,
        controls: RuntimeControls,
        render: Render,
        frame_pause: Duration,
        trailing_silence_frames: usize,
        wait_timeout: Duration,
    ) -> Self {
        Self {
            script,
            live,
            frames: VecDeque::new(),
            controls,
            render,
            frame_pause,
            trailing_silence_frames,
            wait_timeout,
            seq: 0,
            sent: 0,
            mark: (0, 0),
        }
    }

    fn frame(&mut self, samples: Vec<i16>) -> Result<AudioFrame> {
        let frame = AudioFrame::new(self.seq, DEFAULT_SAMPLE_RATE_HZ, DEFAULT_CHANNELS, samples)?;
        self.seq += 1;
        Ok(frame)
    }

    /// Wait for the agent, then queue the next turn's frames. `false` ends input.
    fn queue_next(&mut self, cancel: &Cancel) -> Result<bool> {
        let Some(barge_in) = self.script.peek() else {
            return Ok(false);
        };
        let (started, finished) = self.mark;
        let first = self.sent == 0;
        let controls = self.controls.clone();
        let ready = self.render.wait(self.wait_timeout, cancel, |state| {
            let agent_ready = first
                || if barge_in {
                    state.started > started
                } else {
                    state.finished > finished
                };
            agent_ready && controls.capture_open()
        });
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if !ready {
            self.render.update(|state| state.wait_timed_out = true);
            return Ok(false);
        }
        let transcript = self.live.as_ref().map(LiveLog::lines).unwrap_or_default();
        let turn = match self.script.next_turn(&transcript) {
            Ok(Some(turn)) => turn,
            Ok(None) => return Ok(false),
            Err(message) => {
                self.render
                    .update(|state| state.caller_error = Some(message));
                return Ok(false);
            }
        };
        let mut audio = turn.audio;
        if audio.is_empty() {
            audio.push(0);
        }
        let padded = audio.len().div_ceil(FRAME_SAMPLES) * FRAME_SAMPLES;
        audio.resize(padded, 0);
        for chunk in audio.chunks(FRAME_SAMPLES) {
            let frame = self.frame(chunk.to_vec())?;
            self.frames.push_back(frame);
        }
        for _ in 0..self.trailing_silence_frames {
            let frame = self.frame(vec![0; FRAME_SAMPLES])?;
            self.frames.push_back(frame);
        }
        let (now_started, now_finished, ..) = self.render.snapshot();
        self.mark = (now_started, now_finished);
        self.sent += 1;
        Ok(true)
    }
}

impl AudioCapture for InjectedCapture {
    fn name(&self) -> &'static str {
        "riley-injected"
    }

    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if self.frames.is_empty() && !self.queue_next(cancel)? {
            return Ok(None);
        }
        std::thread::sleep(self.frame_pause);
        Ok(self.frames.pop_front())
    }
}

/// Speaker stand-in: records agent audio and reports playback edges.
pub struct RecordingSink {
    render: Render,
    realtime: bool,
}

impl RecordingSink {
    /// `realtime` sleeps for each chunk's duration so barge-in can cut it.
    pub fn new(render: Render, realtime: bool) -> Self {
        Self { render, realtime }
    }
}

impl AudioSink for RecordingSink {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if cancel.is_stale(audio.generation) {
            return Ok(());
        }
        self.render.update(|state| {
            if state.current != Some(audio.turn) {
                state.current = Some(audio.turn);
                state.started += 1;
            }
        });
        if self.realtime {
            let micros = audio.samples.len() as u64 * 1_000_000 / u64::from(DEFAULT_SAMPLE_RATE_HZ);
            let end = Instant::now() + Duration::from_micros(micros);
            while Instant::now() < end {
                if cancel.is_shutdown() || cancel.is_stale(audio.generation) {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let is_last = audio.is_last;
        let samples = audio.samples.len();
        self.render.update(|state| {
            state.samples += samples;
            if is_last {
                state.finished += 1;
            }
        });
        Ok(())
    }

    fn interrupt(&mut self) {
        self.render.update(|state| state.interrupted += 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syllabix_core::types::GenerationId;

    fn capture(turns: Vec<CallerTurn>, render: &Render, wait: Duration) -> InjectedCapture {
        InjectedCapture::new(
            turns,
            RuntimeControls::with_auto_timeout(false, 0, 0),
            render.clone(),
            Duration::ZERO,
            1,
            wait,
        )
    }

    fn audio(
        turn: u64,
        generation: GenerationId,
        samples: usize,
        is_last: bool,
    ) -> SynthesizedAudio {
        SynthesizedAudio {
            turn: TurnId(turn),
            generation,
            index: 0,
            samples: vec![7; samples],
            is_last,
        }
    }

    #[test]
    fn capture_pads_turns_to_whole_frames_then_waits_for_the_agent() {
        let render = Render::new();
        let turns = vec![
            CallerTurn {
                audio: Vec::new(),
                barge_in: false,
            },
            CallerTurn {
                audio: vec![5; 3],
                barge_in: false,
            },
        ];
        let mut mic = capture(turns, &render, Duration::from_millis(20));
        let cancel = Cancel::new();
        assert_eq!(mic.name(), "riley-injected");
        let speech = mic.next_frame(&cancel).unwrap().unwrap();
        assert_eq!(speech.samples.len(), FRAME_SAMPLES);
        let silence = mic.next_frame(&cancel).unwrap().unwrap();
        assert!(!silence.has_energy());
        assert_eq!((speech.seq, silence.seq), (0, 1));
        // The agent never answered turn 0, so turn 1 gives up and input ends.
        assert!(mic.next_frame(&cancel).unwrap().is_none());
        assert!(render.snapshot().4, "wait_timed_out");
    }

    #[test]
    fn capture_stops_on_shutdown_and_when_turns_run_out() {
        let render = Render::new();
        let cancel = Cancel::new();
        assert!(capture(Vec::new(), &render, Duration::ZERO)
            .next_frame(&cancel)
            .unwrap()
            .is_none());
        cancel.shutdown();
        let turns = vec![CallerTurn {
            audio: vec![1; 4],
            barge_in: true,
        }];
        assert!(matches!(
            capture(turns, &render, Duration::ZERO).next_frame(&cancel),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn sink_counts_turns_and_skips_stale_or_cut_audio() {
        let render = Render::new();
        let mut sink = RecordingSink::new(render.clone(), false);
        let cancel = Cancel::new();
        let generation = cancel.generation();
        sink.play(audio(0, generation, 10, false), &cancel).unwrap();
        sink.play(audio(0, generation, 10, true), &cancel).unwrap();
        sink.interrupt();
        assert_eq!(render.snapshot(), (1, 1, 1, 20, false));

        // Real-time playback stops early when barge-in bumps the generation.
        let mut paced = RecordingSink::new(render.clone(), true);
        let barge = cancel.clone();
        let cutter = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            barge.cancel_generation();
        });
        paced
            .play(audio(1, generation, 16_000, true), &cancel)
            .unwrap();
        cutter.join().unwrap();
        assert_eq!(render.snapshot(), (2, 1, 1, 20, false));

        // Stale chunks are ignored; shutdown is an error.
        paced.play(audio(2, generation, 10, true), &cancel).unwrap();
        assert_eq!(render.snapshot().0, 2);
        cancel.shutdown();
        assert!(matches!(
            paced.play(audio(3, cancel.generation(), 10, true), &cancel),
            Err(Error::Cancelled)
        ));
    }
}
