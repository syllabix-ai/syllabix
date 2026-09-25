//! Opt-in per-turn diagnostics configured under `diagnostics` in `syllabix.yaml`:
//! monotonic turn-timeline sidecars, plus the turn WAVs when `audio: true`.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::audio::{write_wav, PcmFormat, WavPcm};
use crate::error::{Error, Result};
use crate::speech_text::speak_text_for_tts;
use crate::types::{
    LlmDebugMeta, ToolTurnEvent, TurnId, TurnTimings, Utterance, DEFAULT_SAMPLE_RATE_HZ,
};

/// Default directory when `diagnostics:` is enabled without an explicit one.
pub const DEFAULT_TURN_DEBUG_DIR: &str = "target/turn-debug";

/// One monotonic stage boundary on a turn's timeline.
///
/// Anchors are captured once per transition with [`std::time::Instant`]
/// (monotonic) and rendered relative to `speech_start` in the sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TimelineAnchor {
    /// First Silero-positive frame opened the turn (timeline epoch).
    SpeechStart,
    /// Hangover satisfied; the utterance left the VAD.
    SpeechEnd,
    /// The STT worker dequeued the utterance (queue wait separator).
    SttQueued,
    /// First visible provisional STT text while VAD still owns the turn.
    SttPartial,
    /// Transcript text and language are ready.
    SttDone,
    /// The LLM worker started generating.
    LlmStart,
    /// First token streamed.
    LlmFirstToken,
    /// Last token of the generation.
    LlmLastToken,
    /// First PCM the TTS worker synthesized for this turn.
    TtsFirstPcm,
    /// Final PCM chunk of the synthesis.
    TtsLastPcm,
    /// Device callback first consumed samples of this turn.
    PlaybackFirst,
    /// Device drained the ring after the final chunk.
    PlaybackDone,
}

impl TimelineAnchor {
    fn as_str(self) -> &'static str {
        match self {
            Self::SpeechStart => "speech_start",
            Self::SpeechEnd => "speech_end",
            Self::SttQueued => "stt_queued",
            Self::SttPartial => "stt_partial",
            Self::SttDone => "stt_done",
            Self::LlmStart => "llm_start",
            Self::LlmFirstToken => "llm_first_token",
            Self::LlmLastToken => "llm_last_token",
            Self::TtsFirstPcm => "tts_first_pcm",
            Self::TtsLastPcm => "tts_last_pcm",
            Self::PlaybackFirst => "playback_first",
            Self::PlaybackDone => "playback_done",
        }
    }

    /// Sidecar render order: pipeline order, not enum-discriminant order.
    pub const ALL: [TimelineAnchor; 12] = [
        Self::SpeechStart,
        Self::SpeechEnd,
        Self::SttQueued,
        Self::SttPartial,
        Self::SttDone,
        Self::LlmStart,
        Self::LlmFirstToken,
        Self::LlmLastToken,
        Self::TtsFirstPcm,
        Self::TtsLastPcm,
        Self::PlaybackFirst,
        Self::PlaybackDone,
    ];
}

/// How far a dumped turn got through the loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnOutcome {
    /// Last TTS chunk reached the sink.
    Completed,
    /// Recoverable provider error skipped the turn.
    Skipped,
    /// Shutdown or generation cancel interrupted the turn.
    Cancelled,
}

impl TurnOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Skipped => "skipped",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Shared recorder. Cheap to clone; workers write under one mutex.
#[derive(Debug, Clone)]
pub struct TurnDebug {
    root: PathBuf,
    /// When false (yaml `audio: false`), PCM is never buffered — only the
    /// monotonic timeline and sidecar text fields are kept.
    collect_audio: bool,
    inner: Arc<Mutex<Inner>>,
}

#[derive(Debug)]
struct Inner {
    turns: BTreeMap<u64, TurnDump>,
}

#[derive(Debug, Default)]
struct TurnDump {
    capture: Vec<i16>,
    clean: Vec<i16>,
    utterance: Vec<i16>,
    utterance_frames: usize,
    tts: Vec<i16>,
    tts_chunks: usize,
    capture_frames: usize,
    stt_text: Option<String>,
    stt_language: Option<String>,
    llm_text: String,
    llm_meta: Option<LlmDebugMeta>,
    tool_events: Vec<ToolTurnEvent>,
    tts_provider: Option<String>,
    tts_model: Option<String>,
    tts_backend: Option<String>,
    timings: Option<TurnTimings>,
    timeline: BTreeMap<TimelineAnchor, Instant>,
    written: bool,
    /// Outcome of the first dump. Kept so late LLM meta / tool events can
    /// rewrite the sidecar after a fast TTS `complete()` raced ahead of the
    /// LLM worker's post-generate bookkeeping.
    outcome: Option<TurnOutcome>,
}

impl TurnDebug {
    /// Create `dir` (and parents) and confirm it is writable before the loop starts.
    ///
    /// Audio capture (turn WAVs) is on; use [`TurnDebug::open_with_audio`] or
    /// [`TurnDebug::from_config`] for the yaml-driven forms.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_audio(dir, true)
    }

    /// Create the recorder with explicit WAV capture control (`diagnostics.audio`).
    pub fn open_with_audio(dir: impl Into<PathBuf>, collect_audio: bool) -> Result<Self> {
        let root = dir.into();
        prepare_turn_debug_dir(&root)?;
        Ok(Self {
            root,
            collect_audio,
            inner: Arc::new(Mutex::new(Inner {
                turns: BTreeMap::new(),
            })),
        })
    }

    /// Build the recorder from yaml `diagnostics:` settings; `None` when disabled.
    pub fn from_config(config: &crate::config::AgentConfig) -> Result<Option<Self>> {
        if !config.diagnostics_enabled() {
            return Ok(None);
        }
        Self::open_with_audio(
            config.diagnostics_directory.clone(),
            config.diagnostics_audio,
        )
        .map(Some)
    }

    /// Whether turn WAVs are being captured (`diagnostics.audio`).
    pub fn collects_audio(&self) -> bool {
        self.collect_audio
    }

    /// Directory files are written into.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Reserve a turn started by VAD speech-onset.
    pub fn start_turn(&self, turn: TurnId) {
        let mut inner = self.lock();
        inner.turns.entry(turn.0).or_default();
    }

    /// Record the first instant of a timeline anchor (first write wins).
    pub fn note_anchor(&self, turn: TurnId, anchor: TimelineAnchor, at: Instant) {
        let mut inner = self.lock();
        inner
            .turns
            .entry(turn.0)
            .or_default()
            .timeline
            .entry(anchor)
            .or_insert(at);
    }

    /// Record an end-of-stage anchor whose latest instant wins — playback
    /// drains can fire more than once when a long synthesis underruns.
    pub fn note_last_anchor(&self, turn: TurnId, anchor: TimelineAnchor, at: Instant) {
        let mut inner = self.lock();
        inner
            .turns
            .entry(turn.0)
            .or_default()
            .timeline
            .insert(anchor, at);
    }

    /// Timeline anchors recorded for `turn`, as pipeline-order offsets from
    /// `speech_start`. Missing anchors are absent from the vector.
    pub fn anchored(&self, turn: TurnId) -> Vec<(TimelineAnchor, Duration)> {
        let inner = self.lock();
        let Some(dump) = inner.turns.get(&turn.0) else {
            return Vec::new();
        };
        let epoch = match dump.timeline.get(&TimelineAnchor::SpeechStart) {
            Some(at) => *at,
            None => return Vec::new(),
        };
        TimelineAnchor::ALL
            .iter()
            .filter_map(|anchor| {
                dump.timeline
                    .get(anchor)
                    .map(|at| (*anchor, at.saturating_duration_since(epoch)))
            })
            .collect()
    }

    /// Append one Silero-input frame (clean) and its pre-AEC twin (capture).
    pub fn note_frame(&self, turn: TurnId, clean: &[i16], capture: Option<&[i16]>) {
        if !self.collect_audio {
            return;
        }
        let mut inner = self.lock();
        let dump = inner.turns.entry(turn.0).or_default();
        dump.clean.extend_from_slice(clean);
        dump.capture.extend_from_slice(capture.unwrap_or(clean));
        dump.capture_frames += 1;
    }

    /// PCM Whisper will transcribe.
    pub fn note_utterance(&self, utterance: &Utterance) {
        if !self.collect_audio {
            return;
        }
        let mut inner = self.lock();
        let dump = inner.turns.entry(utterance.turn.0).or_default();
        dump.utterance = utterance.pcm();
        dump.utterance_frames = utterance.frames.len();
    }

    /// STT hypothesis and its effective language code.
    pub fn note_stt(&self, turn: TurnId, text: &str, language: &str) {
        let mut inner = self.lock();
        let dump = inner.turns.entry(turn.0).or_default();
        dump.stt_text = Some(text.to_string());
        dump.stt_language = Some(language.to_string());
    }

    /// Full LLM reply (concatenated tokens).
    pub fn note_llm(&self, turn: TurnId, text: String) {
        let mut inner = self.lock();
        inner.turns.entry(turn.0).or_default().llm_text = text;
    }

    /// Provider facts for the sidecar (provider, model, endpoint, request id).
    /// Best-effort: a barge-in dump that already wrote keeps its outcome, but
    /// the JSON is rewritten so late meta is not lost when TTS completes
    /// during `llm.generate` (before the LLM worker notes facts).
    pub fn note_llm_meta(&self, turn: TurnId, meta: Option<LlmDebugMeta>) {
        let Some(meta) = meta else { return };
        let mut inner = self.lock();
        let dump = inner.turns.entry(turn.0).or_default();
        dump.llm_meta = Some(meta);
        let _ = rewrite_sidecar_if_written(&self.root, turn.0, dump);
    }

    /// Ordered API-tool evidence. These events exclude host-only executor
    /// policy and ambient secrets by construction. Rewrites an already-written
    /// sidecar the same way [`Self::note_llm_meta`] does.
    pub fn note_tool_events(&self, turn: TurnId, events: Vec<ToolTurnEvent>) {
        if events.is_empty() {
            return;
        }
        let mut inner = self.lock();
        let dump = inner.turns.entry(turn.0).or_default();
        dump.tool_events.extend(events);
        let _ = rewrite_sidecar_if_written(&self.root, turn.0, dump);
    }

    /// PCM actually handed to the sink, plus the TTS provider facts for the
    /// sidecar (provider and loaded weight id). Provider facts are recorded
    /// even when WAV capture is off.
    pub fn note_tts(
        &self,
        turn: TurnId,
        samples: &[i16],
        provider: &str,
        model: Option<&str>,
        backend: Option<&str>,
    ) {
        let mut inner = self.lock();
        let dump = inner.turns.entry(turn.0).or_default();
        if self.collect_audio {
            dump.tts.extend_from_slice(samples);
            dump.tts_chunks += 1;
        }
        dump.tts_provider = Some(provider.to_string());
        dump.tts_model = model.map(str::to_string);
        dump.tts_backend = backend.map(str::to_string);
    }

    /// Last audio chunk played; write the dump as completed.
    pub fn complete(&self, turn: TurnId, timings: TurnTimings) -> Result<()> {
        let mut inner = self.lock();
        if let Some(dump) = inner.turns.get_mut(&turn.0) {
            dump.timings = Some(timings);
        }
        write_turn(
            &self.root,
            turn.0,
            TurnOutcome::Completed,
            self.collect_audio,
            &mut inner,
        )
    }

    /// Recoverable provider failure; keep whatever audio/text existed.
    pub fn skip(&self, turn: TurnId) -> Result<()> {
        let mut inner = self.lock();
        write_turn(
            &self.root,
            turn.0,
            TurnOutcome::Skipped,
            self.collect_audio,
            &mut inner,
        )
    }

    /// Barge-in or generation cancel; dump interrupted TTS plus what exists so far.
    pub fn interrupt(&self, turn: TurnId) -> Result<()> {
        let mut inner = self.lock();
        write_turn(
            &self.root,
            turn.0,
            TurnOutcome::Cancelled,
            self.collect_audio,
            &mut inner,
        )
    }

    /// Write every turn that never completed or skipped (shutdown / cancel).
    pub fn finish_open(&self) -> Result<()> {
        let mut inner = self.lock();
        let ids: Vec<u64> = inner.turns.keys().copied().collect();
        for id in ids {
            write_turn(
                &self.root,
                id,
                TurnOutcome::Cancelled,
                self.collect_audio,
                &mut inner,
            )?;
        }
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("turn debug mutex")
    }
}

/// Device-side playback anchors: the first callback that consumes a turn's
/// samples and the drain after its final chunk.
///
/// The cpal output closure owns a clone of the watch; the native playback
/// sink (or a collecting sink) calls [`PlaybackWatch::begin_turn`] per chunk
/// and the callback reports how many samples it actually rendered. Edges are
/// tracked with one atomic so underruns mid-turn keep the first
/// `playback_first` (first write wins) while `playback_done` always moves to
/// the latest drain.
#[derive(Debug, Clone)]
pub struct PlaybackWatch {
    inner: Arc<PlaybackWatchInner>,
}

#[derive(Debug)]
struct PlaybackWatchInner {
    recorder: TurnDebug,
    active: Mutex<Option<TurnId>>,
    consuming: AtomicBool,
}

impl PlaybackWatch {
    /// Watch playback on behalf of `recorder`.
    pub fn new(recorder: TurnDebug) -> Self {
        Self {
            inner: Arc::new(PlaybackWatchInner {
                recorder,
                active: Mutex::new(None),
                consuming: AtomicBool::new(false),
            }),
        }
    }

    /// Attribute subsequent callback consumption to `turn`.
    pub fn begin_turn(&self, turn: TurnId) {
        *self
            .inner
            .active
            .lock()
            .expect("playback watch active turn") = Some(turn);
    }

    /// Report one device callback: samples actually rendered this tick.
    pub fn on_callback(&self, rendered_samples: usize) {
        let now = Instant::now();
        let active = self
            .inner
            .active
            .lock()
            .expect("playback watch active turn")
            .as_ref()
            .copied();
        if rendered_samples > 0 {
            if !self.inner.consuming.swap(true, Ordering::SeqCst) {
                if let Some(turn) = active {
                    self.inner
                        .recorder
                        .note_anchor(turn, TimelineAnchor::PlaybackFirst, now);
                }
            }
        } else if self.inner.consuming.swap(false, Ordering::SeqCst) {
            if let Some(turn) = active {
                self.inner
                    .recorder
                    .note_last_anchor(turn, TimelineAnchor::PlaybackDone, now);
            }
        }
    }
}

/// Create `dir` and fail with an actionable error if it cannot be written.
pub fn prepare_turn_debug_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).map_err(|err| turn_debug_io(dir, err))?;
    let meta = fs::metadata(dir).map_err(|err| turn_debug_io(dir, err))?;
    if !meta.is_dir() {
        return Err(Error::Config {
            field: "diagnostics.directory".into(),
            message: format!(
                "{} exists and is not a directory. Point diagnostics.directory at a writable directory.",
                dir.display()
            ),
        });
    }
    let probe = dir.join(".syllabix-turn-debug-write");
    fs::write(&probe, b"ok").map_err(|err| turn_debug_io(dir, err))?;
    fs::remove_file(&probe).map_err(|err| turn_debug_io(dir, err))?;
    Ok(())
}

fn turn_debug_io(dir: &Path, err: std::io::Error) -> Error {
    Error::Config {
        field: "diagnostics.directory".into(),
        message: format!(
            "cannot write diagnostics files under {}: {err}. Point diagnostics.directory at a writable directory.",
            dir.display()
        ),
    }
}

fn write_turn(
    root: &Path,
    id: u64,
    outcome: TurnOutcome,
    collect_audio: bool,
    inner: &mut Inner,
) -> Result<()> {
    let Some(dump) = inner.turns.get_mut(&id) else {
        return Ok(());
    };
    if dump.written {
        return Ok(());
    }
    dump.written = true;
    dump.outcome = Some(outcome);
    let dir = root.join(format!("turn-{id:03}"));
    fs::create_dir_all(&dir).map_err(|err| turn_debug_io(&dir, err))?;
    if collect_audio {
        write_pcm(&dir.join("capture.wav"), &dump.capture)?;
        write_pcm(&dir.join("clean.wav"), &dump.clean)?;
        write_pcm(&dir.join("utterance.wav"), &dump.utterance)?;
        write_pcm(&dir.join("tts.wav"), &dump.tts)?;
    }
    write_sidecar_json(root, id, outcome, dump)
}

/// Refresh `turn.json` after a dump that already landed (late LLM meta / tools).
fn rewrite_sidecar_if_written(root: &Path, id: u64, dump: &TurnDump) -> Result<()> {
    let Some(outcome) = dump.outcome else {
        return Ok(());
    };
    if !dump.written {
        return Ok(());
    }
    write_sidecar_json(root, id, outcome, dump)
}

fn write_sidecar_json(root: &Path, id: u64, outcome: TurnOutcome, dump: &TurnDump) -> Result<()> {
    let dir = root.join(format!("turn-{id:03}"));
    let sidecar = render_sidecar(id, outcome, dump).replacen(
        "\n  \"tts_provider\":",
        &format!(
            "\n  \"tool_events\": {},\n  \"tts_provider\":",
            render_tool_events(&dump.tool_events)
        ),
        1,
    );
    fs::write(dir.join("turn.json"), sidecar).map_err(|err| turn_debug_io(&dir, err))?;
    Ok(())
}

fn render_tool_events(events: &[ToolTurnEvent]) -> String {
    let entries: Vec<_> = events
        .iter()
        .map(|event| {
            format!(
                "{{\"kind\":{},\"name\":{},\"call_id\":{},\"arguments\":{},\"content\":{}}}",
                json_string(&event.kind),
                json_string(&event.name),
                json_string(&event.call_id),
                json_string(&event.arguments),
                json_string(&event.content),
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
}

fn write_pcm(path: &Path, samples: &[i16]) -> Result<()> {
    let file = File::create(path).map_err(|err| turn_debug_io(path, err))?;
    write_wav(
        file,
        &WavPcm {
            format: PcmFormat::v0(),
            samples: samples.to_vec(),
        },
    )
}

fn render_sidecar(id: u64, outcome: TurnOutcome, dump: &TurnDump) -> String {
    let stt = dump.stt_text.as_deref().unwrap_or("");
    let stt_language = dump.stt_language.as_deref().unwrap_or("");
    let speak = speak_text_for_tts(&dump.llm_text);
    let timings = dump.timings.unwrap_or_default();
    let default_meta = LlmDebugMeta::default();
    let meta = dump.llm_meta.as_ref().unwrap_or(&default_meta);
    format!(
        "{{\n  \"turn\": {id},\n  \"outcome\": {},\n  \"stt_text\": {},\n  \"stt_language\": {},\n  \"llm_text\": {},\n  \"llm_provider\": {},\n  \"llm_model\": {},\n  \"llm_endpoint\": {},\n  \"llm_request_id\": {},\n  \"tts_provider\": {},\n  \"tts_model\": {},\n  \"tts_backend\": {},\n  \"tts_speak_text\": {},\n  \"timeline\": {},\n  \"timings\": {{\n    \"stt_ms\": {},\n    \"ttft_ms\": {},\n    \"ttfb_ms\": {},\n    \"total_ms\": {}\n  }},\n  \"capture_samples\": {},\n  \"capture_frames\": {},\n  \"capture_duration_ms\": {},\n  \"clean_samples\": {},\n  \"clean_frames\": {},\n  \"clean_duration_ms\": {},\n  \"utterance_samples\": {},\n  \"utterance_frames\": {},\n  \"utterance_duration_ms\": {},\n  \"tts_samples\": {},\n  \"tts_chunks\": {},\n  \"tts_duration_ms\": {}\n}}\n",
        json_string(outcome.as_str()),
        json_string(stt),
        json_string(stt_language),
        json_string(&dump.llm_text),
        json_string(&meta.provider),
        json_string(&meta.model),
        json_string(&meta.endpoint),
        json_string(&meta.request_id),
        json_string(dump.tts_provider.as_deref().unwrap_or("")),
        json_string(dump.tts_model.as_deref().unwrap_or("")),
        json_string(dump.tts_backend.as_deref().unwrap_or("")),
        json_string(&speak),
        render_timeline(dump),
        duration_ms(timings.stt),
        duration_ms(timings.ttft),
        duration_ms(timings.ttfb),
        duration_ms(timings.total),
        dump.capture.len(),
        dump.capture_frames,
        pcm_duration_ms(dump.capture.len()),
        dump.clean.len(),
        dump.capture_frames,
        pcm_duration_ms(dump.clean.len()),
        dump.utterance.len(),
        dump.utterance_frames,
        pcm_duration_ms(dump.utterance.len()),
        dump.tts.len(),
        dump.tts_chunks,
        pcm_duration_ms(dump.tts.len()),
    )
}

/// Monotonic timeline in pipeline order, milliseconds from `speech_start`
/// (which renders as 0). Anchors the turn never reached render as `null`.
fn render_timeline(dump: &TurnDump) -> String {
    let epoch = match dump.timeline.get(&TimelineAnchor::SpeechStart) {
        Some(at) => *at,
        // Without the epoch the offsets are meaningless; keep schema shape.
        None => return "{}".to_string(),
    };
    let fields: Vec<String> = TimelineAnchor::ALL
        .iter()
        .map(|anchor| match dump.timeline.get(anchor) {
            Some(at) => format!(
                "\"{}_ms\": {}",
                anchor.as_str(),
                at.saturating_duration_since(epoch).as_millis()
            ),
            None => format!("\"{}_ms\": null", anchor.as_str()),
        })
        .collect();
    format!("{{\n    {}\n  }}", fields.join(",\n    "))
}

fn duration_ms(duration: Duration) -> u128 {
    duration.as_millis()
}

fn pcm_duration_ms(samples: usize) -> u64 {
    (samples as u64).saturating_mul(1000) / u64::from(DEFAULT_SAMPLE_RATE_HZ)
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::read_wav;
    use crate::types::{AudioFrame, DEFAULT_CHANNELS, FRAME_SAMPLES};
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "syllabix-turn-debug-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn prepare_rejects_a_file_path() {
        let dir = unique_dir();
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("not-a-dir");
        fs::write(&file, b"x").unwrap();
        let err = prepare_turn_debug_dir(&file).unwrap_err();
        assert!(matches!(err, Error::Config { .. }));
        let message = err.to_string();
        assert!(
            message.contains("not a directory") || message.contains("cannot write"),
            "{message}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn prepare_rejects_when_parent_is_a_file() {
        let dir = unique_dir();
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("blocker");
        fs::write(&file, b"x").unwrap();
        let err = prepare_turn_debug_dir(&file.join("nested")).unwrap_err();
        assert!(matches!(err, Error::Config { .. }));
        assert!(err.to_string().contains("cannot write"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn dumps_four_wavs_and_sidecar() {
        let dir = unique_dir();
        let debug = TurnDebug::open(&dir).unwrap();
        let turn = TurnId(0);
        debug.start_turn(turn);
        let frame = AudioFrame::new(
            0,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CHANNELS,
            vec![3; FRAME_SAMPLES],
        )
        .unwrap();
        debug.note_frame(turn, &frame.samples, None);
        let utterance = Utterance {
            turn,
            frames: vec![frame],
        };
        debug.note_utterance(&utterance);
        debug.note_stt(turn, "hello", "en");
        debug.note_llm(turn, "<think>plan</think> hi **there**".into());
        debug.note_tts(turn, &[9, 8, 7], "kokoro", Some("kokoro"), Some("onnx"));
        debug
            .complete(
                turn,
                TurnTimings {
                    stt: Duration::from_millis(11),
                    ttft: Duration::from_millis(22),
                    ttfb: Duration::from_millis(33),
                    total: Duration::from_millis(44),
                },
            )
            .unwrap();

        let turn_dir = dir.join("turn-000");
        for name in ["capture.wav", "clean.wav", "utterance.wav", "tts.wav"] {
            assert!(turn_dir.join(name).is_file(), "{name}");
        }
        let json = fs::read_to_string(turn_dir.join("turn.json")).unwrap();
        assert!(json.contains("\"outcome\": \"completed\""));
        assert!(json.contains("\"stt_text\": \"hello\""));
        assert!(json.contains("\"stt_language\": \"en\""));
        assert!(json.contains("\"llm_text\": \"<think>plan</think> hi **there**\""));
        assert!(json.contains("\"tts_speak_text\": \"hi there\""));
        assert!(json.contains("\"stt_ms\": 11"));
        let wav = read_wav(Cursor::new(fs::read(turn_dir.join("tts.wav")).unwrap())).unwrap();
        assert_eq!(wav.format, PcmFormat::v0());
        assert_eq!(wav.samples, vec![9, 8, 7]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn skip_and_cancel_still_write() {
        let dir = unique_dir();
        let debug = TurnDebug::open(&dir).unwrap();
        debug.note_stt(TurnId(1), "partial", "fr");
        debug.skip(TurnId(1)).unwrap();
        debug.start_turn(TurnId(2));
        debug.finish_open().unwrap();
        let skipped = fs::read_to_string(dir.join("turn-001").join("turn.json")).unwrap();
        assert!(skipped.contains("skipped"));
        assert!(skipped.contains("\"stt_language\": \"fr\""));
        let cancelled = fs::read_to_string(dir.join("turn-002").join("turn.json")).unwrap();
        assert!(cancelled.contains("cancelled"));
        debug.finish_open().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn interrupt_writes_cancelled_turn_immediately() {
        let dir = unique_dir();
        let debug = TurnDebug::open(&dir).unwrap();
        debug.start_turn(TurnId(0));
        debug.note_tts(
            TurnId(0),
            &[1, 2, 3],
            "local",
            Some("qwen3-tts-1.7b-base"),
            Some("metal"),
        );
        debug.interrupt(TurnId(0)).unwrap();
        let json = fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
        assert!(json.contains("cancelled"));
        // TTS and LLM provider metadata share one sidecar. The provider field
        // describes whether inference ran locally or online.
        assert!(json.contains("\"tts_provider\": \"local\""), "{json}");
        assert!(json.contains("\"tts_backend\": \"metal\""), "{json}");
        assert!(
            json.contains("\"tts_model\": \"qwen3-tts-1.7b-base\""),
            "{json}"
        );
        let wav = read_wav(Cursor::new(
            fs::read(dir.join("turn-000").join("tts.wav")).unwrap(),
        ))
        .unwrap();
        assert_eq!(wav.samples, vec![1, 2, 3]);
        debug.finish_open().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn llm_meta_lands_in_the_sidecar_even_for_skipped_turns() {
        let dir = unique_dir();
        let debug = TurnDebug::open(&dir).unwrap();
        let turn = TurnId(3);
        debug.start_turn(turn);
        debug.note_stt(turn, "hello", "en");
        debug.note_llm_meta(
            turn,
            Some(LlmDebugMeta {
                provider: "online".into(),
                model: "gpt-test".into(),
                endpoint: "https://mock.example/v1/chat/completions".into(),
                request_id: "req-7".into(),
            }),
        );
        // Skipped turns write immediately; the meta must already be attached.
        debug.skip(turn).unwrap();
        let json = fs::read_to_string(dir.join("turn-003").join("turn.json")).unwrap();
        assert!(json.contains("\"llm_provider\": \"online\""), "{json}");
        assert!(json.contains("\"llm_model\": \"gpt-test\""), "{json}");
        assert!(
            json.contains("\"llm_endpoint\": \"https://mock.example/v1/chat/completions\""),
            "{json}"
        );
        assert!(json.contains("\"llm_request_id\": \"req-7\""), "{json}");
        // Local turns keep empty endpoint/request-id fields.
        let local = TurnId(4);
        debug.start_turn(local);
        debug.note_stt(local, "hi", "en");
        debug.note_llm_meta(
            local,
            Some(crate::types::LlmDebugMeta {
                provider: "local".into(),
                model: "lfm2.5-2.6b".into(),
                ..Default::default()
            }),
        );
        debug.skip(local).unwrap();
        let json = fs::read_to_string(dir.join("turn-004").join("turn.json")).unwrap();
        assert!(json.contains("\"llm_provider\": \"local\""), "{json}");
        assert!(json.contains("\"llm_endpoint\": \"\""), "{json}");
        assert!(json.contains("\"llm_request_id\": \"\""), "{json}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn late_llm_meta_rewrites_a_sidecar_already_completed_by_tts() {
        // Mirrors the weekly Linux flake: FakeTts emits is_last audio during
        // llm.generate, sink calls complete() before the LLM worker notes meta.
        let dir = unique_dir();
        let debug = TurnDebug::open(&dir).unwrap();
        let turn = TurnId(0);
        debug.start_turn(turn);
        debug.note_stt(turn, "turn-000", "en");
        debug.note_llm(turn, "echo:turn-000".into());
        debug.note_tts(turn, &[1, 2, 3], "local", None, None);
        debug
            .complete(turn, TurnTimings::default())
            .expect("complete");
        let early = fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
        assert!(
            early.contains("\"llm_provider\": \"\""),
            "precondition: empty meta before rewrite {early}"
        );
        debug.note_llm_meta(
            turn,
            Some(LlmDebugMeta {
                provider: "online".into(),
                model: "gpt-test".into(),
                endpoint: "https://mock.example/v1/chat/completions".into(),
                request_id: "req-42".into(),
            }),
        );
        debug.note_tool_events(
            turn,
            vec![crate::types::ToolTurnEvent {
                kind: "result".into(),
                name: "web_fetch".into(),
                call_id: "call-42".into(),
                arguments: r#"{"url":"https://example.test"}"#.into(),
                content: "fixture result".into(),
            }],
        );
        let json = fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
        assert!(json.contains("\"outcome\": \"completed\""), "{json}");
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
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn json_string_escapes_quotes() {
        assert_eq!(json_string("a\"b\\c"), "\"a\\\"b\\\\c\"");
    }

    #[test]
    fn from_config_disabled_returns_none_without_touching_the_filesystem() {
        let mut config = crate::config::AgentConfig::v0();
        assert!(TurnDebug::from_config(&config).unwrap().is_none());
        config.diagnostics_timestamps = true;
        let debug = TurnDebug::from_config(&config).unwrap().expect("enabled");
        assert!(
            !debug.collects_audio(),
            "timestamps-only records sidecars, not WAVs"
        );
    }

    #[test]
    fn from_config_audio_only_enables_the_recorder() {
        let mut config = crate::config::AgentConfig::v0();
        // Parsed yaml would normalize audio⇒timestamps; honor the raw flags too.
        config.diagnostics_audio = true;
        let dir = unique_dir();
        config.diagnostics_directory = dir.clone();
        let debug = TurnDebug::from_config(&config).unwrap().expect("enabled");
        assert!(debug.collects_audio());
        assert_eq!(debug.root(), dir.as_path());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn from_config_unwritable_directory_fails_before_the_loop() {
        let mut config = crate::config::AgentConfig::v0();
        let dir = unique_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let blocker = dir.join("file");
        std::fs::write(&blocker, b"x").unwrap();
        config.diagnostics_timestamps = true;
        config.diagnostics_directory = blocker.join("nested");
        let err = TurnDebug::from_config(&config).unwrap_err();
        assert!(matches!(err, Error::Config { .. }));
        assert!(err.to_string().contains("diagnostics.directory"), "{err}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn timestamps_only_sidecar_has_no_wavs_and_buffers_no_pcm() {
        let dir = unique_dir();
        let debug = TurnDebug::open_with_audio(&dir, false).unwrap();
        let turn = TurnId(0);
        debug.start_turn(turn);
        debug.note_frame(turn, &[1, 2, 3], Some(&[4, 5, 6]));
        let frame = crate::types::AudioFrame::new(
            0,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CHANNELS,
            vec![7; FRAME_SAMPLES],
        )
        .unwrap();
        debug.note_utterance(&Utterance {
            turn,
            frames: vec![frame],
        });
        debug.note_tts(turn, &[9, 8, 7], "kokoro", Some("kokoro"), Some("onnx"));
        debug.note_anchor(turn, TimelineAnchor::SpeechStart, Instant::now());
        debug.note_last_anchor(turn, TimelineAnchor::PlaybackDone, Instant::now());
        debug.finish_open().unwrap();

        let turn_dir = dir.join("turn-000");
        assert!(turn_dir.join("turn.json").is_file());
        for name in ["capture.wav", "clean.wav", "utterance.wav", "tts.wav"] {
            assert!(!turn_dir.join(name).exists(), "{name} must not exist");
        }
        let json = fs::read_to_string(turn_dir.join("turn.json")).unwrap();
        // Provider facts survive without WAV capture.
        assert!(json.contains("\"tts_provider\": \"kokoro\""), "{json}");
        // Timeline renders with the epoch zeroed and everything else present.
        assert!(json.contains("\"speech_start_ms\": 0"), "{json}");
        assert!(json.contains("\"playback_done_ms\":"), "{json}");
        assert!(!json.contains("\"capture_samples\": 3"), "{json}");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn timeline_renders_pipeline_order_with_nulls_for_missing_anchors() {
        let dir = unique_dir();
        let debug = TurnDebug::open(&dir).unwrap();
        let turn = TurnId(0);
        debug.start_turn(turn);
        let t0 = Instant::now();
        debug.note_anchor(turn, TimelineAnchor::SpeechStart, t0);
        debug.note_anchor(
            turn,
            TimelineAnchor::SpeechEnd,
            t0 + Duration::from_millis(100),
        );
        debug.note_anchor(
            turn,
            TimelineAnchor::LlmFirstToken,
            t0 + Duration::from_millis(250),
        );
        debug
            .complete(
                turn,
                TurnTimings {
                    stt: Duration::from_millis(10),
                    ttft: Duration::from_millis(20),
                    ttfb: Duration::from_millis(30),
                    total: Duration::from_millis(40),
                },
            )
            .unwrap();
        let json = fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
        // Pipeline order in the rendered object:
        let speech_start = json.find("\"speech_start_ms\": 0").expect("epoch zero");
        let speech_end = json.find("\"speech_end_ms\": 100").expect("speech_end");
        let stt_queued = json
            .find("\"stt_queued_ms\": null")
            .expect("stt_queued null");
        let first_token = json
            .find("\"llm_first_token_ms\": 250")
            .expect("first token");
        let playback_first = json.find("\"playback_first_ms\": null").expect("null");
        let playback_done = json.find("\"playback_done_ms\": null").expect("null");
        assert!(speech_start < speech_end);
        assert!(speech_end < stt_queued);
        assert!(stt_queued < first_token);
        assert!(first_token < playback_first);
        assert!(playback_first < playback_done);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn anchored_reports_offsets_in_pipeline_order() {
        let dir = unique_dir();
        let debug = TurnDebug::open(&dir).unwrap();
        let turn = TurnId(2);
        let t0 = Instant::now();
        debug.start_turn(turn);
        debug.note_anchor(turn, TimelineAnchor::SpeechStart, t0);
        debug.note_anchor(
            turn,
            TimelineAnchor::SpeechEnd,
            t0 + Duration::from_millis(5),
        );
        debug.note_last_anchor(
            turn,
            TimelineAnchor::PlaybackDone,
            t0 + Duration::from_millis(9),
        );
        let anchored = debug.anchored(turn);
        assert_eq!(
            anchored.iter().map(|(a, _)| *a).collect::<Vec<_>>(),
            vec![
                TimelineAnchor::SpeechStart,
                TimelineAnchor::SpeechEnd,
                TimelineAnchor::PlaybackDone,
            ]
        );
        assert_eq!(anchored[1].1, Duration::from_millis(5));
        assert_eq!(anchored[2].1, Duration::from_millis(9));
        // No epoch recorded → nothing to report.
        assert!(debug.anchored(TurnId(3)).is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn playback_watch_keeps_first_start_and_latest_drain() {
        let dir = unique_dir();
        let recorder = TurnDebug::open(&dir).unwrap();
        let watch = PlaybackWatch::new(recorder.clone());
        let turn = TurnId(0);
        // Every real turn carries a VAD epoch; seed it like the pipeline does.
        recorder.note_anchor(turn, TimelineAnchor::SpeechStart, Instant::now());
        // Callbacks before any turn is active are ignored.
        watch.on_callback(128);
        watch.on_callback(0);
        assert!(
            recorder
                .anchored(turn)
                .iter()
                .all(|(a, _)| *a == TimelineAnchor::SpeechStart),
            "no playback anchors without an active turn"
        );

        watch.begin_turn(turn);
        watch.on_callback(0); // idle tick before the first chunk lands
        watch.on_callback(512); // rising edge → playback_first
        watch.on_callback(0); // underrun → premature drain
        watch.on_callback(256); // next chunk; first stays pinned
        watch.on_callback(0); // final drain overwrites playback_done

        let anchored = recorder.anchored(turn);
        let names: Vec<_> = anchored.iter().map(|(a, _)| *a).collect();
        assert_eq!(
            names,
            vec![
                TimelineAnchor::SpeechStart,
                TimelineAnchor::PlaybackFirst,
                TimelineAnchor::PlaybackDone,
            ]
        );
        // Drain always moves forward relative to the first consumption.
        let first_at = &anchored[1].1;
        let done_at = &anchored[2].1;
        assert!(first_at <= done_at);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn playback_watch_retargets_to_the_next_turn() {
        let dir = unique_dir();
        let recorder = TurnDebug::open(&dir).unwrap();
        let watch = PlaybackWatch::new(recorder.clone());
        let first = TurnId(0);
        let second = TurnId(1);
        recorder.note_anchor(first, TimelineAnchor::SpeechStart, Instant::now());
        recorder.note_anchor(second, TimelineAnchor::SpeechStart, Instant::now());
        // Turn one plays, then a barge-in flush drains it (the cleared ring
        // makes the next callback pop zero samples).
        watch.begin_turn(first);
        watch.on_callback(64);
        watch.on_callback(0);
        // The next turn's audio arrives afterwards and must retarget.
        watch.begin_turn(second);
        watch.on_callback(64);
        watch.on_callback(0);
        let first_anchors = recorder.anchored(first);
        let second_anchors = recorder.anchored(second);
        assert_eq!(
            first_anchors.iter().map(|(a, _)| *a).collect::<Vec<_>>(),
            vec![
                TimelineAnchor::SpeechStart,
                TimelineAnchor::PlaybackFirst,
                TimelineAnchor::PlaybackDone,
            ],
            "the flushed turn keeps its own first + drain instants"
        );
        assert_eq!(
            second_anchors.iter().map(|(a, _)| *a).collect::<Vec<_>>(),
            vec![
                TimelineAnchor::SpeechStart,
                TimelineAnchor::PlaybackFirst,
                TimelineAnchor::PlaybackDone
            ]
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
