//! Opt-in per-turn WAV + sidecar dumps for `syllabix run --turn-debug`.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audio::{write_wav, PcmFormat, WavPcm};
use crate::error::{Error, Result};
use crate::speech_text::strip_markdown_for_speech;
use crate::types::{TurnId, TurnTimings, Utterance, DEFAULT_SAMPLE_RATE_HZ};

/// Environment override for the default dump directory.
pub const TURN_DEBUG_DIR_ENV: &str = "SYLLABIX_TURN_DEBUG_DIR";

/// Default directory when `--turn-debug` is passed without a path or env override.
pub const DEFAULT_TURN_DEBUG_DIR: &str = "target/turn-debug";

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
    llm_text: String,
    timings: Option<TurnTimings>,
    written: bool,
}

impl TurnDebug {
    /// Create `dir` (and parents) and confirm it is writable before the loop starts.
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let root = dir.into();
        prepare_turn_debug_dir(&root)?;
        Ok(Self {
            root,
            inner: Arc::new(Mutex::new(Inner {
                turns: BTreeMap::new(),
            })),
        })
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

    /// Append one Silero-input frame (clean) and its pre-AEC twin (capture).
    pub fn note_frame(&self, turn: TurnId, clean: &[i16], capture: Option<&[i16]>) {
        let mut inner = self.lock();
        let dump = inner.turns.entry(turn.0).or_default();
        dump.clean.extend_from_slice(clean);
        dump.capture.extend_from_slice(capture.unwrap_or(clean));
        dump.capture_frames += 1;
    }

    /// PCM Whisper will transcribe.
    pub fn note_utterance(&self, utterance: &Utterance) {
        let mut inner = self.lock();
        let dump = inner.turns.entry(utterance.turn.0).or_default();
        dump.utterance = utterance.pcm();
        dump.utterance_frames = utterance.frames.len();
    }

    /// STT hypothesis.
    pub fn note_stt(&self, turn: TurnId, text: &str) {
        let mut inner = self.lock();
        inner.turns.entry(turn.0).or_default().stt_text = Some(text.to_string());
    }

    /// Full LLM reply (concatenated tokens).
    pub fn note_llm(&self, turn: TurnId, text: String) {
        let mut inner = self.lock();
        inner.turns.entry(turn.0).or_default().llm_text = text;
    }

    /// PCM actually handed to the sink.
    pub fn note_tts(&self, turn: TurnId, samples: &[i16]) {
        let mut inner = self.lock();
        let dump = inner.turns.entry(turn.0).or_default();
        dump.tts.extend_from_slice(samples);
        dump.tts_chunks += 1;
    }

    /// Last audio chunk played; write the dump as completed.
    pub fn complete(&self, turn: TurnId, timings: TurnTimings) -> Result<()> {
        let mut inner = self.lock();
        if let Some(dump) = inner.turns.get_mut(&turn.0) {
            dump.timings = Some(timings);
        }
        write_turn(&self.root, turn.0, TurnOutcome::Completed, &mut inner)
    }

    /// Recoverable provider failure; keep whatever audio/text existed.
    pub fn skip(&self, turn: TurnId) -> Result<()> {
        let mut inner = self.lock();
        write_turn(&self.root, turn.0, TurnOutcome::Skipped, &mut inner)
    }

    /// Barge-in or generation cancel; dump interrupted TTS plus what exists so far.
    pub fn interrupt(&self, turn: TurnId) -> Result<()> {
        let mut inner = self.lock();
        write_turn(&self.root, turn.0, TurnOutcome::Cancelled, &mut inner)
    }

    /// Write every turn that never completed or skipped (shutdown / cancel).
    pub fn finish_open(&self) -> Result<()> {
        let mut inner = self.lock();
        let ids: Vec<u64> = inner.turns.keys().copied().collect();
        for id in ids {
            write_turn(&self.root, id, TurnOutcome::Cancelled, &mut inner)?;
        }
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("turn debug mutex")
    }
}

/// Resolve `--turn-debug [DIR]`: explicit path, else env, else [`DEFAULT_TURN_DEBUG_DIR`].
pub fn resolve_turn_debug_dir(explicit: Option<PathBuf>) -> PathBuf {
    if let Some(path) = explicit {
        if !path.as_os_str().is_empty() {
            return path;
        }
    }
    if let Ok(from_env) = std::env::var(TURN_DEBUG_DIR_ENV) {
        if !from_env.is_empty() {
            return PathBuf::from(from_env);
        }
    }
    PathBuf::from(DEFAULT_TURN_DEBUG_DIR)
}

/// Create `dir` and fail with an actionable error if it cannot be written.
pub fn prepare_turn_debug_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).map_err(|err| turn_debug_io(dir, err))?;
    let meta = fs::metadata(dir).map_err(|err| turn_debug_io(dir, err))?;
    if !meta.is_dir() {
        return Err(Error::Config {
            field: "--turn-debug".into(),
            message: format!(
                "{} exists and is not a directory. Pass a writable directory.",
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
        field: "--turn-debug".into(),
        message: format!(
            "cannot write turn debug files under {}: {err}. Pass a writable directory.",
            dir.display()
        ),
    }
}

fn write_turn(root: &Path, id: u64, outcome: TurnOutcome, inner: &mut Inner) -> Result<()> {
    let Some(dump) = inner.turns.get_mut(&id) else {
        return Ok(());
    };
    if dump.written {
        return Ok(());
    }
    dump.written = true;
    let dir = root.join(format!("turn-{id:03}"));
    fs::create_dir_all(&dir).map_err(|err| turn_debug_io(&dir, err))?;
    write_pcm(&dir.join("capture.wav"), &dump.capture)?;
    write_pcm(&dir.join("clean.wav"), &dump.clean)?;
    write_pcm(&dir.join("utterance.wav"), &dump.utterance)?;
    write_pcm(&dir.join("tts.wav"), &dump.tts)?;
    let sidecar = render_sidecar(id, outcome, dump);
    fs::write(dir.join("turn.json"), sidecar).map_err(|err| turn_debug_io(&dir, err))?;
    Ok(())
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
    let speak = strip_markdown_for_speech(&dump.llm_text);
    let timings = dump.timings.unwrap_or_default();
    format!(
        "{{\n  \"turn\": {id},\n  \"outcome\": {},\n  \"stt_text\": {},\n  \"llm_text\": {},\n  \"tts_speak_text\": {},\n  \"timings\": {{\n    \"stt_ms\": {},\n    \"ttft_ms\": {},\n    \"ttfb_ms\": {},\n    \"total_ms\": {}\n  }},\n  \"capture_samples\": {},\n  \"capture_frames\": {},\n  \"capture_duration_ms\": {},\n  \"clean_samples\": {},\n  \"clean_frames\": {},\n  \"clean_duration_ms\": {},\n  \"utterance_samples\": {},\n  \"utterance_frames\": {},\n  \"utterance_duration_ms\": {},\n  \"tts_samples\": {},\n  \"tts_chunks\": {},\n  \"tts_duration_ms\": {}\n}}\n",
        json_string(outcome.as_str()),
        json_string(stt),
        json_string(&dump.llm_text),
        json_string(&speak),
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
    fn resolve_prefers_explicit_then_env_then_default() {
        let explicit = PathBuf::from("/tmp/explicit-turns");
        assert_eq!(resolve_turn_debug_dir(Some(explicit.clone())), explicit);
        assert_eq!(
            resolve_turn_debug_dir(Some(PathBuf::from(""))),
            PathBuf::from(DEFAULT_TURN_DEBUG_DIR)
        );
        assert_eq!(
            resolve_turn_debug_dir(None),
            PathBuf::from(DEFAULT_TURN_DEBUG_DIR)
        );
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
        debug.note_stt(turn, "hello");
        debug.note_llm(turn, "hi **there**".into());
        debug.note_tts(turn, &[9, 8, 7]);
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
        debug.note_stt(TurnId(1), "partial");
        debug.skip(TurnId(1)).unwrap();
        debug.start_turn(TurnId(2));
        debug.finish_open().unwrap();
        let skipped = fs::read_to_string(dir.join("turn-001").join("turn.json")).unwrap();
        assert!(skipped.contains("skipped"));
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
        debug.note_tts(TurnId(0), &[1, 2, 3]);
        debug.interrupt(TurnId(0)).unwrap();
        let json = fs::read_to_string(dir.join("turn-000").join("turn.json")).unwrap();
        assert!(json.contains("cancelled"));
        let wav = read_wav(Cursor::new(
            fs::read(dir.join("turn-000").join("tts.wav")).unwrap(),
        ))
        .unwrap();
        assert_eq!(wav.samples, vec![1, 2, 3]);
        debug.finish_open().unwrap();
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn json_string_escapes_quotes() {
        assert_eq!(json_string("a\"b\\c"), "\"a\\\"b\\\\c\"");
    }
}
