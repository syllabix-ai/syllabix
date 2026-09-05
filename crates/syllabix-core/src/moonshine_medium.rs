//! Official Moonshine streaming-medium STT via the Moonshine C API
//! (`libmoonshine`) and HF `moonshine-ai/moonshine-streaming` `onnx/medium`
//! assets.
//!
//! Layout is incompatible with the Mazino INT8 encoder/decoder/decoder_with_past
//! graphs used by streaming-small (and by competing PR #101): the official
//! medium pin is frontend + encoder + adapter + cross_kv + decoder_kv plus
//! `streaming_config.json` and BinTokenizer `tokenizer.bin`.

use std::fs;
use std::path::{Path, PathBuf};

use syllabix_native::MoonshineContext;

use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::moonshine::{validate_language, LANGUAGE, PROVIDER_NAME};
use crate::moonshine_tokenizer_bin;
use crate::providers::Stt;
use crate::types::{AudioFrame, Transcript, TurnId, Utterance};
use crate::Cancel;

pub const FRONTEND_ASSET: &str = "moonshine-official-medium-frontend";
pub const ENCODER_ASSET: &str = "moonshine-official-medium-encoder";
pub const ADAPTER_ASSET: &str = "moonshine-official-medium-adapter";
pub const CROSS_KV_ASSET: &str = "moonshine-official-medium-cross-kv";
pub const DECODER_KV_ASSET: &str = "moonshine-official-medium-decoder-kv";
pub const CONFIG_ASSET: &str = "moonshine-official-medium-config";
pub const TOKENIZER_JSON_ASSET: &str = "moonshine-official-medium-tokenizer-json";

const PARTIAL_EVERY_FRAMES: usize = 16;
const MODEL_DIR_NAME: &str = "moonshine-official-medium";

fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: PROVIDER_NAME,
        message: message.into(),
    }
}

/// Official medium adapter: C API finalize (+ provisional partials).
pub struct MoonshineMediumStt {
    ctx: MoonshineContext,
    #[allow(dead_code)]
    model_dir: PathBuf,
    language: String,
    active_turn: Option<TurnId>,
    active_frames: Vec<AudioFrame>,
    frames_since_partial: usize,
    last_partial: String,
}

impl MoonshineMediumStt {
    /// Fetch official medium assets, materialize a C-API model directory, and
    /// load `libmoonshine`.
    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Self> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if !MoonshineContext::available() {
            return Err(provider(
                "libmoonshine is not available; build it (see scripts/build-libmoonshine.sh) and set SYLLABIX_LIBMOONSHINE / LD_LIBRARY_PATH",
            ));
        }
        let ids = [
            FRONTEND_ASSET,
            ENCODER_ASSET,
            ADAPTER_ASSET,
            CROSS_KV_ASSET,
            DECODER_KV_ASSET,
            CONFIG_ASSET,
            TOKENIZER_JSON_ASSET,
        ];
        let mut resolved = Vec::with_capacity(ids.len());
        for id in ids {
            let asset = cache
                .manifest()
                .asset(id)
                .ok_or_else(|| Error::ModelCache {
                    message: format!("manifest does not contain the {id} asset"),
                })?;
            resolved.push((id, cache.resolve(asset, fetcher, progress, cancel)?));
        }
        let model_dir = prepare_model_dir(cache, &resolved)?;
        let ctx = MoonshineContext::load(&model_dir).map_err(|err| match err {
            syllabix_native::DecodeError::Cancelled => Error::Cancelled,
            syllabix_native::DecodeError::Failed(message) => provider(message),
        })?;
        Ok(Self {
            ctx,
            model_dir,
            language: LANGUAGE.to_string(),
            active_turn: None,
            active_frames: Vec::new(),
            frames_since_partial: 0,
            last_partial: String::new(),
        })
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Result<Self> {
        let language = language.into();
        validate_language(&language)?;
        self.language = language;
        Ok(self)
    }

    fn transcribe_pcm(&mut self, pcm_f32: &[f32], cancel: &Cancel) -> Result<String> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        self.ctx.transcribe(pcm_f32).map_err(|err| match err {
            syllabix_native::DecodeError::Cancelled => Error::Cancelled,
            syllabix_native::DecodeError::Failed(message) => provider(message),
        })
    }

    fn clear_active(&mut self, turn: TurnId) {
        if self.active_turn == Some(turn) {
            self.active_turn = None;
            self.active_frames.clear();
            self.frames_since_partial = 0;
            self.last_partial.clear();
        }
    }
}

fn prepare_model_dir(cache: &ModelCache, resolved: &[(&str, PathBuf)]) -> Result<PathBuf> {
    let dir = cache.models_dir().join(MODEL_DIR_NAME);
    fs::create_dir_all(&dir).map_err(|err| {
        provider(format!(
            "could not create moonshine model dir {}: {err}",
            dir.display()
        ))
    })?;
    let map = [
        (FRONTEND_ASSET, "frontend.ort"),
        (ENCODER_ASSET, "encoder.ort"),
        (ADAPTER_ASSET, "adapter.ort"),
        (CROSS_KV_ASSET, "cross_kv.ort"),
        (DECODER_KV_ASSET, "decoder_kv.ort"),
        (CONFIG_ASSET, "streaming_config.json"),
        (TOKENIZER_JSON_ASSET, "tokenizer.json"),
    ];
    for (id, canonical) in map {
        let src = resolved
            .iter()
            .find(|(asset_id, _)| *asset_id == id)
            .map(|(_, path)| path)
            .ok_or_else(|| provider(format!("missing resolved asset {id}")))?;
        let dest = dir.join(canonical);
        link_or_copy(src, &dest)?;
    }
    let json = dir.join("tokenizer.json");
    let bin = dir.join("tokenizer.bin");
    if !bin.exists() {
        moonshine_tokenizer_bin::write_bin_beside_json(&json, &bin)?;
    }
    Ok(dir)
}

fn link_or_copy(src: &Path, dest: &Path) -> Result<()> {
    if src == dest {
        return Ok(());
    }
    if let (Ok(a), Ok(b)) = (fs::canonicalize(src), fs::canonicalize(dest)) {
        if a == b {
            return Ok(());
        }
    }
    if dest.exists() {
        // Refresh when the cached source is newer / different size.
        let src_meta = fs::metadata(src).ok();
        let dest_meta = fs::metadata(dest).ok();
        if let (Some(s), Some(d)) = (src_meta, dest_meta) {
            if s.len() == d.len() {
                return Ok(());
            }
        }
        let _ = fs::remove_file(dest);
    }
    #[cfg(unix)]
    {
        if std::os::unix::fs::symlink(src, dest).is_ok() {
            return Ok(());
        }
    }
    fs::copy(src, dest).map_err(|err| {
        provider(format!(
            "could not install {} -> {}: {err}",
            src.display(),
            dest.display()
        ))
    })?;
    Ok(())
}

impl Stt for MoonshineMediumStt {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(provider("utterance has no frames"));
        }
        let pcm = utterance.pcm();
        let audio: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
        let text = self.transcribe_pcm(&audio, cancel)?;
        self.clear_active(utterance.turn);
        Ok(Transcript {
            turn: utterance.turn,
            text,
            language: self.language.clone(),
        })
    }

    fn supports_partials(&self) -> bool {
        true
    }

    fn start_turn(&mut self, turn: TurnId, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        self.active_turn = Some(turn);
        self.active_frames.clear();
        self.frames_since_partial = 0;
        self.last_partial.clear();
        Ok(())
    }

    fn push_frame(&mut self, frame: &AudioFrame, cancel: &Cancel) -> Result<Option<String>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if self.active_turn.is_none() {
            return Ok(None);
        }
        self.active_frames.push(frame.clone());
        self.frames_since_partial += 1;
        if self.frames_since_partial < PARTIAL_EVERY_FRAMES {
            return Ok(None);
        }
        self.frames_since_partial = 0;
        let utterance = Utterance {
            turn: self.active_turn.expect("active turn checked"),
            frames: self.active_frames.clone(),
        };
        let pcm = utterance.pcm();
        let audio: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
        let text = self.transcribe_pcm(&audio, cancel)?;
        if !text.is_empty() && text.starts_with(&self.last_partial) && text != self.last_partial {
            self.last_partial = text.clone();
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }

    fn cancel_turn(&mut self, turn: TurnId) {
        self.clear_active(turn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_ids_are_stable() {
        assert_eq!(FRONTEND_ASSET, "moonshine-official-medium-frontend");
        assert_eq!(
            TOKENIZER_JSON_ASSET,
            "moonshine-official-medium-tokenizer-json"
        );
    }

    /// Full weights + libmoonshine gate. Point `SYLLABIX_MOONSHINE_OFFICIAL_MEDIUM_DIR`
    /// at a directory already containing canonical C-API names (or let from_cache
    /// build one), and set `SYLLABIX_LIBMOONSHINE`.
    #[test]
    #[ignore]
    fn native_recorded_fixtures_meet_word_match_gate() {
        let dir = std::env::var("SYLLABIX_MOONSHINE_OFFICIAL_MEDIUM_DIR")
            .expect("SYLLABIX_MOONSHINE_OFFICIAL_MEDIUM_DIR");
        let mut stt = {
            let ctx = MoonshineContext::load(Path::new(&dir)).expect("load");
            // Wrap via a temporary shim: construct through public fields path.
            MoonshineMediumStt {
                ctx,
                model_dir: PathBuf::from(&dir),
                language: LANGUAGE.to_string(),
                active_turn: None,
                active_frames: Vec::new(),
                frames_since_partial: 0,
                last_partial: String::new(),
            }
        };
        let fixtures: &[(&str, &[&str], f64)] = &[
            (
                "jfk.wav",
                &[
                    "and",
                    "so",
                    "my",
                    "fellow",
                    "americans",
                    "ask",
                    "not",
                    "what",
                    "your",
                    "country",
                    "can",
                    "do",
                    "for",
                    "you",
                    "ask",
                    "what",
                    "you",
                    "can",
                    "do",
                    "for",
                    "your",
                    "country",
                ],
                0.8,
            ),
            (
                "librispeech-1089-134686-0000.wav",
                &[
                    "he", "hoped", "there", "would", "be", "stew", "for", "dinner", "turnips",
                    "and", "carrots", "and", "bruised", "potatoes", "and", "fat", "mutton",
                    "pieces", "to", "be", "ladled", "out", "in", "thick", "peppered", "flour",
                    "fattened", "sauce",
                ],
                0.8,
            ),
            (
                "librispeech-121-127105-0009.wav",
                &["she", "has", "been", "dead", "these", "twenty", "years"],
                0.8,
            ),
            (
                "librispeech-1995-1837-0005.wav",
                &[
                    "she", "was", "so", "strange", "and", "human", "a", "creature",
                ],
                0.8,
            ),
        ];
        for (index, (file, expected, minimum)) in fixtures.iter().enumerate() {
            let wav = std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/stt")
                    .join(file),
            )
            .unwrap_or_else(|err| panic!("{file}: {err}"));
            let pcm = crate::audio::read_wav(std::io::Cursor::new(wav))
                .unwrap_or_else(|err| panic!("{file} parse: {err}"));
            let frames = crate::audio::record_fixture_to_frames(&pcm)
                .unwrap_or_else(|err| panic!("{file} frames: {err}"));
            let transcript = stt
                .transcribe(
                    &Utterance {
                        turn: TurnId(index as u64),
                        frames,
                    },
                    &Cancel::new(),
                )
                .unwrap_or_else(|err| panic!("{file} tx: {err}"));
            let ratio = crate::stt::word_match_ratio(&transcript.text, expected);
            eprintln!(
                "Moonshine official medium {file}: {:.1}% ({:?})",
                ratio * 100.0,
                transcript.text
            );
            assert!(
                ratio >= *minimum,
                "{file} {:?} matched {:.1}% (need {:.0}%)",
                transcript.text,
                ratio * 100.0,
                minimum * 100.0
            );
        }
        // silence unused warning when constructing via struct literal
        let _ = stt.language();
    }
}
