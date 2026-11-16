//! Text-to-speech engines and the [`Tts`] trait they share.
//!
//! Turn engines on with Cargo features. Defaults are `pocket`, `kokoro`, and
//! `remote` (ONNX plus HTTP). `qwen3` is opt-in. With `default-features = false`
//! and only `remote`, this crate builds without CMake, a C++ compiler, or ONNX
//! Runtime.
//!
//! Default ONNX engines use the `ort` crate, which may download a prebuilt ONNX
//! Runtime at build time; some targets have no binary. For HTTP only, use
//! `default-features = false` with the `remote` feature.
//!
//! | Feature | Engine | Needs C++? | Model ids |
//! | --- | --- | --- | --- |
//! | `pocket` | Pocket TTS ONNX (`pocket`) | No (ONNX Runtime) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//! | `kokoro` | Kokoro-82M ONNX ([`KokoroTts`]) | No (ONNX Runtime) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//! | `remote` | OpenAI-compatible `/audio/speech` (`remote`) | No (HTTP only) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//! | `qwen3` | Qwen3-TTS ([`QwenTts`]); Indic-Mio ([`IndicMioTts`]) needs `qwen3` + `kokoro` | Yes (CMake) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//!
//! C++ engines need the host toolchain in
//! [embed — Host compile](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#host-compile).
//! Without `qwen3` its type still exists and returns an error when loaded. The
//! remote engine reads its key from `SYLLABIX_TTS_API_KEY` only. Examples:
//! [`speak-wav-local`](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-tts/examples/speak-wav-local.rs)
//! (Pocket) and
//! [`speak-wav`](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-tts/examples/speak-wav.rs)
//! (remote).

// The types in the `Tts` signatures, so callers need no other crate.
pub use syllabix_native::audio;
pub use syllabix_native::models::{
    cache_root, BlockedFetcher, Fetcher, HttpFetcher, Manifest, ModelAsset, ModelCache, NoProgress,
    Progress, StderrProgress,
};
pub use syllabix_native::types::{
    GenerationId, SynthesizedAudio, TokenChunk, TurnId, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ,
    FRAME_SAMPLES,
};
pub use syllabix_native::{Cancel, Error, Result};

mod engines;
#[cfg(all(feature = "qwen3", any(feature = "kokoro", feature = "pocket")))]
mod indic_mio;
#[cfg(all(feature = "qwen3", any(feature = "kokoro", feature = "pocket")))]
pub use indic_mio::{MioDecoder, INDIC_MIO_TEST_SEED};
#[cfg(feature = "kokoro")]
pub mod g2p;
#[cfg(feature = "pocket")]
pub mod pocket;
#[cfg(feature = "remote")]
pub mod remote;
pub mod speech_text;

pub use engines::*;

/// Streaming text-to-speech.
pub trait Tts: Send {
    /// Config name (`kokoro`).
    fn name(&self) -> &'static str;

    /// Weight id for diagnostics sidecars. `None` means the fake and test
    /// providers; live engines report the loaded asset id.
    fn model_id(&self) -> Option<&str> {
        None
    }

    /// Compute backend confirmed at load (`cpu` / `metal` / `vulkan`).
    /// `None` for engines that have no adaptive placement decision.
    fn backend_id(&self) -> Option<&str> {
        None
    }

    /// Synthesize buffered tokens. May return no chunks until a sentence ends.
    ///
    /// Pipeline calls this per streamed token. The last chunk of a generation
    /// must set `is_last = true` so the loop can complete the turn.
    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>>;

    /// Streaming form of [`Self::synthesize_chunk`]. Existing providers keep
    /// their buffered implementation; native Qwen overrides this to release
    /// vocoder windows while generation is still running.
    fn synthesize_chunk_into(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        for audio in self.synthesize_chunk(token, cancel)? {
            on_audio(audio)?;
        }
        Ok(())
    }
}

/// TTS model menu. Pocket TTS is the default;
/// Kokoro and the Qwen3-TTS backbones are optional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsModel {
    /// Kokoro ONNX (`af_heart`, English).
    Kokoro,
    /// Qwen3-TTS-12Hz-0.6B-Base GGUF (~344 MB fetch).
    Qwen06,
    /// Qwen3-TTS-12Hz-1.7B-Base GGUF.
    Qwen17,
    /// Pocket TTS English ONNX graph set.
    PocketTts,
    /// Indic-Mio Qwen3 GGUF + MioCodec 24 kHz decoder.
    IndicMio,
}

impl TtsModel {
    /// Every menu id, in documentation order.
    pub const ALL: [TtsModel; 5] = [
        TtsModel::Kokoro,
        TtsModel::Qwen06,
        TtsModel::Qwen17,
        TtsModel::PocketTts,
        TtsModel::IndicMio,
    ];

    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kokoro => "kokoro",
            Self::Qwen06 => "qwen3-0.6",
            Self::Qwen17 => "qwen3-1.7",
            Self::PocketTts => "pocket-tts",
            Self::IndicMio => "indic-mio",
        }
    }

    /// Parse a model menu id.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == value)
    }

    /// Manifest asset holding this model's primary weights.
    pub fn asset_id(self) -> &'static str {
        match self {
            Self::Kokoro => KOKORO_ASSET,
            Self::Qwen06 => QWEN_TTS_06B_ASSET,
            Self::Qwen17 => QWEN_TTS_ASSET,
            // Same id as `pocket::POCKET_TTS_TEXT_CONDITIONER_ASSET` (checked in
            // pocket's tests); spelled out so this builds without `pocket`.
            Self::PocketTts => "pocket-tts-text-conditioner",
            Self::IndicMio => INDIC_MIO_ASSET,
        }
    }

    /// Manifest id of the matching speech-tokenizer projector (qwen ids
    /// only; Kokoro has none).
    pub fn mmproj_asset_id(self) -> Option<&'static str> {
        match self {
            Self::Kokoro => None,
            Self::Qwen06 => Some(QWEN_TTS_06B_MMPROJ_ASSET),
            Self::Qwen17 => Some(QWEN_TTS_MMPROJ_ASSET),
            Self::PocketTts => None,
            Self::IndicMio => None,
        }
    }

    /// Turn-debug / log identity of the loaded weights.
    pub fn model_id(self) -> &'static str {
        match self {
            Self::Kokoro => KOKORO_ASSET,
            Self::Qwen06 => "qwen3-tts-0.6b-base",
            Self::Qwen17 => "qwen3-tts-1.7b-base",
            Self::PocketTts => "pocket-tts",
            Self::IndicMio => "indic-mio-q8",
        }
    }
}

/// Synthesize `text` to 16 kHz mono PCM samples.
pub fn speak_to_samples(
    tts: &mut impl Tts,
    text: impl Into<String>,
    cancel: &Cancel,
) -> Result<Vec<i16>> {
    let token = TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index: 0,
        text: text.into(),
        is_last: true,
    };
    Ok(tts
        .synthesize_chunk(&token, cancel)?
        .into_iter()
        .flat_map(|chunk| chunk.samples)
        .collect())
}

/// Synthesize `text` and write a WAV file at `path`.
pub fn speak_to_wav(
    tts: &mut impl Tts,
    text: impl Into<String>,
    path: impl AsRef<std::path::Path>,
    cancel: &Cancel,
) -> Result<()> {
    let samples = speak_to_samples(tts, text, cancel)?;
    let wav = audio::WavPcm {
        format: audio::PcmFormat::v0(),
        samples,
    };
    audio::write_wav(std::fs::File::create(path.as_ref())?, &wav)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_menu_ids_round_trip() {
        let ids: &[&str] = &[
            "kokoro",
            "qwen3-0.6",
            "qwen3-1.7",
            "pocket-tts",
            "indic-mio",
        ];
        assert_eq!(TtsModel::ALL.len(), ids.len());
        for (model, id) in TtsModel::ALL.into_iter().zip(ids.iter().copied()) {
            assert_eq!(model.as_str(), id);
            assert_eq!(TtsModel::parse(id), Some(model));
            assert!(TtsModel::parse(&format!("{id}-nope")).is_none());
        }
        // Provider names are not valid model identifiers.
        for bad in ["qwen", "pansori", "neutts"] {
            assert!(TtsModel::parse(bad).is_none(), "{bad}");
        }
        assert_eq!(TtsModel::Kokoro.asset_id(), KOKORO_ASSET);
        assert_eq!(TtsModel::Qwen06.asset_id(), QWEN_TTS_06B_ASSET);
        assert_eq!(TtsModel::Qwen17.asset_id(), QWEN_TTS_ASSET);
        assert_eq!(TtsModel::Kokoro.mmproj_asset_id(), None);
        assert_eq!(
            TtsModel::Qwen06.mmproj_asset_id(),
            Some(QWEN_TTS_06B_MMPROJ_ASSET)
        );
        assert_eq!(
            TtsModel::Qwen17.mmproj_asset_id(),
            Some(QWEN_TTS_MMPROJ_ASSET)
        );
        assert_eq!(TtsModel::PocketTts.mmproj_asset_id(), None);
        assert_eq!(TtsModel::Kokoro.model_id(), KOKORO_ASSET);
        assert_eq!(TtsModel::Qwen06.model_id(), "qwen3-tts-0.6b-base");
        assert_eq!(TtsModel::Qwen17.model_id(), "qwen3-tts-1.7b-base");
        assert_eq!(TtsModel::PocketTts.model_id(), "pocket-tts");
        assert_eq!(TtsModel::parse("indic-mio"), Some(TtsModel::IndicMio));
        assert_eq!(TtsModel::IndicMio.as_str(), "indic-mio");
        assert_eq!(TtsModel::IndicMio.asset_id(), INDIC_MIO_ASSET);
        assert_eq!(TtsModel::IndicMio.mmproj_asset_id(), None);
        assert_eq!(TtsModel::IndicMio.model_id(), "indic-mio-q8");
        assert!(TtsModel::ALL.contains(&TtsModel::IndicMio));
    }

    #[test]
    fn speak_to_samples_returns_flat_pcm() {
        struct OneChunkTts;
        impl Tts for OneChunkTts {
            fn name(&self) -> &'static str {
                "one-chunk"
            }
            fn synthesize_chunk(
                &mut self,
                token: &TokenChunk,
                _cancel: &Cancel,
            ) -> Result<Vec<SynthesizedAudio>> {
                Ok(vec![SynthesizedAudio {
                    turn: token.turn,
                    generation: token.generation,
                    index: 0,
                    samples: vec![1, 2, 3],
                    is_last: true,
                }])
            }
        }
        let samples = speak_to_samples(&mut OneChunkTts, "hi", &Cancel::new()).expect("samples");
        assert_eq!(samples, vec![1, 2, 3]);
    }

    #[test]
    fn speak_to_wav_rejects_missing_parent_dir() {
        struct NoopTts;
        impl Tts for NoopTts {
            fn name(&self) -> &'static str {
                "noop"
            }
            fn synthesize_chunk(
                &mut self,
                _token: &TokenChunk,
                _cancel: &Cancel,
            ) -> Result<Vec<SynthesizedAudio>> {
                Ok(Vec::new())
            }
        }
        let path =
            std::env::temp_dir().join(format!("syllabix-tts-missing-out-{}", std::process::id()));
        let nested = path.join("no/such/dir/out.wav");
        let err = speak_to_wav(&mut NoopTts, "hi", &nested, &Cancel::new()).expect_err("bad path");
        assert!(err.to_string().contains("No such file") || err.to_string().contains("os error"));
    }
}
