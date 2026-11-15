//! Text-to-speech engines and the [`Tts`] trait they share.
//!
//! Turn engines on with Cargo features. All four are on by default; with
//! `default-features = false` and only `remote`, this crate builds without
//! CMake, a C++ compiler, or ONNX Runtime.
//!
//! | Feature | Engine | Needs C++? | Model ids |
//! | --- | --- | --- | --- |
//! | `pocket` | Pocket TTS ONNX (`pocket`) | No (ONNX Runtime) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//! | `kokoro` | Kokoro-82M ONNX ([`KokoroTts`]) | No (ONNX Runtime) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//! | `qwen3` | Qwen3-TTS ([`QwenTts`]) | Yes (CMake) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//! | `remote` | OpenAI-compatible `/audio/speech` (`remote`) | No (HTTP only) | [engines — TTS](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#tts) |
//!
//! Without `kokoro` or `qwen3` their types still exist and return an error
//! when loaded. The remote engine reads its key from `SYLLABIX_TTS_API_KEY`
//! only. Example: `examples/speak-wav.rs`.

// The types in the `Tts` signatures, so callers need no other crate.
pub use syllabix_native::audio;
pub use syllabix_native::types::{GenerationId, SynthesizedAudio, TokenChunk, TurnId};
pub use syllabix_native::{Cancel, Error, Result};

mod engines;
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

/// TTS model menu under `provider: local`. Pocket TTS is the default;
/// Kokoro and the Qwen3-TTS backbones are YAML opt-ins.
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
}

impl TtsModel {
    /// Every menu id, in documentation order.
    pub const ALL: [TtsModel; 4] = [
        TtsModel::Kokoro,
        TtsModel::Qwen06,
        TtsModel::Qwen17,
        TtsModel::PocketTts,
    ];

    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kokoro => "kokoro",
            Self::Qwen06 => "qwen3-0.6",
            Self::Qwen17 => "qwen3-1.7",
            Self::PocketTts => "pocket-tts",
        }
    }

    /// Parse a yaml `pipeline.tts.model` id.
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
        }
    }

    /// Turn-debug / log identity of the loaded weights.
    pub fn model_id(self) -> &'static str {
        match self {
            Self::Kokoro => KOKORO_ASSET,
            Self::Qwen06 => "qwen3-tts-0.6b-base",
            Self::Qwen17 => "qwen3-tts-1.7b-base",
            Self::PocketTts => "pocket-tts",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_menu_ids_round_trip() {
        let ids = ["kokoro", "qwen3-0.6", "qwen3-1.7", "pocket-tts"];
        for (model, id) in TtsModel::ALL.into_iter().zip(ids) {
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
    }
}
