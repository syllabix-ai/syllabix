//! Coverage-only API surface. Native inference tests are excluded under
//! `cfg(coverage)` because they load large external model artifacts.

use crate::models::{Fetcher, ModelCache, Progress};
use crate::{Cancel, Result, SynthesizedAudio, TokenChunk};

pub const POCKET_TTS_BUNDLE_ASSET: &str = "pocket-tts-bundle";
pub const POCKET_TTS_BOS_ASSET: &str = "pocket-tts-bos";
pub const POCKET_TTS_TOKENIZER_ASSET: &str = "pocket-tts-tokenizer";
pub const POCKET_TTS_TEXT_CONDITIONER_ASSET: &str = "pocket-tts-text-conditioner";
pub const POCKET_TTS_FLOW_MAIN_ASSET: &str = "pocket-tts-flow-main";
pub const POCKET_TTS_FLOW_ASSET: &str = "pocket-tts-flow";
pub const POCKET_TTS_MIMI_DECODER_ASSET: &str = "pocket-tts-mimi-decoder";
pub const POCKET_TTS_VOICE_ASSET: &str = "pocket-tts-voice-alba";

/// Lightweight stand-in used only by llvm-cov. It retains the public TTS
/// API so configuration code can select the native provider without loading
/// its ONNX graphs or model cache in the coverage build.
pub struct PocketTts;

impl PocketTts {
    pub fn from_cache(
        _cache: &ModelCache,
        _fetcher: &dyn Fetcher,
        _progress: &mut dyn Progress,
        _cancel: &Cancel,
    ) -> Result<Self> {
        Ok(Self)
    }
}

impl crate::providers::Tts for PocketTts {
    fn name(&self) -> &'static str {
        "local"
    }

    fn model_id(&self) -> Option<&str> {
        Some("pocket-tts")
    }

    fn synthesize_chunk(
        &mut self,
        token: &TokenChunk,
        cancel: &Cancel,
    ) -> Result<Vec<SynthesizedAudio>> {
        if cancel.is_stale(token.generation) || cancel.is_shutdown() {
            return Err(crate::Error::Cancelled);
        }
        Ok(vec![SynthesizedAudio {
            turn: token.turn,
            generation: token.generation,
            index: token.index,
            samples: vec![0; 16],
            is_last: token.is_last,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Tts;
    use crate::{GenerationId, TurnId};

    fn token() -> TokenChunk {
        TokenChunk {
            turn: TurnId(1),
            generation: GenerationId(0),
            index: 0,
            text: "coverage".into(),
            is_last: true,
        }
    }

    #[test]
    fn stand_in_preserves_provider_identity_and_final_chunk() {
        let mut tts = PocketTts;
        let chunks = tts.synthesize_chunk(&token(), &Cancel::new()).unwrap();
        assert_eq!(tts.name(), "local");
        assert_eq!(tts.model_id(), Some("pocket-tts"));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_last);
    }

    #[test]
    fn stand_in_observes_generation_cancellation() {
        let mut tts = PocketTts;
        let cancel = Cancel::new();
        cancel.cancel_generation();
        assert!(matches!(
            tts.synthesize_chunk(&token(), &cancel),
            Err(crate::Error::Cancelled)
        ));
    }
}
