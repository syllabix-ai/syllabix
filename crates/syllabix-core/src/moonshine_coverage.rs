//! Coverage-only Moonshine API surface.
//!
//! The production adapter loads three ONNX graphs and a tokenizer. llvm-cov
//! deliberately does not download model artifacts, so this stand-in exercises
//! provider selection and streaming contracts without native inference.

use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::Stt;
use crate::{AudioFrame, Cancel, Error, Result, Transcript, TurnId, Utterance};

pub const PROVIDER_NAME: &str = "moonshine";
pub const LANGUAGE: &str = "en";
pub const ENCODER_ASSET: &str = "moonshine-encoder";
pub const DECODER_ASSET: &str = "moonshine-decoder";
pub const DECODER_PAST_ASSET: &str = "moonshine-decoder-past";
pub const TOKENIZER_ASSET: &str = "moonshine-tokenizer";

pub const MEDIUM_ENCODER_ASSET: &str = "moonshine-medium-encoder";
pub const MEDIUM_DECODER_ASSET: &str = "moonshine-medium-decoder";
pub const MEDIUM_DECODER_PAST_ASSET: &str = "moonshine-medium-decoder-past";
pub const MEDIUM_TOKENIZER_ASSET: &str = "moonshine-medium-tokenizer";

/// Placeholder for the native tokenizer type, retained for the public API.
pub struct MoonshineTokenizer;

/// Lightweight `cfg(coverage)` replacement for the native ONNX adapter.
pub struct MoonshineStt {
    active_turn: Option<TurnId>,
}

impl MoonshineStt {
    pub fn from_cache(
        _cache: &ModelCache,
        _fetcher: &dyn Fetcher,
        _progress: &mut dyn Progress,
        cancel: &Cancel,
        model: crate::defaults::SttModel,
    ) -> Result<Self> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if !model.is_moonshine() {
            return Err(Error::Provider {
                provider: PROVIDER_NAME,
                message: format!(
                    "MoonshineStt::from_cache requires a Moonshine model, got {}",
                    model.as_str()
                ),
            });
        }
        Ok(Self { active_turn: None })
    }

    pub fn language(&self) -> &str {
        LANGUAGE
    }
}

impl Stt for MoonshineStt {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(Error::Provider {
                provider: PROVIDER_NAME,
                message: "utterance has no frames".into(),
            });
        }
        self.active_turn = None;
        Ok(Transcript {
            turn: utterance.turn,
            text: "coverage transcript".into(),
            language: LANGUAGE.into(),
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
        Ok(())
    }

    fn push_frame(&mut self, _frame: &AudioFrame, cancel: &Cancel) -> Result<Option<String>> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        Ok(self.active_turn.map(|_| "coverage partial".into()))
    }

    fn cancel_turn(&mut self, turn: TurnId) {
        if self.active_turn == Some(turn) {
            self.active_turn = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::Stt;

    #[test]
    fn stand_in_preserves_moonshine_streaming_identity() {
        let stt = MoonshineStt { active_turn: None };
        assert_eq!(stt.name(), PROVIDER_NAME);
        assert_eq!(stt.language(), LANGUAGE);
        assert!(stt.supports_partials());
        assert_eq!(ENCODER_ASSET, "moonshine-encoder");
        assert_eq!(MEDIUM_ENCODER_ASSET, "moonshine-medium-encoder");
        assert_eq!(MEDIUM_TOKENIZER_ASSET, "moonshine-medium-tokenizer");
    }

    #[test]
    fn stand_in_rejects_cancelled_turn_start() {
        let cancel = Cancel::new();
        cancel.shutdown();
        let mut stt = MoonshineStt { active_turn: None };
        assert!(matches!(
            stt.start_turn(TurnId(1), &cancel),
            Err(Error::Cancelled)
        ));
    }
}
