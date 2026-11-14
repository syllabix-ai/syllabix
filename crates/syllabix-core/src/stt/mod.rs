//! Speech-to-text engines and the [`Stt`] trait they share.

use crate::cancel::Cancel;
use crate::error::Result;
use crate::types::{AudioFrame, Transcript, TurnId, Utterance};

pub(crate) mod language;
pub mod moonshine;
pub mod qwen_asr;
#[cfg(feature = "cloud")]
pub mod remote;
mod whisper;

pub use whisper::*;

/// Speech-to-text for a completed VAD utterance.
pub trait Stt: Send {
    /// Config name (`whisper.cpp`).
    fn name(&self) -> &'static str;

    /// Transcribe one utterance. Must check `cancel` and return [`crate::Error::Cancelled`].
    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript>;

    /// Whether this engine can produce live text while VAD owns an active
    /// turn. The default keeps the established utterance-final Whisper path.
    fn supports_partials(&self) -> bool {
        false
    }

    /// VAD opened `turn`. This is notification only: VAD remains the single
    /// owner of start/end timing and the shared `end_silence_ms` setting.
    fn start_turn(&mut self, _turn: TurnId, _cancel: &Cancel) -> Result<()> {
        Ok(())
    }

    /// One post-AEC frame from an active VAD turn. Returning text updates the
    /// provisional transcript; returning `None` means no visible change.
    fn push_frame(&mut self, _frame: &AudioFrame, _cancel: &Cancel) -> Result<Option<String>> {
        Ok(None)
    }

    /// Forget provisional state for a turn abandoned by cancellation.
    fn cancel_turn(&mut self, _turn: TurnId) {}
}

/// whisper.cpp model menu. One provider exposes several GGML sizes;
/// `whisper-small` is the default. `-q5_0` ids are the published quantizations of their
/// fp16 siblings (`tiny` / `base` are deliberately not offered).
/// `MoonshineStreamingSmall` / `MoonshineStreamingMedium` are Moonshine ONNX engine ids.
/// `QwenAsr06` is Qwen3-ASR 0.6B through llama.cpp mtmd.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttModel {
    /// `whisper-small` multilingual weights used by default.
    Small,
    /// `whisper-medium` multilingual weights.
    Medium,
    /// `whisper-large-v3-turbo` weights.
    LargeV3Turbo,
    /// Published `medium` q5_0 quantization.
    MediumQ5_0,
    /// Published `large-v3-turbo` q5_0 quantization.
    LargeV3TurboQ5_0,
    /// Moonshine streaming-small INT8 ONNX (English only).
    MoonshineStreamingSmall,
    /// Moonshine streaming-medium INT8 ONNX (English only).
    MoonshineStreamingMedium,
    /// Qwen3-ASR 0.6B via llama.cpp mtmd (finalize only).
    QwenAsr06,
}

impl SttModel {
    /// Every yaml-selectable id, manifest order.
    pub const ALL: [SttModel; 8] = [
        SttModel::Small,
        SttModel::Medium,
        SttModel::LargeV3Turbo,
        SttModel::MediumQ5_0,
        SttModel::LargeV3TurboQ5_0,
        SttModel::MoonshineStreamingSmall,
        SttModel::MoonshineStreamingMedium,
        SttModel::QwenAsr06,
    ];

    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "whisper-small",
            Self::Medium => "whisper-medium",
            Self::LargeV3Turbo => "whisper-large-v3-turbo",
            Self::MediumQ5_0 => "whisper-medium-q5_0",
            Self::LargeV3TurboQ5_0 => "whisper-large-v3-turbo-q5_0",
            Self::MoonshineStreamingSmall => "moonshine-streaming-small",
            Self::MoonshineStreamingMedium => "moonshine-streaming-medium",
            Self::QwenAsr06 => "qwen3-asr-0.6",
        }
    }

    /// Manifest asset id downloaded for this model.
    pub fn asset_id(self) -> &'static str {
        match self {
            Self::Small => "whisper-small",
            Self::Medium => "whisper-medium",
            Self::LargeV3Turbo => "whisper-large-v3-turbo",
            Self::MediumQ5_0 => "whisper-medium-q5_0",
            Self::LargeV3TurboQ5_0 => "whisper-large-v3-turbo-q5_0",
            Self::MoonshineStreamingSmall => moonshine::ENCODER_ASSET,
            Self::MoonshineStreamingMedium => moonshine::MEDIUM_ENCODER_ASSET,
            Self::QwenAsr06 => qwen_asr::ASR_ASSET,
        }
    }

    /// True for either Moonshine streaming size (English-only ONNX).
    pub fn is_moonshine(self) -> bool {
        matches!(
            self,
            Self::MoonshineStreamingSmall | Self::MoonshineStreamingMedium
        )
    }

    /// True for the Qwen3-ASR 0.6B mtmd engine.
    pub fn is_qwen_asr(self) -> bool {
        matches!(self, Self::QwenAsr06)
    }

    /// Parse a yaml `pipeline.stt.model` id.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::BuiltinDefaults;

    #[test]
    fn stt_menu_ids_round_trip() {
        let ids = [
            "whisper-small",
            "whisper-medium",
            "whisper-large-v3-turbo",
            "whisper-medium-q5_0",
            "whisper-large-v3-turbo-q5_0",
            "moonshine-streaming-small",
            "moonshine-streaming-medium",
            "qwen3-asr-0.6",
        ];
        for (model, id) in SttModel::ALL.into_iter().zip(ids) {
            assert_eq!(model.as_str(), id);
            assert_eq!(SttModel::parse(id), Some(model));
            assert!(SttModel::parse(&format!("{id}-nope")).is_none());
        }
        assert!(SttModel::parse("tiny").is_none());
        assert!(SttModel::parse("base").is_none());
        assert_eq!(
            BuiltinDefaults::v0().stt_model,
            SttModel::Small,
            "`whisper-small` stays the launch default"
        );
        assert!(SttModel::QwenAsr06.is_qwen_asr());
        assert!(!SttModel::Small.is_qwen_asr());
        assert!(!SttModel::MoonshineStreamingSmall.is_qwen_asr());
        assert!(SttModel::MoonshineStreamingSmall.is_moonshine());
        assert!(!SttModel::QwenAsr06.is_moonshine());
    }
}
