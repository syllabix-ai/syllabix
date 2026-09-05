//! Built-in providers, models, and queue sizes used when no configuration file exists.

use crate::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES};

/// Fixed queue bounds. Stages block on send instead of growing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueCaps {
    /// Mic frames waiting for VAD.
    pub frames: usize,
    /// Completed utterances waiting for final-only STT.
    pub utterances: usize,
    /// Transcripts waiting for the LLM.
    pub transcripts: usize,
    /// Streamed tokens waiting for TTS.
    pub tokens: usize,
    /// Synthesized chunks waiting for playback.
    pub audio: usize,
}

impl QueueCaps {
    /// Defaults sized to expose unbounded buffering while keeping each stage fed.
    pub const fn v0() -> Self {
        Self {
            frames: 32,
            utterances: 4,
            transcripts: 4,
            tokens: 32,
            audio: 16,
        }
    }
}

impl Default for QueueCaps {
    fn default() -> Self {
        Self::v0()
    }
}

/// Voice-activity detector implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadProvider {
    /// Silero ONNX. Fake loop uses the same name.
    Silero,
}

impl VadProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Silero => "silero",
        }
    }
}

/// Speech-to-text implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttProvider {
    /// In-process STT. The selected model chooses Whisper or Moonshine.
    Local,
}

impl SttProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
        }
    }
}

/// whisper.cpp model menu. One provider exposes several GGML sizes;
/// `whisper-small` is the default. `-q5_0` ids are the published quantizations of their
/// fp16 siblings (`tiny` / `base` are deliberately not offered).
/// `MoonshineStreamingSmall` / `MoonshineStreamingMedium` are Moonshine engine ids.
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
    /// Official Moonshine streaming-medium via C API (English only).
    MoonshineStreamingMedium,
}

impl SttModel {
    /// Every yaml-selectable id, manifest order.
    pub const ALL: [SttModel; 7] = [
        SttModel::Small,
        SttModel::Medium,
        SttModel::LargeV3Turbo,
        SttModel::MediumQ5_0,
        SttModel::LargeV3TurboQ5_0,
        SttModel::MoonshineStreamingSmall,
        SttModel::MoonshineStreamingMedium,
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
            Self::MoonshineStreamingSmall => crate::moonshine::ENCODER_ASSET,
            Self::MoonshineStreamingMedium => "moonshine-official-medium-frontend",
        }
    }

    /// True for either Moonshine streaming size (English-only).
    pub fn is_moonshine(self) -> bool {
        matches!(
            self,
            Self::MoonshineStreamingSmall | Self::MoonshineStreamingMedium
        )
    }

    /// Parse a yaml `pipeline.stt.model` id.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == value)
    }
}

/// LLM execution models. `Local` runs weights in-process (llama.cpp GGUF,
/// first-run cache); `Online` streams from any OpenAI-compatible
/// `chat/completions` endpoint (`pipeline.llm.base_url`, BYO key). The words
/// name the posture — where the user's words go — not the engine; VAD/STT/TTS
/// adopt the same vocabulary in follow-up PRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    /// In-process GGUF engine. No network.
    Local,
    /// OpenAI-compatible remote endpoint.
    Online,
}

impl LlmProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Online => "online",
        }
    }
}

/// TTS execution models. `Local` runs weights in-process — Kokoro ONNX or
/// Qwen3-TTS through the shared llama.cpp/ggml path; `Online` is reserved for
/// no online implementation exists and fails fast at config load. The variants describe
/// the posture — where the user's words go — matching [`LlmProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsProvider {
    /// In-process weights. No network.
    Local,
    /// OpenAI-compatible remote endpoint. Not supported yet; reserved.
    Online,
}

impl TtsProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Online => "online",
        }
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
            Self::Kokoro => crate::tts::KOKORO_ASSET,
            Self::Qwen06 => crate::tts::QWEN_TTS_06B_ASSET,
            Self::Qwen17 => crate::tts::QWEN_TTS_ASSET,
            Self::PocketTts => crate::pocket_tts::POCKET_TTS_TEXT_CONDITIONER_ASSET,
        }
    }

    /// Manifest id of the matching speech-tokenizer projector (qwen ids
    /// only; Kokoro has none).
    pub fn mmproj_asset_id(self) -> Option<&'static str> {
        match self {
            Self::Kokoro => None,
            Self::Qwen06 => Some(crate::tts::QWEN_TTS_06B_MMPROJ_ASSET),
            Self::Qwen17 => Some(crate::tts::QWEN_TTS_MMPROJ_ASSET),
            Self::PocketTts => None,
        }
    }

    /// Turn-debug / log identity of the loaded weights.
    pub fn model_id(self) -> &'static str {
        match self {
            Self::Kokoro => crate::tts::KOKORO_ASSET,
            Self::Qwen06 => "qwen3-tts-0.6b-base",
            Self::Qwen17 => "qwen3-tts-1.7b-base",
            Self::PocketTts => "pocket-tts",
        }
    }
}

/// Stack used when YAML is missing and written by `init`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinDefaults {
    /// Agent name for a generated yaml.
    pub name: &'static str,
    /// VAD provider.
    pub vad: VadProvider,
    /// STT provider.
    pub stt: SttProvider,
    /// STT model.
    pub stt_model: SttModel,
    /// LLM provider.
    pub llm: LlmProvider,
    /// Instruct GGUF id. Fetched into the first-run cache; not packed.
    pub llm_model: &'static str,
    /// Qwen thinking. Off by default; yaml `thinking: true` enables it on
    /// `qwen3.5-2b` only.
    pub llm_thinking: bool,
    /// TTS provider.
    pub tts: TtsProvider,
    /// TTS model (`pocket-tts` default; other local models are opt-in).
    pub tts_model: TtsModel,
    /// STT language code used when YAML does not override it.
    pub language: &'static str,
    /// Capture/playback sample rate.
    pub sample_rate_hz: u32,
    /// Capture/playback channels.
    pub channels: u16,
    /// PCM samples per frame.
    pub frame_samples: usize,
    /// Bounded queue sizes.
    pub queues: QueueCaps,
}

impl BuiltinDefaults {
    /// Built-in values used when configuration is absent.
    pub const fn v0() -> Self {
        Self {
            name: "demo-agent",
            vad: VadProvider::Silero,
            stt: SttProvider::Local,
            stt_model: SttModel::Small,
            llm: LlmProvider::Local,
            llm_model: "lfm2.5-2.6b",
            llm_thinking: false,
            tts: TtsProvider::Local,
            tts_model: TtsModel::PocketTts,
            language: "en",
            sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
            channels: DEFAULT_CHANNELS,
            frame_samples: FRAME_SAMPLES,
            queues: QueueCaps::v0(),
        }
    }
}

impl Default for BuiltinDefaults {
    fn default() -> Self {
        Self::v0()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v0_defaults_match_launch_yaml() {
        let d = BuiltinDefaults::v0();
        assert_eq!(d.name, "demo-agent");
        assert_eq!(d.vad.as_str(), "silero");
        assert_eq!(d.stt.as_str(), "local");
        assert_eq!(d.stt_model.as_str(), "whisper-small");
        assert_eq!(d.llm.as_str(), "local");
        assert_eq!(d.llm_model, "lfm2.5-2.6b");
        assert!(!d.llm_thinking);
        assert_eq!(d.tts.as_str(), "local");
        assert_eq!(d.tts_model.as_str(), "pocket-tts");
        assert_eq!(d.language, "en");
        assert_eq!(d.sample_rate_hz, 16_000);
        assert_eq!(d.channels, 1);
        assert_eq!(d.queues, QueueCaps::v0());
    }

    #[test]
    fn one_provider_per_layer() {
        assert_eq!(VadProvider::Silero.as_str(), "silero");
        assert_eq!(SttProvider::Local.as_str(), "local");
        assert_eq!(LlmProvider::Local.as_str(), "local");
        assert_eq!(LlmProvider::Online.as_str(), "online");
        assert_eq!(TtsProvider::Local.as_str(), "local");
        assert_eq!(TtsProvider::Online.as_str(), "online");
        // The zero-config stack stays on-device; online is yaml opt-in only.
        assert_eq!(BuiltinDefaults::v0().llm, LlmProvider::Local);
        assert_eq!(BuiltinDefaults::v0().tts, TtsProvider::Local);
    }

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
        assert_eq!(
            BuiltinDefaults::v0().tts_model,
            TtsModel::PocketTts,
            "`pocket-tts` is the launch default"
        );
    }

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
    }
}
