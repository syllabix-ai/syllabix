//! Built-in on-device stack. One provider per layer; the only cloud path is
//! the row-30 BYO-key LLM adapter (`pipeline.llm.provider: openai`).

use crate::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES};

/// Fixed v0 queue bounds. Stages block on send instead of growing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueCaps {
    /// Mic frames waiting for VAD.
    pub frames: usize,
    /// Completed utterances waiting for STT.
    pub utterances: usize,
    /// Transcripts waiting for the LLM.
    pub transcripts: usize,
    /// Streamed tokens waiting for TTS.
    pub tokens: usize,
    /// Synthesized chunks waiting for playback.
    pub audio: usize,
}

impl QueueCaps {
    /// Launch defaults: small enough to catch unbounded buffering in tests.
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

/// The only v0 VAD.
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

/// The only v0 STT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttProvider {
    /// whisper.cpp.
    WhisperCpp,
}

impl SttProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WhisperCpp => "whisper.cpp",
        }
    }
}

/// whisper.cpp model menu. One provider, several GGML sizes; `small` stays
/// the launch default. `-q5_0` ids are the published quantizations of their
/// fp16 siblings (`tiny` / `base` are deliberately not offered).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttModel {
    /// `small` multilingual weights (v0 launch default).
    Small,
    /// `medium` multilingual weights.
    Medium,
    /// `large-v3-turbo` weights.
    LargeV3Turbo,
    /// Published `medium` q5_0 quantization.
    MediumQ5_0,
    /// Published `large-v3-turbo` q5_0 quantization.
    LargeV3TurboQ5_0,
}

impl SttModel {
    /// Every yaml-selectable id, manifest order.
    pub const ALL: [SttModel; 5] = [
        SttModel::Small,
        SttModel::Medium,
        SttModel::LargeV3Turbo,
        SttModel::MediumQ5_0,
        SttModel::LargeV3TurboQ5_0,
    ];

    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::LargeV3Turbo => "large-v3-turbo",
            Self::MediumQ5_0 => "medium-q5_0",
            Self::LargeV3TurboQ5_0 => "large-v3-turbo-q5_0",
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
        }
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

/// The only v0 TTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsProvider {
    /// Kokoro ONNX.
    Kokoro,
}

impl TtsProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Kokoro => "kokoro",
        }
    }
}

/// Zero-config stack written by a later `init`, and used when yaml is missing.
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
    /// Qwen thinking. Off by default; yaml `thinking: true` enables it.
    pub llm_thinking: bool,
    /// TTS provider.
    pub tts: TtsProvider,
    /// STT language code (`en`). YAML may set this; v0 allows only `en`.
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
    /// Launch built-ins from `V0_LAUNCH.md`.
    pub const fn v0() -> Self {
        Self {
            name: "demo-agent",
            vad: VadProvider::Silero,
            stt: SttProvider::WhisperCpp,
            stt_model: SttModel::Small,
            llm: LlmProvider::Local,
            llm_model: "llama-3.2-1b",
            llm_thinking: false,
            tts: TtsProvider::Kokoro,
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
        assert_eq!(d.stt.as_str(), "whisper.cpp");
        assert_eq!(d.stt_model.as_str(), "small");
        assert_eq!(d.llm.as_str(), "local");
        assert_eq!(d.llm_model, "llama-3.2-1b");
        assert!(!d.llm_thinking);
        assert_eq!(d.tts.as_str(), "kokoro");
        assert_eq!(d.language, "en");
        assert_eq!(d.sample_rate_hz, 16_000);
        assert_eq!(d.channels, 1);
        assert_eq!(d.queues, QueueCaps::v0());
    }

    #[test]
    fn one_provider_per_layer() {
        assert_eq!(VadProvider::Silero.as_str(), "silero");
        assert_eq!(SttProvider::WhisperCpp.as_str(), "whisper.cpp");
        assert_eq!(LlmProvider::Local.as_str(), "local");
        assert_eq!(LlmProvider::Online.as_str(), "online");
        assert_eq!(TtsProvider::Kokoro.as_str(), "kokoro");
        // The zero-config stack stays on-device; online is yaml opt-in only.
        assert_eq!(BuiltinDefaults::v0().llm, LlmProvider::Local);
    }

    #[test]
    fn stt_menu_ids_round_trip() {
        let ids = [
            "small",
            "medium",
            "large-v3-turbo",
            "medium-q5_0",
            "large-v3-turbo-q5_0",
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
            "`small` stays the launch default"
        );
    }
}
