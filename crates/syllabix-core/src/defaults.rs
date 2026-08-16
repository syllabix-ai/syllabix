//! Built-in on-device stack. One provider per layer; no cloud variants in v0.

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

/// whisper.cpp model id for v0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttModel {
    /// `small` English / multilingual weights.
    Small,
}

impl SttModel {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
        }
    }
}

/// The only v0 LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    /// llama.cpp GGUF, in-process.
    LlamaCpp,
}

impl LlmProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LlamaCpp => "llama.cpp",
        }
    }
}

/// llama.cpp GGUF id for v0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmModel {
    /// Llama-3.2-1B-Instruct Q4_K_M.
    Llama32_1b,
}

impl LlmModel {
    /// Config / log / manifest id.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Llama32_1b => "llama-3.2-1b",
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
    /// Small instruct GGUF id (bundled or first-run cache later).
    pub llm_model: &'static str,
    /// TTS provider.
    pub tts: TtsProvider,
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
            llm: LlmProvider::LlamaCpp,
            llm_model: "llama-3.2-1b",
            tts: TtsProvider::Kokoro,
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
        assert_eq!(d.llm.as_str(), "llama.cpp");
        assert_eq!(d.llm_model, "llama-3.2-1b");
        assert_eq!(d.tts.as_str(), "kokoro");
        assert_eq!(d.sample_rate_hz, 16_000);
        assert_eq!(d.channels, 1);
        assert_eq!(d.queues, QueueCaps::v0());
    }

    #[test]
    fn one_provider_per_layer() {
        assert_eq!(VadProvider::Silero.as_str(), "silero");
        assert_eq!(SttProvider::WhisperCpp.as_str(), "whisper.cpp");
        assert_eq!(LlmProvider::LlamaCpp.as_str(), "llama.cpp");
        assert_eq!(LlmModel::Llama32_1b.as_str(), "llama-3.2-1b");
        assert_eq!(TtsProvider::Kokoro.as_str(), "kokoro");
    }
}
