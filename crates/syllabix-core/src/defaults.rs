//! Built-in providers, models, and queue sizes used when no configuration file exists.

use crate::types::{DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ, FRAME_SAMPLES};

/// Fixed queue bounds. Control stages block on send instead of growing;
/// the streaming-STT partial channel drops instead of blocking VAD.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueCaps {
    /// Mic frames waiting for VAD.
    pub frames: usize,
    /// Turn-control messages (`Start` / `Finalize`) waiting for STT. VAD is
    /// the single endpoint authority, so this path stays blocking and ordered.
    pub utterances: usize,
    /// Provisional per-frame audio waiting for streaming-STT partial decode.
    /// Full when a partial engine decodes slower than real time; VAD drops
    /// instead of stalling, and the canonical `Finalize` utterance still
    /// carries every frame.
    pub stt_partials: usize,
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
            stt_partials: 16,
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

/// Speech-to-text execution models. `Local` runs weights in-process;
/// `Online` points at an OpenAI-compatible transcriptions endpoint
/// (`pipeline.stt.base_url`). Online STT posts finished utterances and does
/// not fetch STT weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttProvider {
    /// In-process STT. The selected model chooses Whisper, Moonshine, or Qwen-ASR.
    Local,
    /// OpenAI-compatible remote transcriptions endpoint.
    Online,
}

impl SttProvider {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Online => "online",
        }
    }
}

pub use crate::stt::SttModel;

/// LLM execution models. `Local` runs weights in-process (llama.cpp GGUF,
/// first-run cache). `OpenAiCompatible` streams from any OpenAI-compatible
/// `chat/completions` endpoint (`pipeline.llm.base_url`, BYO key). Any other
/// yaml value is a Goose provider name when `agent_harness.agent: goose`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlmProvider {
    /// In-process GGUF engine. No network.
    Local,
    /// OpenAI-compatible remote endpoint (builtin LLM or Goose `openai`).
    OpenAiCompatible,
    /// Goose-native provider id (`anthropic`, `ollama`, …).
    External(String),
}

impl LlmProvider {
    /// Config / log name.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Local => "local",
            Self::OpenAiCompatible => "openai-compatible",
            Self::External(name) => name.as_str(),
        }
    }

    /// Non-Goose reply backends (`agent_harness.agent: syllabix` or `none`).
    pub fn builtin_supported(&self) -> bool {
        matches!(self, Self::Local | Self::OpenAiCompatible)
    }

    pub fn is_openai_compatible(&self) -> bool {
        matches!(self, Self::OpenAiCompatible)
    }
}

/// `agent_harness.agent` yaml value. Default [`Self::Syllabix`].
/// `Goose` uses `pipeline.llm` to write Goose's provider/model config.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentBackend {
    /// Plain voice chat with no sandboxed tools.
    None,
    /// Built-in Syllabix harness with sandboxed shell tools.
    Syllabix,
    /// `goose`. Syllabix writes Goose's provider and model from `pipeline.llm`.
    Goose,
}

impl AgentBackend {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Syllabix => "syllabix",
            Self::Goose => "goose",
        }
    }

    /// Whether the built-in tool loop and `pipeline.llm.developer_permissions` apply.
    pub fn syllabix_tools(self) -> bool {
        matches!(self, Self::Syllabix)
    }

    /// Whether optional `skills:` roots are allowed.
    pub fn allows_skills(self) -> bool {
        matches!(self, Self::Syllabix | Self::Goose)
    }
}

/// TTS execution models. `Local` runs weights in-process — Kokoro ONNX,
/// Pocket TTS, or Qwen3-TTS through the shared llama.cpp/ggml path;
/// `Online` points at an OpenAI-compatible speech endpoint
/// (`pipeline.tts.base_url`). Yaml accepts the same `local` / `online`
/// words as [`LlmProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsProvider {
    /// In-process weights. No network.
    Local,
    /// OpenAI-compatible remote endpoint. No local TTS weights.
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

/// Qwen ggml compute placement for TTS and ASR. `Auto` probes Metal on Apple
/// Silicon (or Vulkan on a Vulkan-enabled Linux build), then reloads on CPU if
/// the preferred GPU path cannot run. TTS probes after STT/LLM are resident;
/// ASR probes at STT load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsCompute {
    Auto,
    Cpu,
    Metal,
    Vulkan,
}

impl TtsCompute {
    pub const ALL: [Self; 4] = [Self::Auto, Self::Cpu, Self::Metal, Self::Vulkan];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.as_str() == value)
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
            llm_model: "lfm3-1.7b",
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
        assert_eq!(d.llm_model, "lfm3-1.7b");
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
        assert_eq!(SttProvider::Online.as_str(), "online");
        assert_eq!(LlmProvider::Local.as_str(), "local");
        assert_eq!(LlmProvider::OpenAiCompatible.as_str(), "openai-compatible");
        assert_eq!(
            LlmProvider::External("anthropic".into()).as_str(),
            "anthropic"
        );
        assert_eq!(TtsProvider::Local.as_str(), "local");
        assert_eq!(TtsProvider::Online.as_str(), "online");
        // The zero-config stack stays on-device; online is yaml opt-in only.
        assert_eq!(BuiltinDefaults::v0().stt, SttProvider::Local);
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
}
