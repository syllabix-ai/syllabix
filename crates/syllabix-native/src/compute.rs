//! Qwen ggml compute placement for TTS and ASR.

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
