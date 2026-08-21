//! Compiled-in v0 model list. The binary is the manifest; there is no extra file.

/// Pipeline layer an asset belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLayer {
    /// Silero VAD.
    Vad,
    /// whisper.cpp STT.
    Stt,
    /// llama.cpp GGUF.
    Llm,
    /// Kokoro ONNX (weights or voice).
    Tts,
}

impl ModelLayer {
    /// Config / log name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vad => "vad",
            Self::Stt => "stt",
            Self::Llm => "llm",
            Self::Tts => "tts",
        }
    }
}

/// One downloadable weight file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAsset {
    /// Stable id used in logs and cache lookups (`silero`, `whisper-small`, …).
    pub id: String,
    /// Pipeline layer.
    pub layer: ModelLayer,
    /// File name inside the cache directory.
    pub file_name: String,
    /// HTTPS URL. Fetched only when the cache miss or checksum fails.
    pub url: String,
    /// Lower-hex SHA-256 of the bytes.
    pub sha256: String,
    /// Exact size in bytes. Used to detect truncated downloads.
    pub size_bytes: u64,
}

/// Versioned set of assets. Bump `version` when the file set or hashes change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Cache subdirectory (`models/v{version}`).
    pub version: u32,
    /// Every weight the v0 binary may need.
    pub assets: Vec<ModelAsset>,
}

impl Manifest {
    /// Launch stack: Silero, whisper.cpp `small`, Llama 3.2 1B (default),
    /// Qwen3.5 0.8B and 2B (yaml), Kokoro + default English voice.
    pub fn v0() -> Self {
        Self {
            version: 1,
            assets: vec![
                asset(
                    "silero",
                    ModelLayer::Vad,
                    "silero_vad.onnx",
                    "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx",
                    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
                    2_327_524,
                ),
                asset(
                    "whisper-small",
                    ModelLayer::Stt,
                    "ggml-small.bin",
                    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
                    "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
                    487_601_967,
                ),
                asset(
                    "qwen3.5-0.8b",
                    ModelLayer::Llm,
                    "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
                    "https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/resolve/main/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf",
                    "fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526",
                    579_615_840,
                ),
                asset(
                    "qwen3.5-2b",
                    ModelLayer::Llm,
                    "Qwen_Qwen3.5-2B-Q4_K_M.gguf",
                    "https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf",
                    "57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8",
                    1_396_198_496,
                ),
                asset(
                    "llama-3.2-1b",
                    ModelLayer::Llm,
                    "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                    "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83",
                    807_694_464,
                ),
                asset(
                    "kokoro",
                    ModelLayer::Tts,
                    "kokoro-v1.0.onnx",
                    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx",
                    "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb",
                    325_532_232,
                ),
                asset(
                    "kokoro-voice",
                    ModelLayer::Tts,
                    "af_heart.bin",
                    "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af_heart.bin",
                    "d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b",
                    522_240,
                ),
            ],
        }
    }

    /// Look up an asset by id.
    pub fn asset(&self, id: &str) -> Option<&ModelAsset> {
        self.assets.iter().find(|a| a.id == id)
    }
}

fn asset(
    id: &str,
    layer: ModelLayer,
    file_name: &str,
    url: &str,
    sha256: &str,
    size_bytes: u64,
) -> ModelAsset {
    ModelAsset {
        id: id.to_string(),
        layer,
        file_name: file_name.to_string(),
        url: url.to_string(),
        sha256: sha256.to_string(),
        size_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::{BuiltinDefaults, SttModel};

    #[test]
    fn v0_lists_every_launch_layer() {
        let m = Manifest::v0();
        assert_eq!(m.version, 1);
        assert_eq!(m.assets.len(), 7);
        assert_eq!(m.asset("silero").unwrap().layer, ModelLayer::Vad);
        assert_eq!(m.asset("whisper-small").unwrap().layer, ModelLayer::Stt);
        assert_eq!(
            m.asset("whisper-small").unwrap().file_name,
            "ggml-small.bin"
        );
        assert_eq!(
            m.asset(BuiltinDefaults::v0().llm_model).unwrap().layer,
            ModelLayer::Llm
        );
        assert_eq!(m.asset("qwen3.5-2b").unwrap().layer, ModelLayer::Llm);
        assert_eq!(m.asset("llama-3.2-1b").unwrap().layer, ModelLayer::Llm);
        assert_eq!(m.asset("kokoro").unwrap().layer, ModelLayer::Tts);
        assert_eq!(m.asset("kokoro-voice").unwrap().layer, ModelLayer::Tts);
        assert!(m.asset("missing").is_none());
        assert_eq!(SttModel::Small.as_str(), "small");
        for asset in &m.assets {
            assert_eq!(asset.sha256.len(), 64);
            assert!(asset.size_bytes > 0);
            assert!(asset.url.starts_with("https://"));
        }
        assert_eq!(ModelLayer::Vad.as_str(), "vad");
        assert_eq!(ModelLayer::Stt.as_str(), "stt");
        assert_eq!(ModelLayer::Llm.as_str(), "llm");
        assert_eq!(ModelLayer::Tts.as_str(), "tts");
    }

    #[test]
    fn v0_weights_are_first_run_https_cache_not_packed() {
        let m = Manifest::v0();
        let mut total = 0u64;
        for asset in &m.assets {
            assert!(
                asset.url.starts_with("https://"),
                "{} must be a first-run HTTPS fetch, not an embedded blob",
                asset.id
            );
            total += asset.size_bytes;
        }
        // Whisper + Llama + Kokoro alone exceed 1.5 GiB; packing them into
        // the executable is not the v0 installable-artifact policy.
        assert!(total > 1_500_000_000);
        assert!(m.asset("silero").unwrap().size_bytes < 8_000_000);
    }
}
