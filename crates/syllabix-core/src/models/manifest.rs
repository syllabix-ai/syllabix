//! Compiled-in v0 model list. The binary is the manifest; there is no extra file.

use crate::defaults::BuiltinDefaults;

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
    /// Launch stack from `V0_LAUNCH.md`: Silero, whisper.cpp `small`,
    /// llama-3.2-1b GGUF, Kokoro + default English voice.
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
                    BuiltinDefaults::v0().llm_model,
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
    use crate::defaults::SttModel;

    #[test]
    fn v0_lists_every_launch_layer() {
        let m = Manifest::v0();
        assert_eq!(m.version, 1);
        assert_eq!(m.assets.len(), 5);
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
}
