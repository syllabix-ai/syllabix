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
/// (Row 32 deliberately kept `v1` across a purely additive asset: existing
/// file names and hashes were untouched, and a bump would force every user to
/// re-fetch all weights into a fresh cache directory.)
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
                    "whisper-medium",
                    ModelLayer::Stt,
                    "ggml-medium.bin",
                    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
                    "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
                    1_533_763_059,
                ),
                asset(
                    "whisper-large-v3-turbo",
                    ModelLayer::Stt,
                    "ggml-large-v3-turbo.bin",
                    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
                    "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
                    1_624_555_275,
                ),
                asset(
                    "whisper-medium-q5_0",
                    ModelLayer::Stt,
                    "ggml-medium-q5_0.bin",
                    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
                    "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f",
                    539_212_467,
                ),
                asset(
                    "whisper-large-v3-turbo-q5_0",
                    ModelLayer::Stt,
                    "ggml-large-v3-turbo-q5_0.bin",
                    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
                    "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
                    574_041_195,
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
                // P1 Pocket TTS feasibility contract. These are deliberately
                // not selectable from yaml until P2: the native suite alone
                // resolves them. The fixed precomputed voice means P1 never
                // accepts microphone/user audio or performs voice registration.
                asset(
                    "pocket-tts-bundle",
                    ModelLayer::Tts,
                    "pocket-tts-english-bundle.json",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/bundle.json",
                    "bab643150f437f37df080a710520ff39ed9ebd9a339f8ebdc739f7eddfc28b3f",
                    24_381,
                ),
                asset(
                    "pocket-tts-bos",
                    ModelLayer::Tts,
                    "pocket-tts-english-bos_before_voice.npy",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/bos_before_voice.npy",
                    "f46edf4f7007b7ba4ea58831f49d003e59e167b4641c44bb3addfe9231a780b1",
                    4_224,
                ),
                asset(
                    "pocket-tts-tokenizer",
                    ModelLayer::Tts,
                    "pocket-tts-english-tokenizer.model",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/tokenizer.model",
                    "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6",
                    59_339,
                ),
                asset(
                    "pocket-tts-text-conditioner",
                    ModelLayer::Tts,
                    "pocket-tts-english-text_conditioner.onnx",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/text_conditioner.onnx",
                    "4ecee995fb69f85c7a7493d11f7b5ee15d9950facc7ab3f5c9c49ef1e03847bb",
                    16_388_344,
                ),
                asset(
                    "pocket-tts-flow-main",
                    ModelLayer::Tts,
                    "pocket-tts-english-flow_lm_main_int8.onnx",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/flow_lm_main_int8.onnx",
                    "f9bd8106b79a0192c1c43399ab938fb24900a95c1c599870d75a884e99000116",
                    76_341_079,
                ),
                asset(
                    "pocket-tts-flow",
                    ModelLayer::Tts,
                    "pocket-tts-english-flow_lm_flow_int8.onnx",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/flow_lm_flow_int8.onnx",
                    "3dd781ee5abee9e195320bf0106bebd6372a852b3b36352524ee78b40554635d",
                    9_962_530,
                ),
                asset(
                    "pocket-tts-mimi-decoder",
                    ModelLayer::Tts,
                    "pocket-tts-english-mimi_decoder.onnx",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/mimi_decoder.onnx",
                    "86f038caa02a96a0ff9c25526a0ff43a4906c418197ed72d3e30f720ac7ce802",
                    41_471_926,
                ),
                asset(
                    "pocket-tts-voice-alba",
                    ModelLayer::Tts,
                    "pocket-tts-english-alba.safetensors",
                    "https://huggingface.co/OpenVoiceOS/phoonnx-pocket-tts/resolve/81b240f24f896f4cc480dc90858d1f72c9b7d64a/english_2026-04/voices/alba.safetensors",
                    "69c32db63ca56843d994f81f343f62e0bf2d73f7e4c9bc73e44bb1110b1d8845",
                    6_194_424,
                ),
                asset(
                    "qwen3-tts",
                    ModelLayer::Tts,
                    "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
                    "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
                    "8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129",
                    1_035_965_280,
                ),
                asset(
                    "qwen3-tts-mmproj",
                    ModelLayer::Tts,
                    "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
                    "6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2",
                    446_422_912,
                ),
                // Row 32: the smaller Qwen3-TTS backbone plus its own
                // speech-tokenizer projector. The upstream tokenizer
                // *encoder* is byte-identical across both sizes, but the
                // mmproj also carries the projector into the LM embedding
                // space (2048-d for 1.7B, 1024-d for 0.6B), so each
                // backbone pairs with its own mmproj file.
                asset(
                    "qwen3-tts-06b",
                    ModelLayer::Tts,
                    "Qwen3-TTS-12Hz-0.6B-Base.Q4_K_M.gguf",
                    "https://huggingface.co/mradermacher/Qwen3-TTS-12Hz-0.6B-Base-GGUF/resolve/main/Qwen3-TTS-12Hz-0.6B-Base.Q4_K_M.gguf",
                    "2dec66bcf1595f48a6dfb64c14e7e8437315e1606eba03094ed7372e60a4b9af",
                    361_023_168,
                ),
                asset(
                    "qwen3-tts-06b-mmproj",
                    ModelLayer::Tts,
                    "Qwen3-TTS-12Hz-0.6B-Base.mmproj-Q8_0.gguf",
                    "https://huggingface.co/mradermacher/Qwen3-TTS-12Hz-0.6B-Base-GGUF/resolve/main/Qwen3-TTS-12Hz-0.6B-Base.mmproj-Q8_0.gguf",
                    "202c1fbf3a0f00b7586ca45b8328d696cc6cd980c3798979eaa4fefe8efd8320",
                    401_129_632,
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
        // Silero + five whisper ids + three GGUFs + Kokoro weights + voice
        // + eight internal-only P1 Pocket TTS assets + two Qwen3-TTS
        // backbones, each with its own mmproj.
        assert_eq!(m.assets.len(), 23);
        for model in SttModel::ALL {
            let asset = m
                .asset(model.asset_id())
                .unwrap_or_else(|| panic!("manifest must contain {}", model.asset_id()));
            assert_eq!(asset.layer, ModelLayer::Stt, "{}", asset.id);
            assert!(asset.file_name.starts_with("ggml-"), "{}", asset.id);
        }
        assert_eq!(
            m.asset(SttModel::Small.asset_id()).unwrap().file_name,
            "ggml-small.bin"
        );
        assert_eq!(
            m.asset("whisper-large-v3-turbo-q5_0").unwrap().size_bytes,
            574_041_195
        );
        assert_eq!(
            m.asset(BuiltinDefaults::v0().llm_model).unwrap().layer,
            ModelLayer::Llm
        );
        assert_eq!(m.asset("qwen3.5-2b").unwrap().layer, ModelLayer::Llm);
        assert_eq!(m.asset("llama-3.2-1b").unwrap().layer, ModelLayer::Llm);
        assert_eq!(m.asset("kokoro").unwrap().layer, ModelLayer::Tts);
        assert_eq!(m.asset("kokoro-voice").unwrap().layer, ModelLayer::Tts);
        assert_eq!(m.asset("pocket-tts-bundle").unwrap().layer, ModelLayer::Tts);
        assert_eq!(
            m.asset("pocket-tts-voice-alba").unwrap().size_bytes,
            6_194_424
        );
        assert_eq!(m.asset("qwen3-tts").unwrap().layer, ModelLayer::Tts);
        assert_eq!(m.asset("qwen3-tts-mmproj").unwrap().layer, ModelLayer::Tts);
        assert_eq!(m.asset("qwen3-tts-06b").unwrap().layer, ModelLayer::Tts);
        assert_eq!(
            m.asset("qwen3-tts-06b-mmproj").unwrap().layer,
            ModelLayer::Tts
        );
        assert_eq!(
            m.asset("qwen3-tts-06b").unwrap().sha256,
            "2dec66bcf1595f48a6dfb64c14e7e8437315e1606eba03094ed7372e60a4b9af"
        );
        assert_eq!(m.asset("qwen3-tts-06b").unwrap().size_bytes, 361_023_168);
        // The 0.6B backbone is ~3x smaller than the 1.7B one.
        assert!(
            m.asset("qwen3-tts-06b").unwrap().size_bytes * 2
                < m.asset("qwen3-tts").unwrap().size_bytes
        );
        assert_eq!(
            m.asset("qwen3-tts").unwrap().sha256,
            "8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129"
        );
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
