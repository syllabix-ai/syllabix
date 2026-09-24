//! Compiled-in model asset metadata used for verified cache downloads.

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

/// Versioned set of assets. Bump `version` when existing file names or hashes change.
/// Purely additive assets keep the current version because existing downloads
/// remain valid and every new file has a distinct name and checksum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Cache subdirectory (`models/v{version}`).
    pub version: u32,
    /// Every model asset the executable can load.
    pub assets: Vec<ModelAsset>,
}

impl Manifest {
    /// Assets for the default Silero, whisper.cpp `small`, LFM2.5-2.6B, and
    /// Pocket TTS stack, plus models selectable through YAML.
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
                // Moonshine streaming-small INT8 ONNX (Mazino0 export of
                // moonshine-ai/moonshine-streaming-small, MIT). Three graphs
                // plus the BPE tokenizer; fetched only when yaml selects the
                // moonshine STT model.
                asset(
                    "moonshine-encoder",
                    ModelLayer::Stt,
                    "moonshine-streaming-small-encoder-int8.onnx",
                    "https://huggingface.co/Mazino0/moonshine-streaming-small-onnx/resolve/d47300364351d5dd673073c16f0159b5ee46f512/encoder_model_int8.onnx",
                    "9bb6562667da35c8b6994bd76139528610738a33c1c3fa234024c75a6affa509",
                    75_143_940,
                ),
                asset(
                    "moonshine-decoder",
                    ModelLayer::Stt,
                    "moonshine-streaming-small-decoder-int8.onnx",
                    "https://huggingface.co/Mazino0/moonshine-streaming-small-onnx/resolve/d47300364351d5dd673073c16f0159b5ee46f512/decoder_model_int8.onnx",
                    "8c1a86e1b3059950d8285a47f3dae1fb6166f0337046e115965498e7957be158",
                    149_164_324,
                ),
                asset(
                    "moonshine-decoder-past",
                    ModelLayer::Stt,
                    "moonshine-streaming-small-decoder-with-past-int8.onnx",
                    "https://huggingface.co/Mazino0/moonshine-streaming-small-onnx/resolve/d47300364351d5dd673073c16f0159b5ee46f512/decoder_with_past_model_int8.onnx",
                    "e9bfbc4f2b34ea82ff5b562cc20d3eafcf87a8a25ea9bcaabd8513078dbc0565",
                    133_455_145,
                ),
                asset(
                    "moonshine-tokenizer",
                    ModelLayer::Stt,
                    "moonshine-streaming-small-tokenizer.json",
                    "https://huggingface.co/Mazino0/moonshine-streaming-small-onnx/resolve/d47300364351d5dd673073c16f0159b5ee46f512/tokenizer.json",
                    "7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f",
                    3_761_754,
                ),
                // Moonshine streaming-medium INT8 ONNX (Mazino0 export of
                // moonshine-ai/moonshine-streaming-medium, MIT). Same four-file
                // shape as small; fetched only when yaml selects medium.
                asset(
                    "moonshine-medium-encoder",
                    ModelLayer::Stt,
                    "moonshine-streaming-medium-encoder-int8.onnx",
                    "https://huggingface.co/Mazino0/moonshine-streaming-medium-onnx/resolve/8da8adfd536fe69a255dcd20831ec2361bd63280/encoder_model_int8.onnx",
                    "4f6c491eb4018a06f2e9ecf5b6bab5c6fa4e679c9ed5dde02a0a27969649be90",
                    142_060_073,
                ),
                asset(
                    "moonshine-medium-decoder",
                    ModelLayer::Stt,
                    "moonshine-streaming-medium-decoder-int8.onnx",
                    "https://huggingface.co/Mazino0/moonshine-streaming-medium-onnx/resolve/8da8adfd536fe69a255dcd20831ec2361bd63280/decoder_model_int8.onnx",
                    "38dfe5829fcb814e33634c00baedceaa877acaac7b731203e88eb956d4419875",
                    236_021_354,
                ),
                asset(
                    "moonshine-medium-decoder-past",
                    ModelLayer::Stt,
                    "moonshine-streaming-medium-decoder-with-past-int8.onnx",
                    "https://huggingface.co/Mazino0/moonshine-streaming-medium-onnx/resolve/8da8adfd536fe69a255dcd20831ec2361bd63280/decoder_with_past_model_int8.onnx",
                    "36d7ea3cf4feb6e37fe784ba3ac7cee0bb5f4d757ab05433e2550b8eae035a7e",
                    211_467_644,
                ),
                asset(
                    "moonshine-medium-tokenizer",
                    ModelLayer::Stt,
                    "moonshine-streaming-medium-tokenizer.json",
                    "https://huggingface.co/Mazino0/moonshine-streaming-medium-onnx/resolve/8da8adfd536fe69a255dcd20831ec2361bd63280/tokenizer.json",
                    "7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f",
                    3_761_754,
                ),
                // Qwen3-ASR 0.6B: llama.cpp mtmd decoder + qwen3a encoder.
                // Fetched only when yaml selects qwen3-asr-0.6.
                asset(
                    "qwen3-asr-0.6",
                    ModelLayer::Stt,
                    "Qwen3-ASR-0.6B-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B-Q8_0.gguf",
                    "bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971",
                    804_749_248,
                ),
                asset(
                    "qwen3-asr-0.6-mmproj",
                    ModelLayer::Stt,
                    "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
                    "https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/resolve/main/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
                    "41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d",
                    214_392_480,
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
                // Pocket TTS uses a fixed precomputed voice and never accepts
                // microphone/user audio or performs voice registration.
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
                // The smaller Qwen3-TTS backbone uses its own
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
                // LiquidAI LFM2.5-2.6B QAD Q4_0 GGUF. SHA-256 is the HF LFS
                // object id, size is the LFS/XET byte count, and the model uses
                // the `lfm1.0` license.
                asset(
                    "lfm2.5-2.6b",
                    ModelLayer::Llm,
                    "LFM2.5-2.6B-QAD-Q4_0.gguf",
                    "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main/LFM2.5-2.6B-QAD-Q4_0.gguf",
                    "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03",
                    1_593_894_944,
                ),
                // LiquidAI LFM2.5-350M / 230M QAD Q4_0 GGUFs. Same LFM Open
                // License v1.0 family as 2.6B; yaml opt-in only.
                asset(
                    "lfm2.5-350m",
                    ModelLayer::Llm,
                    "LFM2.5-350M-QAD-Q4_0.gguf",
                    "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-QAD-Q4_0.gguf",
                    "3d10b6ab8fc91a919534b9558e266255aca0bbc7f6d015963599aa9e74e05b1d",
                    219_312_832,
                ),
                asset(
                    "lfm2.5-230m",
                    ModelLayer::Llm,
                    "LFM2.5-230M-QAD-Q4_0.gguf",
                    "https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF/resolve/main/LFM2.5-230M-QAD-Q4_0.gguf",
                    "e75f83268de11b2a1bcfab5f3b5c5c0c97569ddbbc0990aad88437e45b8ba292",
                    149_081_056,
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
        // Silero + five whisper ids + eight moonshine assets (small+medium) +
        // Qwen3-ASR 0.6B decoder/mmproj + three GGUFs + Kokoro weights + voice +
        // eight internal-only Pocket TTS assets + two Qwen3-TTS backbones, each
        // with its own projector, plus three LFM assets.
        assert_eq!(m.assets.len(), 36);
        for model in SttModel::ALL {
            let asset = m
                .asset(model.asset_id())
                .unwrap_or_else(|| panic!("manifest must contain {}", model.asset_id()));
            assert_eq!(asset.layer, ModelLayer::Stt, "{}", asset.id);
            if model == SttModel::MoonshineStreamingSmall {
                assert_eq!(
                    asset.file_name,
                    "moonshine-streaming-small-encoder-int8.onnx"
                );
                continue;
            }
            if model == SttModel::MoonshineStreamingMedium {
                assert_eq!(
                    asset.file_name,
                    "moonshine-streaming-medium-encoder-int8.onnx"
                );
                continue;
            }
            if model == SttModel::QwenAsr06 {
                assert_eq!(asset.file_name, "Qwen3-ASR-0.6B-Q8_0.gguf");
                continue;
            }
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
        assert_eq!(m.asset("moonshine-encoder").unwrap().layer, ModelLayer::Stt);
        assert_eq!(m.asset("moonshine-encoder").unwrap().size_bytes, 75_143_940);
        assert_eq!(
            m.asset("moonshine-decoder").unwrap().size_bytes,
            149_164_324
        );
        assert_eq!(
            m.asset("moonshine-decoder-past").unwrap().size_bytes,
            133_455_145
        );
        assert_eq!(
            m.asset("moonshine-tokenizer").unwrap().size_bytes,
            3_761_754
        );
        assert_eq!(
            m.asset("moonshine-medium-encoder").unwrap().layer,
            ModelLayer::Stt
        );
        assert_eq!(
            m.asset("moonshine-medium-encoder").unwrap().size_bytes,
            142_060_073
        );
        assert_eq!(
            m.asset("moonshine-medium-decoder").unwrap().size_bytes,
            236_021_354
        );
        assert_eq!(
            m.asset("moonshine-medium-decoder-past").unwrap().size_bytes,
            211_467_644
        );
        assert_eq!(
            m.asset("moonshine-medium-tokenizer").unwrap().size_bytes,
            3_761_754
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
        assert_eq!(m.asset("lfm2.5-2.6b").unwrap().layer, ModelLayer::Llm);
        assert_eq!(
            m.asset("lfm2.5-2.6b").unwrap().sha256,
            "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03"
        );
        assert_eq!(m.asset("lfm2.5-2.6b").unwrap().size_bytes, 1_593_894_944);
        assert_eq!(m.asset("lfm2.5-350m").unwrap().layer, ModelLayer::Llm);
        assert_eq!(
            m.asset("lfm2.5-350m").unwrap().sha256,
            "3d10b6ab8fc91a919534b9558e266255aca0bbc7f6d015963599aa9e74e05b1d"
        );
        assert_eq!(m.asset("lfm2.5-350m").unwrap().size_bytes, 219_312_832);
        assert_eq!(m.asset("lfm2.5-230m").unwrap().layer, ModelLayer::Llm);
        assert_eq!(
            m.asset("lfm2.5-230m").unwrap().sha256,
            "e75f83268de11b2a1bcfab5f3b5c5c0c97569ddbbc0990aad88437e45b8ba292"
        );
        assert_eq!(m.asset("lfm2.5-230m").unwrap().size_bytes, 149_081_056);
        assert!(m.asset("missing").is_none());
        assert_eq!(SttModel::Small.as_str(), "whisper-small");
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
        // the executable would make the distribution artifact excessively large.
        assert!(total > 1_500_000_000);
        assert!(m.asset("silero").unwrap().size_bytes < 8_000_000);
    }
}
