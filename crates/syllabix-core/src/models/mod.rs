//! Versioned model manifest, checksummed downloads, and on-disk cache.

#[cfg(all(not(coverage), feature = "ggml", feature = "cloud"))]
mod goose;

pub use syllabix_native::models::{cache_root, ModelCache};
pub use syllabix_native::models::{format_progress_line, NoProgress, Progress, StderrProgress};
pub use syllabix_native::models::{BlockedFetcher, Fetcher, HttpFetcher};
pub use syllabix_native::models::{Manifest, ModelAsset, ModelLayer};

#[cfg(all(not(coverage), feature = "ggml", feature = "cloud"))]
pub(crate) use goose::resolve_goose_binary;

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
        // with its own projector, plus five LFM assets.
        assert_eq!(m.assets.len(), 38);
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
            ModelLayer::Tts,
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
            m.asset("lfm3-1.7b").unwrap().sha256,
            "8f6b24182258205e9e01e21f1f24a1f208c1e8115578825d202ce39976af20b6"
        );
        assert_eq!(m.asset("lfm3-1.7b").unwrap().size_bytes, 1_846_238_496);
        assert_eq!(
            m.asset("lfm3-1.7b-bf16").unwrap().sha256,
            "39158a79f56edb85e0c4252ce6447b8d7883ef82ffedd02f0cda7c84da5e9f68"
        );
        assert_eq!(m.asset("lfm3-1.7b-bf16").unwrap().size_bytes, 3_466_626_336);
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
        assert_eq!(ModelLayer::Agent.as_str(), "agent");
    }
}
