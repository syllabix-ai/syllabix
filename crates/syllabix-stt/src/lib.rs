//! Speech-to-text engines and the [`Stt`] trait they share.
//!
//! Turn engines on with Cargo features. Defaults are `moonshine` and `remote`
//! (ONNX plus HTTP). `whisper`, `qwen3-asr`, `parakeet`, and `hima` are opt-in. With
//! `default-features = false` and only `remote`, this crate builds without
//! CMake, a C++ compiler, or ONNX Runtime.
//!
//! Default ONNX engines use the `ort` crate, which may download a prebuilt ONNX
//! Runtime at build time; some targets have no binary. For HTTP only, use
//! `default-features = false` with the `remote` feature.
//!
//! | Feature | Engine | Needs C++? | Model ids |
//! | --- | --- | --- | --- |
//! | `moonshine` | Moonshine streaming ONNX (`moonshine`) | No (ONNX Runtime) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
//! | `remote` | OpenAI-compatible `/audio/transcriptions` (`remote`) | No (HTTP only) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
//! | `whisper` | whisper.cpp ([`WhisperStt`]) | Yes (CMake) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
//! | `qwen3-asr` | Qwen3-ASR 0.6B ([`qwen_asr`]) | Yes (CMake) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
//! | `parakeet` | Fluid Speech Pro GGUF ([`fluid_gguf`]) | Yes (CMake) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
//! | `hima` | Hima-ASR ([`HimaStt`]) | No (ONNX Runtime) | [engines — STT](https://github.com/syllabix-ai/syllabix/blob/main/docs/engines.md#stt) |
//!
//! C++ engines need the host toolchain in
//! [embed — Host compile](https://github.com/syllabix-ai/syllabix/blob/main/docs/embed.md#host-compile).
//! Without `whisper` or `qwen3-asr` their types still exist and return an
//! error when loaded. The remote engine reads its key from
//! `SYLLABIX_STT_API_KEY` only. Examples:
//! [`transcribe-wav-local`](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-stt/examples/transcribe-wav-local.rs)
//! (Moonshine) and
//! [`transcribe-wav`](https://github.com/syllabix-ai/syllabix/blob/main/crates/syllabix-stt/examples/transcribe-wav.rs)
//! (remote).

// The types in the `Stt` signatures, so callers need no other crate.
pub use syllabix_native::audio;
pub use syllabix_native::models::{
    cache_root, BlockedFetcher, Fetcher, HttpFetcher, Manifest, ModelAsset, ModelCache, NoProgress,
    Progress, StderrProgress,
};
pub use syllabix_native::types::{
    AudioFrame, Transcript, TurnId, Utterance, DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE_HZ,
    FRAME_SAMPLES,
};
pub use syllabix_native::{Cancel, Error, Result};

pub mod fluid_gguf;
#[cfg(feature = "hima")]
pub mod hima;
pub mod language;
#[cfg(feature = "moonshine")]
pub mod moonshine;
#[cfg(feature = "hima")]
pub mod npy;
pub mod qwen_asr;
#[cfg(feature = "remote")]
pub mod remote;
#[cfg(feature = "hima")]
pub mod stft;
mod whisper;

pub use whisper::*;

/// Speech-to-text for a completed VAD utterance.
pub trait Stt: Send {
    /// Config name (`whisper.cpp`).
    fn name(&self) -> &'static str;

    /// Transcribe one utterance. Must check `cancel` and return [`syllabix_native::Error::Cancelled`].
    fn transcribe(&mut self, utterance: &Utterance, cancel: &Cancel) -> Result<Transcript>;

    /// Whether this engine can produce live text while VAD owns an active
    /// turn. The default keeps the established utterance-final Whisper path.
    fn supports_partials(&self) -> bool {
        false
    }

    /// VAD opened `turn`. This is notification only: VAD remains the single
    /// owner of start/end timing and the shared `end_silence_ms` setting.
    fn start_turn(&mut self, _turn: TurnId, _cancel: &Cancel) -> Result<()> {
        Ok(())
    }

    /// One post-AEC frame from an active VAD turn. Returning text updates the
    /// provisional transcript; returning `None` means no visible change.
    fn push_frame(&mut self, _frame: &AudioFrame, _cancel: &Cancel) -> Result<Option<String>> {
        Ok(None)
    }

    /// Forget provisional state for a turn abandoned by cancellation.
    fn cancel_turn(&mut self, _turn: TurnId) {}
}

/// whisper.cpp model menu. One provider exposes several GGML sizes;
/// `whisper-small` is the default. `-q5_0` ids are the published quantizations of their
/// fp16 siblings (`tiny` / `base` are deliberately not offered).
/// `MoonshineStreamingSmall` / `MoonshineStreamingMedium` are Moonshine ONNX engine ids.
/// `QwenAsr06` is Qwen3-ASR 0.6B through llama.cpp mtmd.
/// `HimaAsr` is Hima-ASR int8 ONNX (Indic / code-switched, finalize only).
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
    /// Moonshine streaming-medium INT8 ONNX (English only).
    MoonshineStreamingMedium,
    /// Qwen3-ASR 0.6B via llama.cpp mtmd (finalize only).
    QwenAsr06,
    /// Hima-ASR int8 ONNX (Indic / code-switched, finalize only).
    HimaAsr,
    /// Fluid Speech Pro Q8_0 GGUF via parakeet.cpp (finalize only).
    FluidSpeechProGgufQ8,
}

impl SttModel {
    /// Every menu id, manifest order.
    pub const ALL: [SttModel; 10] = [
        SttModel::Small,
        SttModel::Medium,
        SttModel::LargeV3Turbo,
        SttModel::MediumQ5_0,
        SttModel::LargeV3TurboQ5_0,
        SttModel::MoonshineStreamingSmall,
        SttModel::MoonshineStreamingMedium,
        SttModel::QwenAsr06,
        SttModel::HimaAsr,
        SttModel::FluidSpeechProGgufQ8,
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
            Self::QwenAsr06 => "qwen3-asr-0.6",
            Self::HimaAsr => "hima-asr",
            Self::FluidSpeechProGgufQ8 => "fluid-speech-pro-gguf-q8_0",
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
            // Same ids as `moonshine::ENCODER_ASSET` / `MEDIUM_ENCODER_ASSET`
            // (checked in moonshine's tests); spelled out so this builds
            // without the `moonshine` feature.
            Self::MoonshineStreamingSmall => "moonshine-encoder",
            Self::MoonshineStreamingMedium => "moonshine-medium-encoder",
            Self::QwenAsr06 => qwen_asr::ASR_ASSET,
            // Same id as `hima::ENCODER_ASSET`; spelled out so this builds
            // without the `hima` feature.
            Self::HimaAsr => "hima-encoder",
            // Same id as `fluid_gguf::ASSET`; spelled out so this builds
            // without the `parakeet` feature.
            Self::FluidSpeechProGgufQ8 => "fluid-speech-pro-gguf-q8_0",
        }
    }

    /// True for either Moonshine streaming size (English-only ONNX).
    pub fn is_moonshine(self) -> bool {
        matches!(
            self,
            Self::MoonshineStreamingSmall | Self::MoonshineStreamingMedium
        )
    }

    /// True for the Qwen3-ASR 0.6B mtmd engine.
    pub fn is_qwen_asr(self) -> bool {
        matches!(self, Self::QwenAsr06)
    }

    /// True for the Hima-ASR int8 ONNX engine.
    pub fn is_hima(self) -> bool {
        matches!(self, Self::HimaAsr)
    }

    /// True for Fluid Speech Pro GGUF through parakeet.cpp.
    pub fn is_fluid_gguf(self) -> bool {
        matches!(self, Self::FluidSpeechProGgufQ8)
    }

    /// Parse a model menu id.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.as_str() == value)
    }
}

/// Build one [`Utterance`] from PCM in any WAV layout, resampling to 16 kHz mono.
pub fn pcm_to_utterance(wav: &audio::WavPcm, turn: TurnId) -> Result<Utterance> {
    Ok(Utterance {
        turn,
        frames: wav_to_frames(wav)?,
    })
}

/// Read a WAV file, convert to engine PCM, and transcribe it.
pub fn transcribe_wav(
    stt: &mut impl Stt,
    path: impl AsRef<std::path::Path>,
    cancel: &Cancel,
) -> Result<Transcript> {
    let wav = audio::read_wav(std::fs::File::open(path.as_ref())?)?;
    let utterance = pcm_to_utterance(&wav, TurnId(0))?;
    stt.transcribe(&utterance, cancel)
}

fn wav_to_frames(wav: &audio::WavPcm) -> Result<Vec<AudioFrame>> {
    use audio::{f32_to_i16, i16_to_f32, FrameSplitter, PcmConverter, PcmFormat};
    let mut converter = PcmConverter::new(wav.format, PcmFormat::v0())?;
    let mut mono = converter.push(&i16_to_f32(&wav.samples));
    mono.extend(converter.flush());
    let mut splitter = FrameSplitter::new();
    let mut frames = splitter.push(&f32_to_i16(&mono))?;
    frames.extend(splitter.flush()?);
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopStt;

    impl Stt for NoopStt {
        fn name(&self) -> &'static str {
            "noop"
        }

        fn transcribe(&mut self, _utterance: &Utterance, _cancel: &Cancel) -> Result<Transcript> {
            Ok(Transcript {
                turn: TurnId(0),
                text: "ok".into(),
                language: "en".into(),
            })
        }
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
            "qwen3-asr-0.6",
            "hima-asr",
            "fluid-speech-pro-gguf-q8_0",
        ];
        for (model, id) in SttModel::ALL.into_iter().zip(ids) {
            assert_eq!(model.as_str(), id);
            assert_eq!(SttModel::parse(id), Some(model));
            assert!(SttModel::parse(&format!("{id}-nope")).is_none());
        }
        assert!(SttModel::parse("tiny").is_none());
        assert!(SttModel::parse("base").is_none());
        assert!(SttModel::QwenAsr06.is_qwen_asr());
        assert!(!SttModel::Small.is_qwen_asr());
        assert!(!SttModel::MoonshineStreamingSmall.is_qwen_asr());
        assert!(SttModel::MoonshineStreamingSmall.is_moonshine());
        assert!(!SttModel::QwenAsr06.is_moonshine());
        assert!(SttModel::HimaAsr.is_hima());
        assert!(!SttModel::Small.is_hima());
        assert!(!SttModel::QwenAsr06.is_hima());
        assert!(!SttModel::HimaAsr.is_moonshine());
        assert!(!SttModel::HimaAsr.is_qwen_asr());
    }

    #[test]
    fn pcm_to_utterance_accepts_v0_pcm() {
        let wav = audio::WavPcm {
            format: audio::PcmFormat::v0(),
            samples: vec![0; FRAME_SAMPLES],
        };
        let utterance = pcm_to_utterance(&wav, TurnId(3)).expect("v0 wav");
        assert_eq!(utterance.turn, TurnId(3));
        assert!(!utterance.frames.is_empty());
    }

    #[test]
    fn transcribe_wav_reads_valid_wav() {
        let mut stt = NoopStt;
        assert_eq!(stt.name(), "noop");
        stt.start_turn(TurnId(0), &Cancel::new())
            .expect("default start");
        assert!(stt
            .push_frame(
                &AudioFrame {
                    seq: 0,
                    sample_rate_hz: DEFAULT_SAMPLE_RATE_HZ,
                    channels: DEFAULT_CHANNELS,
                    samples: vec![0; FRAME_SAMPLES],
                    capture_pcm: None,
                },
                &Cancel::new(),
            )
            .expect("default push")
            .is_none());
        stt.cancel_turn(TurnId(0));
        let path =
            std::env::temp_dir().join(format!("syllabix-stt-tiny-wav-{}", std::process::id()));
        let wav = audio::WavPcm {
            format: audio::PcmFormat::v0(),
            samples: vec![0; FRAME_SAMPLES],
        };
        audio::write_wav(std::fs::File::create(&path).expect("temp wav"), &wav).expect("write wav");
        let transcript = transcribe_wav(&mut stt, &path, &Cancel::new()).expect("transcribe");
        assert_eq!(transcript.text, "ok");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn transcribe_wav_rejects_missing_file() {
        let mut stt = NoopStt;
        assert_eq!(stt.name(), "noop");
        let path =
            std::env::temp_dir().join(format!("syllabix-stt-missing-wav-{}", std::process::id()));
        let err = transcribe_wav(&mut stt, &path, &Cancel::new()).expect_err("missing wav");
        assert!(err.to_string().contains("No such file") || err.to_string().contains("os error"));
    }
}
