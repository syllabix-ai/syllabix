//! Turn-based LFM3-Audio backend for [`crate::s2s::S2sEngine`].

use std::path::{Path, PathBuf};

use syllabix_native::audio::{f32_to_i16, i16_to_f32, PcmConverter, PcmFormat};
use syllabix_native::models::{Fetcher, Manifest, ModelCache, Progress};
use syllabix_native::{Lfm3AudioContext, Lfm3S2sError};

use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::s2s::{
    S2sEngine, LFM3_AUDIO_2B, LFM3_AUDIO_2B_MMPROJ, LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
};
use crate::types::{GenerationId, SynthesizedAudio, Utterance};

/// Default cap on native PCM samples per `generate` (~42 s at 24 kHz).
pub const LFM3_AUDIO_DEFAULT_MAX_OUT_SAMPLES: i32 = 1_024_000;

/// LFM3-Audio-2B turn-based S2S engine over libmtmd.
pub struct Lfm3AudioEngine {
    ctx: Lfm3AudioContext,
    n_threads: i32,
    max_out_samples: i32,
    /// Native-rate PCM samples emitted on the last `generate` (24 kHz for LFM3-A).
    last_native_out_samples: u64,
    last_text: String,
}

impl Lfm3AudioEngine {
    pub fn name(&self) -> &'static str {
        LFM3_AUDIO_2B
    }

    pub fn from_paths(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        n_threads: i32,
    ) -> Result<Self> {
        Self::from_paths_with_seed(model, mmproj, n_threads, u32::MAX)
    }

    pub fn from_paths_with_seed(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        n_threads: i32,
        seed: u32,
    ) -> Result<Self> {
        Self::from_paths_with_n_ctx(model, mmproj, n_threads, seed, 0)
    }

    pub fn from_paths_with_n_ctx(
        model: impl AsRef<Path>,
        mmproj: impl AsRef<Path>,
        n_threads: i32,
        seed: u32,
        n_ctx: u32,
    ) -> Result<Self> {
        let ctx = Lfm3AudioContext::load(
            model,
            mmproj,
            n_threads,
            seed,
            n_ctx,
            LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
        )
        .map_err(|message| Error::Provider {
            provider: LFM3_AUDIO_2B,
            message,
        })?;
        Ok(Self {
            ctx,
            n_threads,
            max_out_samples: LFM3_AUDIO_DEFAULT_MAX_OUT_SAMPLES,
            last_native_out_samples: 0,
            last_text: String::new(),
        })
    }

    pub fn from_cache(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        n_threads: i32,
    ) -> Result<Self> {
        Self::from_cache_with_seed(cache, fetcher, progress, cancel, n_threads, u32::MAX)
    }

    pub fn from_cache_with_seed(
        cache: &ModelCache,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
        n_threads: i32,
        seed: u32,
    ) -> Result<Self> {
        let manifest = Manifest::v0();
        let model = resolve_asset(cache, &manifest, LFM3_AUDIO_2B, fetcher, progress, cancel)?;
        let mmproj = resolve_asset(
            cache,
            &manifest,
            LFM3_AUDIO_2B_MMPROJ,
            fetcher,
            progress,
            cancel,
        )?;
        Self::from_paths_with_seed(model, mmproj, n_threads, seed)
    }

    pub fn with_max_out_samples(mut self, max_out_samples: i32) -> Self {
        self.max_out_samples = max_out_samples;
        self
    }

    /// Thread count passed through to the native loader.
    pub fn n_threads(&self) -> i32 {
        self.n_threads
    }

    /// Native PCM sample count from the most recent `generate` (before rate conversion).
    pub fn last_native_out_samples(&self) -> u64 {
        self.last_native_out_samples
    }

    pub fn n_past(&self) -> i32 {
        self.ctx.n_past()
    }

    pub fn reset_conversation(&mut self) {
        self.ctx.reset();
    }

    fn utterance_to_f32_in(&self, utterance: &Utterance) -> Result<Vec<f32>> {
        let pcm_i16 = utterance.pcm();
        let mut f32 = i16_to_f32(&pcm_i16);
        if self.ctx.in_rate_hz() != 16_000 {
            let mut conv = PcmConverter::new(
                PcmFormat {
                    sample_rate_hz: 16_000,
                    channels: 1,
                },
                PcmFormat {
                    sample_rate_hz: self.ctx.in_rate_hz() as u32,
                    channels: 1,
                },
            )
            .map_err(|err| Error::Provider {
                provider: LFM3_AUDIO_2B,
                message: err.to_string(),
            })?;
            f32 = conv.push(&f32);
            f32.extend(conv.flush());
        }
        Ok(f32)
    }
}

impl S2sEngine for Lfm3AudioEngine {
    fn name(&self) -> &'static str {
        Self::name(self)
    }

    fn last_text(&self) -> Option<&str> {
        if self.last_text.is_empty() {
            None
        } else {
            Some(&self.last_text)
        }
    }

    fn generate(
        &mut self,
        utterance: &Utterance,
        cancel: &Cancel,
        on_audio: &mut dyn FnMut(SynthesizedAudio) -> Result<()>,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if utterance.frames.is_empty() {
            return Err(Error::Provider {
                provider: self.name(),
                message: "utterance has no frames".into(),
            });
        }
        let generation = cancel.generation();
        self.last_native_out_samples = 0;
        self.last_text.clear();
        let pcm_in = self.utterance_to_f32_in(utterance)?;
        let out_rate = self.ctx.out_rate_hz();
        let mut conv = PcmConverter::new(
            PcmFormat {
                sample_rate_hz: out_rate as u32,
                channels: 1,
            },
            PcmFormat::v0(),
        )
        .map_err(|err| Error::Provider {
            provider: self.name(),
            message: err.to_string(),
        })?;
        let mut index: u32 = 0;
        let mut native_out_samples = 0u64;
        let mut abort_check = S2sAbortCheck { cancel, generation };
        let abort_user = (&mut abort_check as *mut S2sAbortCheck).cast();
        let turn = utterance.turn;
        let mut emit = |samples: Vec<i16>, is_last: bool| -> Result<(), Lfm3S2sError> {
            let chunk = SynthesizedAudio {
                turn,
                generation,
                index,
                samples,
                is_last,
            };
            index = index.saturating_add(1);
            on_audio(chunk).map_err(|err| Lfm3S2sError::Failed(err.to_string()))
        };
        let result = unsafe {
            self.ctx.generate_streaming(
                &pcm_in,
                self.max_out_samples,
                Some(s2s_abort),
                abort_user,
                &mut |rate, native, is_last| {
                    if cancel.is_shutdown() || cancel.is_stale(generation) {
                        return Err(Lfm3S2sError::Cancelled);
                    }
                    if rate != out_rate {
                        return Err(Lfm3S2sError::Failed(format!(
                            "unexpected native rate {rate} (expected {out_rate})"
                        )));
                    }
                    native_out_samples += native.len() as u64;
                    let mut out = conv.push(native);
                    if is_last {
                        out.extend(conv.flush());
                    }
                    let pcm16 = f32_to_i16(&out);
                    if !pcm16.is_empty() || is_last {
                        emit(pcm16, is_last)?;
                    }
                    Ok(())
                },
            )
        };
        self.last_native_out_samples = native_out_samples;
        match result {
            Ok(()) => {
                if cancel.is_stale(generation) || cancel.is_shutdown() {
                    return Err(Error::Cancelled);
                }
                self.last_text = self.ctx.last_text().to_string();
                Ok(())
            }
            Err(Lfm3S2sError::Cancelled) => Err(Error::Cancelled),
            Err(Lfm3S2sError::Failed(_)) if cancel.is_shutdown() => Err(Error::Cancelled),
            Err(Lfm3S2sError::Failed(message)) => Err(Error::Provider {
                provider: self.name(),
                message,
            }),
        }
    }
}

fn resolve_asset(
    cache: &ModelCache,
    manifest: &Manifest,
    id: &str,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
) -> Result<PathBuf> {
    let asset = manifest.asset(id).ok_or_else(|| Error::Provider {
        provider: LFM3_AUDIO_2B,
        message: format!("manifest missing asset {id}"),
    })?;
    cache
        .resolve(asset, fetcher, progress, cancel)
        .map_err(|err| Error::Provider {
            provider: LFM3_AUDIO_2B,
            message: err.to_string(),
        })
}

struct S2sAbortCheck<'a> {
    cancel: &'a Cancel,
    generation: GenerationId,
}

unsafe extern "C" fn s2s_abort(user_data: *mut std::ffi::c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    let check = unsafe { &*(user_data as *const S2sAbortCheck<'_>) };
    check.cancel.is_shutdown() || check.cancel.is_stale(check.generation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_is_manifest_id() {
        #[cfg(coverage)]
        {
            use crate::s2s::S2sEngine;
            let engine = Lfm3AudioEngine {
                ctx: Lfm3AudioContext::load(
                    "fake-s2s",
                    "fake-mmproj",
                    1,
                    0,
                    0,
                    LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
                )
                .expect("fake ctx"),
                n_threads: 1,
                max_out_samples: 0,
                last_native_out_samples: 0,
                last_text: String::new(),
            };
            assert_eq!(engine.name(), LFM3_AUDIO_2B);
            assert_eq!(S2sEngine::name(&engine), LFM3_AUDIO_2B);
        }
        #[cfg(not(coverage))]
        {
            assert_eq!(LFM3_AUDIO_2B, "lfm3-audio-2b");
        }
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_streams_fake_pcm() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("coverage ctx"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(1),
            frames: scripted_frames(1, 1, 0),
        };
        let cancel = Cancel::new();
        let mut chunks = Vec::new();
        engine
            .generate(&utterance, &cancel, &mut |audio| {
                chunks.push(audio);
                Ok(())
            })
            .expect("fake generate");
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| c.is_last));
        assert!(engine.last_native_out_samples() > 0);
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_s2s_abort_handles_null_user_data() {
        assert!(!unsafe { s2s_abort(std::ptr::null_mut()) });
    }

    #[cfg(coverage)]
    #[test]
    fn empty_utterance_is_provider_error() {
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("coverage ctx"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(0),
            frames: vec![],
        };
        let cancel = Cancel::new();
        let err = engine
            .generate(&utterance, &cancel, &mut |_| Ok(()))
            .unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_from_cache_reports_missing_assets_without_download() {
        use syllabix_native::models::{BlockedFetcher, NoProgress};
        let root = std::env::temp_dir().join(format!(
            "syllabix-s2s-missing-cov-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(root, Manifest::v0());
        let outcome = Lfm3AudioEngine::from_cache(
            &cache,
            &BlockedFetcher::default(),
            &mut NoProgress,
            &Cancel::new(),
            1,
        );
        assert!(outcome.is_err());
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_from_paths_builds_engine_and_accessors() {
        use crate::s2s::S2sEngine;
        let engine = Lfm3AudioEngine::from_paths("fake-s2s", "fake-mmproj", 2).expect("from_paths");
        assert_eq!(engine.n_threads(), 2);
        assert_eq!(engine.last_native_out_samples(), 0);
        assert_eq!(S2sEngine::name(&engine), LFM3_AUDIO_2B);
        let engine = Lfm3AudioEngine::from_paths_with_seed("fake-s2s", "fake-mmproj", 3, 7)
            .expect("from_paths_with_seed")
            .with_max_out_samples(512);
        assert_eq!(engine.n_threads(), 3);
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_from_paths_reports_load_failure() {
        let err = match Lfm3AudioEngine::from_paths("fail-s2s", "fake-mmproj", 1) {
            Err(err) => err,
            Ok(_) => panic!("fail-s2s must not load"),
        };
        assert!(matches!(err, Error::Provider { .. }));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_from_cache_with_seed_is_blocked_without_weights() {
        use syllabix_native::models::{BlockedFetcher, NoProgress};
        let root = std::env::temp_dir().join(format!(
            "syllabix-s2s-cache-seed-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(root, Manifest::v0());
        assert!(Lfm3AudioEngine::from_cache_with_seed(
            &cache,
            &BlockedFetcher::default(),
            &mut NoProgress,
            &Cancel::new(),
            1,
            9,
        )
        .is_err());
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_on_audio_failure_is_provider_error() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("coverage ctx"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(3),
            frames: scripted_frames(3, 1, 0),
        };
        let err = engine
            .generate(&utterance, &Cancel::new(), &mut |_| {
                Err(Error::Provider {
                    provider: "test",
                    message: "callback fail".into(),
                })
            })
            .unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_stale_in_pcm_callback() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("coverage ctx"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(4),
            frames: scripted_frames(4, 1, 0),
        };
        let cancel = Cancel::new();
        let err = engine
            .generate(&utterance, &cancel, &mut |_| {
                cancel.cancel_generation();
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_stale_after_success() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("coverage ctx"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(5),
            frames: scripted_frames(5, 1, 0),
        };
        let cancel = Cancel::new();
        let err = engine
            .generate(&utterance, &cancel, &mut |audio| {
                if audio.is_last {
                    cancel.cancel_generation();
                }
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_wrong_native_rate() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s-badrate",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("badrate"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(6),
            frames: scripted_frames(6, 1, 0),
        };
        let err = engine
            .generate(&utterance, &Cancel::new(), &mut |_| Ok(()))
            .unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_resamples_utterance_pcm() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s-in24k",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("in24k"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(7),
            frames: scripted_frames(7, 2, 0),
        };
        engine
            .generate(&utterance, &Cancel::new(), &mut |_| Ok(()))
            .expect("resampled generate");
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_native_failure() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s-nativeerr",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("nativeerr"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(8),
            frames: scripted_frames(8, 1, 0),
        };
        let err = engine
            .generate(&utterance, &Cancel::new(), &mut |_| Ok(()))
            .unwrap_err();
        assert!(matches!(err, Error::Provider { .. }));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_honours_abort_callback() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("coverage ctx"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(9),
            frames: scripted_frames(9, 1, 0),
        };
        let cancel = Cancel::new();
        let mut saw_pcm = false;
        let err = engine
            .generate(&utterance, &cancel, &mut |_| {
                if !saw_pcm {
                    saw_pcm = true;
                    cancel.cancel_generation();
                }
                Ok(())
            })
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_generate_honours_cancel() {
        use crate::fake::scripted_frames;
        use crate::types::TurnId;
        let mut engine = Lfm3AudioEngine {
            ctx: Lfm3AudioContext::load(
                "fake-s2s",
                "fake-mmproj",
                1,
                0,
                0,
                LFM3_AUDIO_VOICE_CHAT_SYSTEM_PROMPT,
            )
            .expect("coverage ctx"),
            n_threads: 1,
            max_out_samples: 0,
            last_native_out_samples: 0,
            last_text: String::new(),
        };
        let utterance = Utterance {
            turn: TurnId(2),
            frames: scripted_frames(2, 1, 0),
        };
        let cancel = Cancel::new();
        cancel.shutdown();
        let err = engine
            .generate(&utterance, &cancel, &mut |_| Ok(()))
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
    }

    #[cfg(not(coverage))]
    #[test]
    fn from_cache_reports_missing_assets_without_download() {
        use syllabix_native::models::{BlockedFetcher, NoProgress};
        let root = std::env::temp_dir().join(format!(
            "syllabix-s2s-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(root, Manifest::v0());
        let outcome = Lfm3AudioEngine::from_cache(
            &cache,
            &BlockedFetcher::default(),
            &mut NoProgress,
            &Cancel::new(),
            1,
        );
        assert!(outcome.is_err());
    }
}
