//! Load the blessed v0 native providers from the model cache.

use crate::cancel::Cancel;
use crate::error::Result;
use crate::llm::LlamaLlm;
use crate::models::{Fetcher, ModelCache, Progress};
use crate::stt::WhisperStt;
use crate::tts::KokoroTts;
use crate::vad::SileroVad;

/// Silero + whisper.cpp + llama.cpp + Kokoro, resolved through the v0 manifest.
pub fn load_real_providers(
    cache: &ModelCache,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
) -> Result<(SileroVad, WhisperStt, LlamaLlm, KokoroTts)> {
    let vad = SileroVad::from_cache(cache, fetcher, progress, cancel)?;
    let stt = WhisperStt::from_cache(cache, fetcher, progress, cancel)?;
    let llm = LlamaLlm::from_cache(cache, fetcher, progress, cancel)?;
    let tts = KokoroTts::from_cache(cache, fetcher, progress, cancel)?;
    Ok((vad, stt, llm, tts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Manifest, ModelCache, NoProgress};
    use crate::Cancel;
    use std::path::PathBuf;

    #[test]
    fn missing_manifest_assets_fail_before_native_load() {
        let root: PathBuf = std::env::temp_dir().join(format!(
            "syllabix-real-missing-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let cache = ModelCache::new(
            root,
            Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = match load_real_providers(
            &cache,
            &crate::models::BlockedFetcher::default(),
            &mut NoProgress,
            &Cancel::new(),
        ) {
            Err(err) => err,
            Ok(_) => panic!("empty manifest should fail"),
        };
        assert!(matches!(err, crate::Error::ModelCache { .. }));
    }
}
