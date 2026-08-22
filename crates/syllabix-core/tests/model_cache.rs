//! Merge gate for PR 6: interrupted/corrupt downloads and offline cache reuse.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use sha2::{Digest, Sha256};
use syllabix_core::{
    BlockedFetcher, Cancel, Fetcher, Manifest, ModelAsset, ModelCache, ModelLayer, NoProgress,
    Result,
};

struct BytesFetcher {
    body: Vec<u8>,
    fail_first: bool,
    calls: AtomicUsize,
}

impl Fetcher for BytesFetcher {
    fn fetch(
        &self,
        _url: &str,
        writer: &mut dyn Write,
        on_chunk: &mut dyn FnMut(u64),
        cancel: &Cancel,
    ) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(syllabix_core::Error::Cancelled);
        }
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_first && n == 0 {
            writer.write_all(&self.body[..self.body.len() / 2])?;
            on_chunk((self.body.len() / 2) as u64);
            return Err(syllabix_core::Error::ModelCache {
                message: "interrupted".into(),
            });
        }
        writer.write_all(&self.body)?;
        on_chunk(self.body.len() as u64);
        Ok(())
    }
}

fn asset(id: &str, body: &[u8]) -> ModelAsset {
    ModelAsset {
        id: id.into(),
        layer: ModelLayer::Stt,
        file_name: format!("{id}.bin"),
        url: format!("https://example.invalid/{id}"),
        sha256: hex(Sha256::digest(body)),
        size_bytes: body.len() as u64,
    }
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn cache_with(assets: Vec<ModelAsset>) -> ModelCache {
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir()
        .join("syllabix-pr6-model-cache")
        .join(format!(
            "{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::SeqCst)
        ));
    std::fs::create_dir_all(&root).unwrap();
    ModelCache::new(root, Manifest { version: 1, assets })
}

#[test]
fn interrupted_then_retry_commits_only_complete_file() {
    let body = b"complete-whisper-fixture".to_vec();
    let item = asset("whisper-small", &body);
    let cache = cache_with(vec![item.clone()]);
    let fetcher = BytesFetcher {
        body: body.clone(),
        fail_first: true,
        calls: AtomicUsize::new(0),
    };

    let err = cache
        .resolve(&item, &fetcher, &mut NoProgress, &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("interrupted"));
    assert!(!cache.asset_path(&item).exists());

    let path = cache
        .resolve(&item, &fetcher, &mut NoProgress, &Cancel::new())
        .unwrap();
    assert_eq!(std::fs::read(path).unwrap(), body);
}

#[test]
fn corrupt_payload_never_becomes_the_cached_file() {
    let good = b"kokoro-onnx-fixture".to_vec();
    let item = asset("kokoro", &good);
    let cache = cache_with(vec![item.clone()]);
    let fetcher = BytesFetcher {
        body: b"not-the-checksummed-bytes".to_vec(),
        fail_first: false,
        calls: AtomicUsize::new(0),
    };
    let err = cache
        .resolve(&item, &fetcher, &mut NoProgress, &Cancel::new())
        .unwrap_err();
    assert!(
        err.to_string().contains("checksum") || err.to_string().contains("size"),
        "{err}"
    );
    assert!(!cache.asset_path(&item).exists());
}

#[test]
fn populated_cache_resolves_every_asset_with_network_blocked() {
    let bodies = [
        ("silero", &b"vad"[..]),
        ("whisper-small", &b"stt"[..]),
        ("qwen3.5-0.8b", &b"llm-small"[..]),
        ("qwen3.5-2b", &b"llm"[..]),
        ("llama-3.2-1b", &b"llm-llama"[..]),
        ("kokoro", &b"tts"[..]),
        ("kokoro-voice", &b"voice"[..]),
    ];
    let assets: Vec<_> = bodies.iter().map(|(id, body)| asset(id, body)).collect();
    let cache = cache_with(assets.clone());
    for (asset, (_, body)) in assets.iter().zip(bodies) {
        let fetcher = BytesFetcher {
            body: body.to_vec(),
            fail_first: false,
            calls: AtomicUsize::new(0),
        };
        cache
            .resolve(asset, &fetcher, &mut NoProgress, &Cancel::new())
            .unwrap();
    }

    let blocked = BlockedFetcher::default();
    let paths = cache
        .resolve_all(&blocked, &mut NoProgress, &Cancel::new())
        .expect("populated cache must not fetch");
    assert_eq!(paths.len(), 7);
    assert_eq!(blocked.hits.load(Ordering::SeqCst), 0);
    for (path, (_, body)) in paths.iter().zip(bodies) {
        assert_eq!(std::fs::read(path).unwrap(), body);
    }
}

#[test]
fn resolving_one_llm_id_does_not_fetch_the_other() {
    let small = asset("qwen3.5-0.8b", b"zero-eight");
    let two = asset("qwen3.5-2b", b"two-b-bytes");
    let llama = asset("llama-3.2-1b", b"llama-bytes");
    let cache = cache_with(vec![small.clone(), two.clone(), llama.clone()]);
    let two_fetch = BytesFetcher {
        body: b"two-b-bytes".to_vec(),
        fail_first: false,
        calls: AtomicUsize::new(0),
    };
    let small_fetch = BytesFetcher {
        body: b"zero-eight".to_vec(),
        fail_first: false,
        calls: AtomicUsize::new(0),
    };
    cache
        .resolve(&small, &small_fetch, &mut NoProgress, &Cancel::new())
        .unwrap();
    assert_eq!(small_fetch.calls.load(Ordering::SeqCst), 1);
    assert_eq!(two_fetch.calls.load(Ordering::SeqCst), 0);
    cache
        .resolve(&small, &two_fetch, &mut NoProgress, &Cancel::new())
        .unwrap();
    assert_eq!(two_fetch.calls.load(Ordering::SeqCst), 0);
    assert!(!cache.asset_path(&two).exists());
    assert!(!cache.asset_path(&llama).exists());
}

#[test]
fn v0_manifest_is_complete() {
    let m = Manifest::v0();
    assert_eq!(m.assets.len(), 11);
    assert!(m.asset("silero").is_some());
    assert!(m.asset("whisper-small").is_some());
    assert!(m.asset("whisper-medium").is_some());
    assert!(m.asset("whisper-large-v3-turbo").is_some());
    assert!(m.asset("whisper-medium-q5_0").is_some());
    assert!(m.asset("whisper-large-v3-turbo-q5_0").is_some());
    assert!(m.asset("qwen3.5-0.8b").is_some());
    assert!(m.asset("qwen3.5-2b").is_some());
    assert!(m.asset("llama-3.2-1b").is_some());
    assert!(m.asset("kokoro").is_some());
    assert!(m.asset("kokoro-voice").is_some());
}
