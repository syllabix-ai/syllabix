//! On-disk cache: lookup, atomic write, checksum, offline reuse.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::download::Fetcher;
use super::manifest::{Manifest, ModelAsset};
use super::progress::Progress;
use crate::cancel::Cancel;
use crate::error::{Error, Result};

/// Resolve the cache root from environment variables.
///
/// Order: `SYLLABIX_CACHE_DIR`, then `XDG_CACHE_HOME/syllabix`, then
/// `%LOCALAPPDATA%/syllabix/cache` on Windows, then `HOME/.cache/syllabix`.
pub fn cache_root() -> PathBuf {
    cache_root_from(
        env::var_os("SYLLABIX_CACHE_DIR"),
        env::var_os("XDG_CACHE_HOME"),
        env::var_os("LOCALAPPDATA"),
        env::var_os("HOME"),
    )
}

pub(crate) fn cache_root_from(
    syllabix: Option<std::ffi::OsString>,
    xdg: Option<std::ffi::OsString>,
    local_app_data: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> PathBuf {
    if let Some(dir) = syllabix {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = xdg {
        return PathBuf::from(xdg).join("syllabix");
    }
    if cfg!(windows) {
        if let Some(local) = local_app_data {
            return PathBuf::from(local).join("syllabix").join("cache");
        }
    }
    if let Some(home) = home {
        return PathBuf::from(home).join(".cache").join("syllabix");
    }
    PathBuf::from(".syllabix-cache")
}

/// Manifest + directory that holds verified weight files.
#[derive(Debug, Clone)]
pub struct ModelCache {
    root: PathBuf,
    manifest: Manifest,
}

impl ModelCache {
    /// `root` is the product cache directory (`…/syllabix`). Assets live in
    /// `root/models/v{version}/`.
    pub fn new(root: PathBuf, manifest: Manifest) -> Self {
        Self { root, manifest }
    }

    /// Cache under [`cache_root`] with the v0 manifest.
    pub fn v0() -> Self {
        Self::new(cache_root(), Manifest::v0())
    }

    /// Manifest this cache was constructed with.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Directory that holds the files for this manifest version.
    pub fn models_dir(&self) -> PathBuf {
        self.root
            .join("models")
            .join(format!("v{}", self.manifest.version))
    }

    /// Path where `asset` is stored after a successful download.
    pub fn asset_path(&self, asset: &ModelAsset) -> PathBuf {
        self.models_dir().join(&asset.file_name)
    }

    /// True when the file exists and matches the manifest checksum and size.
    pub fn is_cached(&self, asset: &ModelAsset) -> bool {
        verify_file(&self.asset_path(asset), asset).is_ok()
    }

    /// Return the path to a verified file, downloading if needed.
    pub fn resolve(
        &self,
        asset: &ModelAsset,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<PathBuf> {
        progress.start(asset);
        let dest = self.asset_path(asset);
        if verify_file(&dest, asset).is_ok() {
            progress.finish(asset, true);
            return Ok(dest);
        }
        if dest.exists() {
            let _ = fs::remove_file(&dest);
        }
        download_atomic(asset, &dest, fetcher, progress, cancel)?;
        progress.finish(asset, false);
        Ok(dest)
    }

    /// Resolve every asset in the manifest, in list order.
    pub fn resolve_all(
        &self,
        fetcher: &dyn Fetcher,
        progress: &mut dyn Progress,
        cancel: &Cancel,
    ) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::with_capacity(self.manifest.assets.len());
        for asset in &self.manifest.assets {
            paths.push(self.resolve(asset, fetcher, progress, cancel)?);
        }
        Ok(paths)
    }
}

fn download_atomic(
    asset: &ModelAsset,
    dest: &Path,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
) -> Result<()> {
    if cancel.is_shutdown() {
        return Err(Error::Cancelled);
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name().unwrap_or_default().to_string_lossy()
    ));
    let _ = fs::remove_file(&part);

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&part)?;

    let fetch_result = fetcher.fetch(
        &asset.url,
        &mut file,
        &mut |downloaded| progress.bytes(downloaded, asset.size_bytes),
        cancel,
    );
    if let Err(err) = fetch_result {
        drop(file);
        let _ = fs::remove_file(&part);
        return Err(err);
    }
    file.flush()?;
    file.sync_all()?;
    drop(file);

    if let Err(err) = verify_file(&part, asset) {
        let _ = fs::remove_file(&part);
        return Err(err);
    }
    atomic_replace(&part, dest)?;
    Ok(())
}

fn atomic_replace(from: &Path, to: &Path) -> Result<()> {
    match fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists || cfg!(windows) => {
            let _ = fs::remove_file(to);
            fs::rename(from, to)?;
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

fn verify_file(path: &Path, asset: &ModelAsset) -> Result<()> {
    let mut file = File::open(path).map_err(|err| Error::ModelCache {
        message: format!("{}: {err}", asset.id),
    })?;
    let meta = file.metadata()?;
    if meta.len() != asset.size_bytes {
        return Err(Error::ModelCache {
            message: format!(
                "{}: size {} does not match manifest {}",
                asset.id,
                meta.len(),
                asset.size_bytes
            ),
        });
    }
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    let digest = hex_encode(&hasher.finalize());
    if digest != asset.sha256 {
        return Err(Error::ModelCache {
            message: format!(
                "{}: checksum mismatch (got {digest}, expected {})",
                asset.id, asset.sha256
            ),
        });
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::download::BlockedFetcher;
    use crate::models::manifest::ModelLayer;
    use crate::models::progress::NoProgress;
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    struct ScriptedFetcher {
        body: Vec<u8>,
        fail_after: Option<usize>,
        corrupt: bool,
        calls: AtomicUsize,
        written: Mutex<Vec<usize>>,
    }

    impl ScriptedFetcher {
        fn ok(body: Vec<u8>) -> Self {
            Self {
                body,
                fail_after: None,
                corrupt: false,
                calls: AtomicUsize::new(0),
                written: Mutex::new(Vec::new()),
            }
        }

        fn fail_after(body: Vec<u8>, n: usize) -> Self {
            Self {
                body,
                fail_after: Some(n),
                corrupt: false,
                calls: AtomicUsize::new(0),
                written: Mutex::new(Vec::new()),
            }
        }

        fn corrupt(body: Vec<u8>) -> Self {
            Self {
                body,
                fail_after: None,
                corrupt: true,
                calls: AtomicUsize::new(0),
                written: Mutex::new(Vec::new()),
            }
        }
    }

    impl Fetcher for ScriptedFetcher {
        fn fetch(
            &self,
            _url: &str,
            writer: &mut dyn Write,
            on_chunk: &mut dyn FnMut(u64),
            cancel: &Cancel,
        ) -> Result<()> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            let payload = if self.corrupt {
                let mut v = self.body.clone();
                if let Some(b) = v.first_mut() {
                    *b ^= 0xff;
                }
                v
            } else {
                self.body.clone()
            };
            let limit = if call == 0 {
                self.fail_after.unwrap_or(payload.len())
            } else {
                payload.len()
            };
            let slice = &payload[..limit.min(payload.len())];
            writer.write_all(slice)?;
            on_chunk(slice.len() as u64);
            self.written.lock().unwrap().push(slice.len());
            if call == 0 && self.fail_after.is_some() {
                return Err(Error::ModelCache {
                    message: "connection reset".into(),
                });
            }
            Ok(())
        }
    }

    fn hashed_asset(id: &str, body: &[u8]) -> ModelAsset {
        let digest = hex_encode(&Sha256::digest(body));
        ModelAsset {
            id: id.into(),
            layer: ModelLayer::Vad,
            file_name: format!("{id}.bin"),
            url: format!("https://example.invalid/{id}.bin"),
            sha256: digest,
            size_bytes: body.len() as u64,
        }
    }

    fn scratch() -> PathBuf {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let dir = env::temp_dir()
            .join("syllabix-model-cache-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::SeqCst)
            ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn cache_root_prefers_explicit_override() {
        let p = cache_root_from(
            Some("/custom".into()),
            Some("/xdg".into()),
            Some("/local".into()),
            Some("/home/me".into()),
        );
        assert_eq!(p, PathBuf::from("/custom"));
    }

    #[test]
    fn cache_root_uses_xdg_then_home() {
        let p = cache_root_from(None, Some("/xdg".into()), None, Some("/home/me".into()));
        assert_eq!(p, PathBuf::from("/xdg/syllabix"));
        let p = cache_root_from(None, None, None, Some("/home/me".into()));
        assert_eq!(p, PathBuf::from("/home/me/.cache/syllabix"));
        let p = cache_root_from(None, None, None, None);
        assert_eq!(p, PathBuf::from(".syllabix-cache"));
    }

    #[test]
    fn download_then_offline_reuse_blocks_network() {
        let body = b"silero-onnx-bytes".to_vec();
        let asset = hashed_asset("silero", &body);
        let cache = ModelCache::new(
            scratch(),
            Manifest {
                version: 1,
                assets: vec![asset.clone()],
            },
        );
        let fetcher = ScriptedFetcher::ok(body.clone());
        let path = cache
            .resolve(&asset, &fetcher, &mut NoProgress, &Cancel::new())
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), body);
        assert!(cache.is_cached(&asset));
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 1);

        let blocked = BlockedFetcher::default();
        let again = cache
            .resolve(&asset, &blocked, &mut NoProgress, &Cancel::new())
            .unwrap();
        assert_eq!(again, path);
        assert_eq!(blocked.hits.load(Ordering::SeqCst), 0);

        let all = cache
            .resolve_all(&blocked, &mut NoProgress, &Cancel::new())
            .unwrap();
        assert_eq!(all, vec![path]);
        assert_eq!(blocked.hits.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn interrupted_download_does_not_commit_and_retries() {
        let body = b"0123456789abcdef".to_vec();
        let asset = hashed_asset("whisper-small", &body);
        let cache = ModelCache::new(
            scratch(),
            Manifest {
                version: 1,
                assets: vec![asset.clone()],
            },
        );
        let fetcher = ScriptedFetcher::fail_after(body.clone(), 4);
        let err = cache
            .resolve(&asset, &fetcher, &mut NoProgress, &Cancel::new())
            .unwrap_err();
        assert!(err.to_string().contains("connection reset"));
        assert!(!cache.asset_path(&asset).exists());
        let part = cache
            .asset_path(&asset)
            .with_file_name(format!("{}.part", asset.file_name));
        assert!(!part.exists(), "partial file must not remain after failure");

        let path = cache
            .resolve(&asset, &fetcher, &mut NoProgress, &Cancel::new())
            .unwrap();
        assert_eq!(fs::read(path).unwrap(), body);
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn corrupt_download_is_rejected() {
        let body = b"good-weights-body!!".to_vec();
        let asset = hashed_asset("qwen3.5-2b", &body);
        let cache = ModelCache::new(
            scratch(),
            Manifest {
                version: 1,
                assets: vec![asset.clone()],
            },
        );
        let fetcher = ScriptedFetcher::corrupt(body);
        let err = cache
            .resolve(&asset, &fetcher, &mut NoProgress, &Cancel::new())
            .unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
        assert!(!cache.asset_path(&asset).exists());
    }

    #[test]
    fn corrupt_cached_file_is_redownloaded() {
        let body = b"fresh-bytes".to_vec();
        let asset = hashed_asset("kokoro", &body);
        let cache = ModelCache::new(
            scratch(),
            Manifest {
                version: 1,
                assets: vec![asset.clone()],
            },
        );
        let dest = cache.asset_path(&asset);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(&dest, b"stale").unwrap();
        assert!(!cache.is_cached(&asset));
        let fetcher = ScriptedFetcher::ok(body.clone());
        cache
            .resolve(&asset, &fetcher, &mut NoProgress, &Cancel::new())
            .unwrap();
        assert_eq!(fs::read(dest).unwrap(), body);
    }

    #[test]
    fn cancel_skips_fetch() {
        let body = b"nope".to_vec();
        let asset = hashed_asset("x", &body);
        let cache = ModelCache::new(
            scratch(),
            Manifest {
                version: 1,
                assets: vec![asset.clone()],
            },
        );
        let cancel = Cancel::new();
        cancel.shutdown();
        let fetcher = ScriptedFetcher::ok(body);
        let err = cache
            .resolve(&asset, &fetcher, &mut NoProgress, &cancel)
            .unwrap_err();
        assert!(matches!(err, Error::Cancelled));
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn v0_cache_layout_uses_manifest_version() {
        let cache = ModelCache::new(PathBuf::from("/tmp/syllabix-test-root"), Manifest::v0());
        assert_eq!(cache.manifest().version, 1);
        assert!(cache.models_dir().ends_with(Path::new("models/v1")));
        let silero = cache.manifest().asset("silero").unwrap();
        assert!(cache.asset_path(silero).ends_with("silero_vad.onnx"));
    }
}
