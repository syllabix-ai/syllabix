//! Fetch, checksum, and unpack the pinned Goose CLI through [`ModelCache`].

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::cache::ModelCache;
use super::download::Fetcher;
use super::manifest::{ModelAsset, ModelLayer};
use super::progress::Progress;
use crate::cancel::Cancel;
use crate::dist::{
    goose_binary_for_target, goose_exe_name, goose_release_download_url, host_rustc_target,
    GooseBinary, GOOSE_VERSION,
};
use crate::error::{Error, Result};

/// Return the path to the unpacked Goose executable for this host.
///
/// Downloads the pinned archive through [`ModelCache`] when missing. A
/// checksum mismatch is a hard error. Not called for `developer_harness.agent: syllabix`.
pub(crate) fn resolve_goose_binary(
    cache: &ModelCache,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
) -> Result<PathBuf> {
    let target = host_rustc_target().ok_or_else(|| Error::Config {
        field: "developer_harness.agent".into(),
        message: format!(
            "goose has no pinned binary for {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ),
    })?;
    let pin = goose_binary_for_target(target).ok_or_else(|| Error::Config {
        field: "developer_harness.agent".into(),
        message: format!("goose has no pinned binary for {target}"),
    })?;
    resolve_goose_binary_from_pin(cache, fetcher, progress, cancel, pin)
}

pub(crate) fn resolve_goose_binary_from_pin(
    cache: &ModelCache,
    fetcher: &dyn Fetcher,
    progress: &mut dyn Progress,
    cancel: &Cancel,
    pin: &GooseBinary,
) -> Result<PathBuf> {
    // Stamp matches pin and the binary is present: skip archive re-hash.
    if let Some(exe) = stamped_goose_exe(cache, pin) {
        return Ok(exe);
    }
    let archive = cache.resolve(&goose_asset(pin), fetcher, progress, cancel)?;
    unpack_goose_archive(cache, pin, &archive)
}

fn goose_asset(pin: &GooseBinary) -> ModelAsset {
    ModelAsset {
        id: "goose".into(),
        layer: ModelLayer::Agent,
        file_name: pin.archive_name.to_string(),
        url: goose_release_download_url(pin.archive_name),
        sha256: pin.sha256.to_string(),
        size_bytes: pin.size_bytes,
    }
}

fn unpack_dir(cache: &ModelCache, pin: &GooseBinary) -> PathBuf {
    cache
        .models_dir()
        .join(format!("goose-{GOOSE_VERSION}-{}", pin.rustc_target))
}

fn unpacked_exe(dir: &Path, pin: &GooseBinary) -> PathBuf {
    let name = goose_exe_name(pin.rustc_target);
    let nested = dir.join("goose-package").join(name);
    let top = dir.join(name);
    if nested.is_file() {
        nested
    } else {
        top
    }
}

fn stamped_goose_exe(cache: &ModelCache, pin: &GooseBinary) -> Option<PathBuf> {
    let dest = unpack_dir(cache, pin);
    let stamp = dest.join(".sha256");
    let exe = unpacked_exe(&dest, pin);
    if exe.is_file()
        && fs::read_to_string(stamp)
            .map(|s| s.trim() == pin.sha256)
            .unwrap_or(false)
    {
        Some(exe)
    } else {
        None
    }
}

fn unpack_goose_archive(cache: &ModelCache, pin: &GooseBinary, archive: &Path) -> Result<PathBuf> {
    if let Some(exe) = stamped_goose_exe(cache, pin) {
        return Ok(exe);
    }
    let dest = unpack_dir(cache, pin);
    let tmp = dest.with_extension("unpacking");
    if tmp.exists() {
        fs::remove_dir_all(&tmp).map_err(|err| Error::ModelCache {
            message: format!("goose unpack temp {}: {err}", tmp.display()),
        })?;
    }
    if dest.exists() {
        fs::remove_dir_all(&dest).map_err(|err| Error::ModelCache {
            message: format!("goose unpack {}: {err}", dest.display()),
        })?;
    }
    fs::create_dir_all(&tmp).map_err(|err| Error::ModelCache {
        message: format!("goose unpack {}: {err}", tmp.display()),
    })?;
    extract_archive(archive, &tmp)?;
    reject_escaping_paths(&tmp)?;
    let exe = unpacked_exe(&tmp, pin);
    if !exe.is_file() {
        return Err(Error::ModelCache {
            message: format!(
                "goose: unpacked archive {} has no {}",
                pin.archive_name,
                goose_exe_name(pin.rustc_target)
            ),
        });
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&exe)?.permissions();
        perms.set_mode(perms.mode() | 0o755);
        fs::set_permissions(&exe, perms)?;
    }
    fs::write(tmp.join(".sha256"), pin.sha256).map_err(|err| Error::ModelCache {
        message: format!("goose unpack stamp: {err}"),
    })?;
    fs::rename(&tmp, &dest).map_err(|err| Error::ModelCache {
        message: format!("goose unpack rename: {err}"),
    })?;
    Ok(unpacked_exe(&dest, pin))
}

fn extract_archive(archive: &Path, dest: &Path) -> Result<()> {
    let mut cmd = Command::new("tar");
    let name = archive
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    if name.ends_with(".tar.bz2") {
        cmd.arg("-xjf");
    } else {
        cmd.arg("-xf");
    }
    let output = cmd
        .arg(archive)
        .arg("-C")
        .arg(dest)
        .output()
        .map_err(|err| Error::ModelCache {
            message: format!("goose unpack: tar failed to start: {err}"),
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(Error::ModelCache {
        message: format!("goose unpack: tar failed: {}", stderr.trim()),
    })
}

fn reject_escaping_paths(root: &Path) -> Result<()> {
    let canon = fs::canonicalize(root).map_err(|err| Error::ModelCache {
        message: format!("goose unpack canonicalize {}: {err}", root.display()),
    })?;
    fn walk(dir: &Path, root: &Path) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let meta = fs::symlink_metadata(&path)?;
            if meta.file_type().is_symlink() {
                let target = fs::canonicalize(&path).map_err(|err| Error::ModelCache {
                    message: format!("goose unpack symlink {}: {err}", path.display()),
                })?;
                if !target.starts_with(root) {
                    return Err(Error::ModelCache {
                        message: format!("goose unpack: archive path escapes {}", root.display()),
                    });
                }
            } else {
                let target = fs::canonicalize(&path).map_err(|err| Error::ModelCache {
                    message: format!("goose unpack {}: {err}", path.display()),
                })?;
                if !target.starts_with(root) {
                    return Err(Error::ModelCache {
                        message: format!("goose unpack: archive path escapes {}", root.display()),
                    });
                }
            }
            if meta.is_dir() && !meta.file_type().is_symlink() {
                walk(&path, root)?;
            }
        }
        Ok(())
    }
    walk(&canon, &canon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::GOOSE_BINARIES;
    use crate::models::download::BlockedFetcher;
    use crate::models::progress::NoProgress;
    use sha2::{Digest, Sha256};
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct BytesFetcher {
        body: Vec<u8>,
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
                return Err(Error::Cancelled);
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            writer.write_all(&self.body)?;
            on_chunk(self.body.len() as u64);
            Ok(())
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

    fn scratch() -> PathBuf {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir()
            .join("syllabix-goose-cache")
            .join(format!(
                "{}-{}",
                std::process::id(),
                SEQ.fetch_add(1, Ordering::SeqCst)
            ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_archive() -> (Vec<u8>, PathBuf) {
        let staging = scratch().join("src");
        fs::create_dir_all(&staging).unwrap();
        let bin = staging.join("goose");
        fs::write(&bin, b"#!/bin/sh\necho goose-fixture\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&bin).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&bin, perms).unwrap();
        }
        let archive = scratch().join("goose-x86_64-unknown-linux-gnu.tar.bz2");
        let status = Command::new("tar")
            .arg("-cjf")
            .arg(&archive)
            .arg("-C")
            .arg(&staging)
            .arg("goose")
            .status()
            .expect("tar must start");
        assert!(status.success(), "tar -cjf fixture");
        (fs::read(&archive).unwrap(), archive)
    }

    fn pin_for(body: &[u8], archive_name: &'static str) -> GooseBinary {
        let digest = hex(Sha256::digest(body));
        // Leak test pins: GOOSE_BINARIES is 'static; tests pass a local GooseBinary.
        GooseBinary {
            rustc_target: "x86_64-unknown-linux-gnu",
            archive_name,
            sha256: Box::leak(digest.into_boxed_str()),
            size_bytes: body.len() as u64,
        }
    }

    #[test]
    fn host_pin_uses_the_dist_table() {
        if let Some(target) = host_rustc_target() {
            let pin = goose_binary_for_target(target).expect("host is a dist target");
            let asset = goose_asset(pin);
            assert_eq!(asset.id, "goose");
            assert_eq!(asset.layer, ModelLayer::Agent);
            assert_eq!(asset.file_name, pin.archive_name);
            assert_eq!(asset.sha256, pin.sha256);
            assert_eq!(asset.size_bytes, pin.size_bytes);
            assert!(asset.url.contains(pin.archive_name));
        }
    }

    #[test]
    fn checksum_mismatch_is_a_hard_error() {
        let (body, _) = tiny_archive();
        let mut pin = pin_for(&body, "goose-x86_64-unknown-linux-gnu.tar.bz2");
        pin.sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let cache = ModelCache::new(
            scratch(),
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = resolve_goose_binary_from_pin(
            &cache,
            &BytesFetcher {
                body,
                calls: AtomicUsize::new(0),
            },
            &mut NoProgress,
            &Cancel::new(),
            &pin,
        )
        .expect_err("mismatch");
        assert!(matches!(err, Error::ModelCache { .. }), "{err:?}");
        assert!(err.to_string().contains("checksum mismatch"), "{err}");
        assert!(
            !unpack_dir(&cache, &pin).exists(),
            "mismatch must not unpack"
        );
    }

    #[test]
    fn fetch_verify_and_unpack_returns_the_binary() {
        let (body, _) = tiny_archive();
        let pin = pin_for(&body, "goose-x86_64-unknown-linux-gnu.tar.bz2");
        let cache = ModelCache::new(
            scratch(),
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let fetcher = BytesFetcher {
            body: body.clone(),
            calls: AtomicUsize::new(0),
        };
        let path =
            resolve_goose_binary_from_pin(&cache, &fetcher, &mut NoProgress, &Cancel::new(), &pin)
                .expect("unpack");
        assert_eq!(path.file_name().unwrap(), "goose");
        assert!(path.is_file());
        assert_eq!(fetcher.calls.load(Ordering::SeqCst), 1);

        let blocked = BlockedFetcher::default();
        let again =
            resolve_goose_binary_from_pin(&cache, &blocked, &mut NoProgress, &Cancel::new(), &pin)
                .expect("reuse cache");
        assert_eq!(again, path);
        assert_eq!(blocked.hits.load(Ordering::SeqCst), 0);

        // Warm path: stamp proves the unpack. Drop the archive so any
        // ModelCache::resolve would have to re-fetch (BlockedFetcher) or
        // re-hash a missing file. Stamp short-circuit must win.
        let archive_path = cache.asset_path(&goose_asset(&pin));
        assert!(archive_path.is_file(), "first resolve left the archive");
        fs::remove_file(&archive_path).unwrap();
        let warm =
            resolve_goose_binary_from_pin(&cache, &blocked, &mut NoProgress, &Cancel::new(), &pin)
                .expect("stamped unpack skips archive re-hash");
        assert_eq!(warm, path);
        assert_eq!(blocked.hits.load(Ordering::SeqCst), 0);
        assert!(
            !archive_path.exists(),
            "warm path must not recreate archive"
        );

        fs::remove_file(unpack_dir(&cache, &pin).join(".sha256")).unwrap();
        let leftover = unpack_dir(&cache, &pin).with_extension("unpacking");
        fs::create_dir_all(&leftover).unwrap();
        // Archive gone and stamp cleared: resolve must fail without network.
        let missing =
            resolve_goose_binary_from_pin(&cache, &blocked, &mut NoProgress, &Cancel::new(), &pin)
                .expect_err("no stamp and no archive");
        assert!(matches!(missing, Error::ModelCache { .. }), "{missing:?}");
        assert_eq!(blocked.hits.load(Ordering::SeqCst), 1);

        // Put the archive back and re-unpack from a cold stamp.
        fs::write(&archive_path, &body).unwrap();
        let rebuilt = resolve_goose_binary_from_pin(
            &cache,
            &BlockedFetcher::default(),
            &mut NoProgress,
            &Cancel::new(),
            &pin,
        )
        .expect("re-unpack from cached archive");
        assert_eq!(rebuilt.file_name().unwrap(), "goose");
        assert!(!leftover.exists());
    }

    #[test]
    fn stamped_cache_skips_archive_resolve() {
        let (body, _) = tiny_archive();
        let pin = pin_for(&body, "goose-x86_64-unknown-linux-gnu.tar.bz2");
        let cache = ModelCache::new(
            scratch(),
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let first = resolve_goose_binary_from_pin(
            &cache,
            &BytesFetcher {
                body: body.clone(),
                calls: AtomicUsize::new(0),
            },
            &mut NoProgress,
            &Cancel::new(),
            &pin,
        )
        .expect("first unpack");
        let archive_path = cache.asset_path(&goose_asset(&pin));
        // Corrupt the archive so a re-hash would fail hard.
        fs::write(&archive_path, b"not-the-pinned-bytes").unwrap();
        let blocked = BlockedFetcher::default();
        let second =
            resolve_goose_binary_from_pin(&cache, &blocked, &mut NoProgress, &Cancel::new(), &pin)
                .expect("stamp short-circuit");
        assert_eq!(second, first);
        assert_eq!(blocked.hits.load(Ordering::SeqCst), 0);
        assert_eq!(
            fs::read(&archive_path).unwrap(),
            b"not-the-pinned-bytes",
            "warm path must not touch the archive"
        );
    }

    #[test]
    fn unpack_finds_nested_windows_layout() {
        let staging = scratch().join("nested-src");
        let nested = staging.join("goose-package");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("goose"), b"nested-bin").unwrap();
        let archive_path = scratch().join("nested.tar.bz2");
        let status = Command::new("tar")
            .arg("-cjf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&staging)
            .arg("goose-package")
            .status()
            .unwrap();
        assert!(status.success());
        let body = fs::read(&archive_path).unwrap();
        let pin = pin_for(&body, "goose-x86_64-unknown-linux-gnu.tar.bz2");
        let cache = ModelCache::new(
            scratch(),
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let path = resolve_goose_binary_from_pin(
            &cache,
            &BytesFetcher {
                body,
                calls: AtomicUsize::new(0),
            },
            &mut NoProgress,
            &Cancel::new(),
            &pin,
        )
        .expect("nested");
        assert_eq!(
            path,
            unpack_dir(&cache, &pin).join("goose-package").join("goose")
        );
        assert_eq!(fs::read(&path).unwrap(), b"nested-bin");
    }

    #[test]
    fn blocked_fetch_does_not_unpack() {
        let pin = GOOSE_BINARIES[0];
        let cache = ModelCache::new(
            scratch(),
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let fetcher = BlockedFetcher::default();
        let err =
            resolve_goose_binary_from_pin(&cache, &fetcher, &mut NoProgress, &Cancel::new(), &pin)
                .expect_err("blocked");
        assert!(matches!(err, Error::ModelCache { .. }), "{err:?}");
        assert!(err.to_string().contains("network blocked"), "{err}");
        assert_eq!(fetcher.hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn host_resolve_uses_the_pin_and_blocked_fetcher() {
        if host_rustc_target().is_none() {
            return;
        }
        let cache = ModelCache::new(
            scratch(),
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let fetcher = BlockedFetcher::default();
        let err = resolve_goose_binary(&cache, &fetcher, &mut NoProgress, &Cancel::new())
            .expect_err("blocked host pin");
        assert!(matches!(err, Error::ModelCache { .. }), "{err:?}");
        assert!(err.to_string().contains("network blocked"), "{err}");
        assert_eq!(fetcher.hits.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn unpack_rejects_archive_without_goose_binary() {
        let staging = scratch().join("empty-src");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("readme.txt"), b"no binary").unwrap();
        let archive_path = scratch().join("empty.tar.bz2");
        let status = Command::new("tar")
            .arg("-cjf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&staging)
            .arg("readme.txt")
            .status()
            .unwrap();
        assert!(status.success());
        let body = fs::read(&archive_path).unwrap();
        let pin = pin_for(&body, "goose-x86_64-unknown-linux-gnu.tar.bz2");
        let cache = ModelCache::new(
            scratch(),
            crate::models::Manifest {
                version: 1,
                assets: vec![],
            },
        );
        let err = resolve_goose_binary_from_pin(
            &cache,
            &BytesFetcher {
                body,
                calls: AtomicUsize::new(0),
            },
            &mut NoProgress,
            &Cancel::new(),
            &pin,
        )
        .expect_err("missing binary");
        assert!(err.to_string().contains("has no goose"), "{err}");
    }

    #[test]
    fn tar_xf_extracts_uncompressed_archive() {
        let staging = scratch().join("tar-src");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("goose"), b"ok").unwrap();
        let archive = scratch().join("goose.zip");
        let status = Command::new("tar")
            .arg("-cf")
            .arg(&archive)
            .arg("-C")
            .arg(&staging)
            .arg("goose")
            .status()
            .unwrap();
        assert!(status.success());
        let dest = scratch().join("out");
        fs::create_dir_all(&dest).unwrap();
        extract_archive(&archive, &dest).unwrap();
        assert!(dest.join("goose").is_file());
    }

    #[test]
    fn tar_failure_is_a_cache_error() {
        let dest = scratch().join("bad-out");
        fs::create_dir_all(&dest).unwrap();
        let garbage = scratch().join("not-an-archive.tar.bz2");
        fs::write(&garbage, b"not tar").unwrap();
        let err = extract_archive(&garbage, &dest).expect_err("tar fails");
        assert!(err.to_string().contains("tar failed"), "{err}");
    }
}
