//! Names of installable `dist` profile artifacts and GitHub Release URLs.
//!
//! Model weights are downloaded into the cache and are not packed into these files.

/// Cargo profile that produces the standalone executable.
pub const DIST_PROFILE: &str = "dist";

/// Supported distribution target triples. Indices match [`ARTIFACT_FILE_NAMES`].
pub const DIST_TARGETS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-msvc",
];

/// File names copied into `dist/` for [`DIST_TARGETS`].
pub const ARTIFACT_FILE_NAMES: &[&str] = &[
    "syllabix-Linux-x86_64",
    "syllabix-Darwin-arm64",
    "syllabix-Darwin-x86_64",
    "syllabix-Windows-x86_64.exe",
];

/// GNU `sha256sum` text file published next to the four binaries.
pub const SHA256SUMS_FILE_NAME: &str = "SHA256SUMS";

/// `https://github.com/syllabix-ai/syllabix/releases/latest/download/`
pub const RELEASE_LATEST_DOWNLOAD_PREFIX: &str =
    "https://github.com/syllabix-ai/syllabix/releases/latest/download/";

/// Packed-weight size that would mean the binary swallowed Whisper+Llama+Kokoro.
/// Dist artifacts must stay below this; first-run cache holds the models.
pub const MAX_DIST_BINARY_BYTES: u64 = 400_000_000;

/// Return the artifact name for a supported Rust target triple.
pub fn artifact_name_for_target(rustc_target: &str) -> Option<&'static str> {
    DIST_TARGETS
        .iter()
        .position(|t| *t == rustc_target)
        .map(|i| ARTIFACT_FILE_NAMES[i])
}

/// File name for `uname -s` / `uname -m` (and Windows `Windows_NT` / `AMD64`).
///
/// Matches the README curl: `syllabix-$(uname -s)-$(uname -m)`.
pub fn artifact_name_for_uname(sysname: &str, machine: &str) -> Option<&'static str> {
    let windows = sysname == "Windows_NT"
        || sysname.starts_with("MINGW")
        || sysname.starts_with("MSYS")
        || sysname.starts_with("CYGWIN");
    let arch = match machine {
        "x86_64" | "amd64" | "AMD64" => "x86_64",
        "arm64" | "aarch64" => "arm64",
        _ => return None,
    };
    if windows {
        return (arch == "x86_64").then_some(ARTIFACT_FILE_NAMES[3]);
    }
    match (sysname, arch) {
        ("Linux", "x86_64") => Some(ARTIFACT_FILE_NAMES[0]),
        ("Darwin", "arm64") => Some(ARTIFACT_FILE_NAMES[1]),
        ("Darwin", "x86_64") => Some(ARTIFACT_FILE_NAMES[2]),
        _ => None,
    }
}

/// Build the latest-release download URL for an artifact file name.
pub fn latest_release_download_url(file_name: &str) -> String {
    format!("{RELEASE_LATEST_DOWNLOAD_PREFIX}{file_name}")
}

/// Goose CLI version fetched when `developer_harness.agent: goose`.
pub const GOOSE_VERSION: &str = "1.54.0";

/// GitHub release tag for [`GOOSE_VERSION`].
pub const GOOSE_RELEASE_TAG: &str = "v1.54.0";

/// `https://github.com/aaif-goose/goose/releases/download/v1.54.0/`
pub const GOOSE_RELEASE_DOWNLOAD_PREFIX: &str =
    "https://github.com/aaif-goose/goose/releases/download/v1.54.0/";

/// One pinned Goose CLI archive. Rows match [`DIST_TARGETS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GooseBinary {
    /// Rust target triple (same strings as [`DIST_TARGETS`]).
    pub rustc_target: &'static str,
    /// GitHub Release file name (`goose-<triple>.tar.bz2` or `.zip`).
    pub archive_name: &'static str,
    /// Lower-hex SHA-256 of the archive bytes.
    pub sha256: &'static str,
    /// Exact archive size in bytes.
    pub size_bytes: u64,
}

/// Pinned Goose CLI archives, one per Syllabix dist target.
pub const GOOSE_BINARIES: &[GooseBinary] = &[
    GooseBinary {
        rustc_target: "x86_64-unknown-linux-gnu",
        archive_name: "goose-x86_64-unknown-linux-gnu.tar.bz2",
        sha256: "bf01cb6359f7a2b1acb54ca4827e1d059b18aabb3d2b75202e21066cfd3a3396",
        size_bytes: 80_909_320,
    },
    GooseBinary {
        rustc_target: "aarch64-apple-darwin",
        archive_name: "goose-aarch64-apple-darwin.tar.bz2",
        sha256: "63738f0eed2fe9d8585a4ba988c3ffb23d600db0b43858a6cd868fa6b035a7af",
        size_bytes: 74_774_082,
    },
    GooseBinary {
        rustc_target: "x86_64-apple-darwin",
        archive_name: "goose-x86_64-apple-darwin.tar.bz2",
        sha256: "81fd55f4003950bb72719402e2defb1375eef4b9a739d901a365fb737b73aeee",
        size_bytes: 76_781_506,
    },
    GooseBinary {
        rustc_target: "x86_64-pc-windows-msvc",
        archive_name: "goose-x86_64-pc-windows-msvc.zip",
        sha256: "5c39bc91c3eadc231847e21a8a469e778431084437b233a63a950fee22db6239",
        size_bytes: 77_666_256,
    },
];

/// Look up the pinned Goose archive for a Rust target triple.
pub fn goose_binary_for_target(rustc_target: &str) -> Option<&'static GooseBinary> {
    GOOSE_BINARIES
        .iter()
        .find(|binary| binary.rustc_target == rustc_target)
}

/// Rust target triple for this build, when it is a Syllabix dist target.
pub fn host_rustc_target() -> Option<&'static str> {
    if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("x86_64-unknown-linux-gnu")
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("aarch64-apple-darwin")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some("x86_64-apple-darwin")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some("x86_64-pc-windows-msvc")
    } else {
        None
    }
}

/// HTTPS URL of a pinned Goose archive.
pub fn goose_release_download_url(archive_name: &str) -> String {
    format!("{GOOSE_RELEASE_DOWNLOAD_PREFIX}{archive_name}")
}

/// File name of the Goose executable inside a pinned archive.
pub fn goose_exe_name(rustc_target: &str) -> &'static str {
    if rustc_target.contains("windows") {
        "goose.exe"
    } else {
        "goose"
    }
}

/// One GNU `sha256sum` text-mode line (`<hex>  <name>`).
pub fn format_sha256sums_line(sha256_hex: &str, file_name: &str) -> String {
    format!("{sha256_hex}  {file_name}")
}

/// Parse GNU `sha256sum` output. Ignores blank lines and `#` comments.
///
/// Each record is `(lowercase hex, file name)`. Names must be supported artifacts.
pub fn parse_sha256sums(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (hex, name) = split_sha256sums_line(line)
            .ok_or_else(|| format!("line {}: expected '<sha256>  <filename>'", i + 1))?;
        if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("line {}: sha256 must be 64 hex digits", i + 1));
        }
        if !ARTIFACT_FILE_NAMES.contains(&name) {
            return Err(format!("line {}: unknown artifact {name}", i + 1));
        }
        out.push((hex.to_ascii_lowercase(), name.to_string()));
    }
    if out.is_empty() {
        return Err("SHA256SUMS has no artifact lines".into());
    }
    Ok(out)
}

fn split_sha256sums_line(line: &str) -> Option<(&str, &str)> {
    // Text mode: two spaces. Binary mode: space then '*'.
    if let Some((hex, name)) = line.split_once("  ") {
        return Some((hex.trim(), name.trim()));
    }
    if let Some((hex, name)) = line.split_once(" *") {
        return Some((hex.trim(), name.trim()));
    }
    None
}

/// True when `digest` matches the SHA-256 listed for `file_name`.
pub fn sha256sums_contains(entries: &[(String, String)], file_name: &str, digest: &str) -> bool {
    let want = digest.to_ascii_lowercase();
    entries
        .iter()
        .any(|(hex, name)| name == file_name && hex == &want)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_matrix_names_match_uname_style() {
        assert_eq!(DIST_TARGETS.len(), ARTIFACT_FILE_NAMES.len());
        assert_eq!(DIST_PROFILE, "dist");
        assert_eq!(
            artifact_name_for_target("x86_64-unknown-linux-gnu"),
            Some("syllabix-Linux-x86_64")
        );
        assert_eq!(
            artifact_name_for_target("aarch64-apple-darwin"),
            Some("syllabix-Darwin-arm64")
        );
        assert_eq!(
            artifact_name_for_target("x86_64-apple-darwin"),
            Some("syllabix-Darwin-x86_64")
        );
        assert_eq!(
            artifact_name_for_target("x86_64-pc-windows-msvc"),
            Some("syllabix-Windows-x86_64.exe")
        );
        assert!(artifact_name_for_target("wasm32-unknown-unknown").is_none());
        assert!(ARTIFACT_FILE_NAMES[3].ends_with(".exe"));
        for name in &ARTIFACT_FILE_NAMES[..3] {
            assert!(!name.contains('.'), "{name} is a single executable file");
        }
    }

    #[test]
    fn uname_matches_readme_curl_template() {
        assert_eq!(
            artifact_name_for_uname("Linux", "x86_64"),
            Some("syllabix-Linux-x86_64")
        );
        assert_eq!(
            artifact_name_for_uname("Darwin", "arm64"),
            Some("syllabix-Darwin-arm64")
        );
        assert_eq!(
            artifact_name_for_uname("Darwin", "x86_64"),
            Some("syllabix-Darwin-x86_64")
        );
        assert_eq!(
            artifact_name_for_uname("Windows_NT", "AMD64"),
            Some("syllabix-Windows-x86_64.exe")
        );
        assert_eq!(
            artifact_name_for_uname("MINGW64_NT-10.0", "x86_64"),
            Some("syllabix-Windows-x86_64.exe")
        );
        assert!(artifact_name_for_uname("Linux", "aarch64").is_none());
        assert!(artifact_name_for_uname("Darwin", "i386").is_none());
    }

    #[test]
    fn latest_download_urls_are_single_files_under_releases() {
        for name in ARTIFACT_FILE_NAMES {
            let url = latest_release_download_url(name);
            assert!(url.starts_with(RELEASE_LATEST_DOWNLOAD_PREFIX), "{url}");
            assert!(url.ends_with(name), "{url}");
            assert!(!url.contains(' '), "{url}");
        }
        assert_eq!(
            latest_release_download_url(SHA256SUMS_FILE_NAME),
            format!("{RELEASE_LATEST_DOWNLOAD_PREFIX}{SHA256SUMS_FILE_NAME}")
        );
    }

    #[test]
    fn sha256sums_round_trip_rejects_unknown_names() {
        let hex = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let text = format!(
            "{}\n{}\n",
            format_sha256sums_line(hex, ARTIFACT_FILE_NAMES[0]),
            format_sha256sums_line(hex, ARTIFACT_FILE_NAMES[3])
        );
        let parsed = parse_sha256sums(&text).expect("valid");
        assert_eq!(parsed.len(), 2);
        assert!(sha256sums_contains(&parsed, "syllabix-Linux-x86_64", hex));
        assert!(parse_sha256sums(&format_sha256sums_line(hex, "syllabix.exe")).is_err());
        assert!(parse_sha256sums("not-a-sum\n").is_err());
        assert!(parse_sha256sums("").is_err());
        let binary_mode = format!("{hex} *{}", ARTIFACT_FILE_NAMES[1]);
        assert_eq!(
            parse_sha256sums(&binary_mode).unwrap()[0].1,
            "syllabix-Darwin-arm64"
        );
    }

    #[test]
    fn packed_weight_ceiling_is_below_whisper_small() {
        let manifest = crate::models::Manifest::v0();
        let whisper = manifest
            .asset("whisper-small")
            .expect("whisper-small is a v0 asset");
        assert!(MAX_DIST_BINARY_BYTES < whisper.size_bytes);
    }

    #[test]
    fn goose_pins_match_dist_targets_and_v1_54_0() {
        assert_eq!(GOOSE_VERSION, "1.54.0");
        assert_eq!(GOOSE_RELEASE_TAG, "v1.54.0");
        assert_eq!(GOOSE_BINARIES.len(), DIST_TARGETS.len());
        for (i, target) in DIST_TARGETS.iter().enumerate() {
            let pin = &GOOSE_BINARIES[i];
            assert_eq!(pin.rustc_target, *target);
            assert_eq!(
                goose_binary_for_target(target).map(|p| p.archive_name),
                Some(pin.archive_name)
            );
            assert_eq!(pin.sha256.len(), 64);
            assert!(
                pin.sha256.chars().all(|c| c.is_ascii_hexdigit()),
                "{}",
                pin.sha256
            );
            assert_eq!(pin.sha256, pin.sha256.to_ascii_lowercase());
            assert!(pin.size_bytes > 0);
            let url = goose_release_download_url(pin.archive_name);
            assert!(url.starts_with(GOOSE_RELEASE_DOWNLOAD_PREFIX), "{url}");
            assert!(url.ends_with(pin.archive_name), "{url}");
            assert!(url.contains(GOOSE_RELEASE_TAG), "{url}");
        }
        assert!(goose_binary_for_target("wasm32-unknown-unknown").is_none());
        assert_eq!(goose_exe_name("x86_64-unknown-linux-gnu"), "goose");
        assert_eq!(goose_exe_name("x86_64-pc-windows-msvc"), "goose.exe");
        if let Some(host) = host_rustc_target() {
            assert!(
                goose_binary_for_target(host).is_some(),
                "host {host} must have a Goose pin"
            );
        }
        assert!(
            crate::models::Manifest::v0().asset("goose").is_none(),
            "goose must not be in the default manifest; fetch only when agent: goose"
        );
    }
}
