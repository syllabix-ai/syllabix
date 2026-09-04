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
}
