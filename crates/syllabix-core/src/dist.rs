//! Names of installable `dist` profile artifacts (V0_LAUNCH sequence 22).
//!
//! Weights are first-run cached, not packed into these files. GitHub Release
//! publication of the files is sequence 23.

/// Cargo profile that produces the standalone executable.
pub const DIST_PROFILE: &str = "dist";

/// Launch target triples. Index matches [`ARTIFACT_FILE_NAMES`].
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

/// Packed-weight size that would mean the binary swallowed Whisper+Llama+Kokoro.
/// Dist artifacts must stay below this; first-run cache holds the models.
pub const MAX_DIST_BINARY_BYTES: u64 = 400_000_000;

/// File name for a rustc target triple, if it is on the launch matrix.
pub fn artifact_name_for_target(rustc_target: &str) -> Option<&'static str> {
    DIST_TARGETS
        .iter()
        .position(|t| *t == rustc_target)
        .map(|i| ARTIFACT_FILE_NAMES[i])
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
    fn packed_weight_ceiling_is_below_whisper_small() {
        let manifest = crate::models::Manifest::v0();
        let whisper = manifest
            .asset("whisper-small")
            .expect("whisper-small is a v0 asset");
        assert!(MAX_DIST_BINARY_BYTES < whisper.size_bytes);
    }
}
