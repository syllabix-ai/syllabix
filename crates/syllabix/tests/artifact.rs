//! Sidecar and SHA256SUMS written next to a `dist/` executable.

use serde::Deserialize;
use syllabix_core::{
    artifact_name_for_target, artifact_name_for_uname, format_sha256sums_line,
    latest_release_download_url, parse_sha256sums, sha256sums_contains, ARTIFACT_FILE_NAMES,
    DIST_PROFILE, DIST_TARGETS, MAX_DIST_BINARY_BYTES, RELEASE_LATEST_DOWNLOAD_PREFIX,
    SHA256SUMS_FILE_NAME,
};

#[derive(Debug, Deserialize)]
struct ReproSidecar {
    git_sha: String,
    rustc: String,
    cargo: String,
    target: String,
    profile: String,
    artifact: String,
    size_bytes: u64,
    sha256: String,
    cargo_lock_sha256: String,
    source_date_epoch: String,
}

#[test]
fn dist_target_matrix_is_four_launch_oses() {
    assert_eq!(DIST_TARGETS.len(), 4);
    assert_eq!(ARTIFACT_FILE_NAMES.len(), 4);
    assert_eq!(DIST_PROFILE, "dist");
    assert_eq!(
        artifact_name_for_target(DIST_TARGETS[0]).unwrap(),
        ARTIFACT_FILE_NAMES[0]
    );
}

#[test]
fn readme_curl_uname_maps_to_release_asset_names() {
    assert_eq!(
        artifact_name_for_uname("Linux", "x86_64").unwrap(),
        "syllabix-Linux-x86_64"
    );
    let url = latest_release_download_url("syllabix-Linux-x86_64");
    assert_eq!(
        url,
        format!("{RELEASE_LATEST_DOWNLOAD_PREFIX}syllabix-Linux-x86_64")
    );
    assert_eq!(SHA256SUMS_FILE_NAME, "SHA256SUMS");
}

#[test]
fn sha256sums_file_covers_launch_artifacts() {
    let hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    let mut body = String::new();
    for name in ARTIFACT_FILE_NAMES {
        body.push_str(&format_sha256sums_line(hex, name));
        body.push('\n');
    }
    let parsed = parse_sha256sums(&body).expect("four launch artifacts");
    assert_eq!(parsed.len(), 4);
    for name in ARTIFACT_FILE_NAMES {
        assert!(sha256sums_contains(&parsed, name, hex), "{name}");
    }
}

#[test]
fn example_sidecar_fields_round_trip() {
    let json = r#"
        {
          "git_sha": "abc123",
          "rustc": "rustc 1.91.0",
          "cargo": "cargo 1.91.0",
          "target": "x86_64-unknown-linux-gnu",
          "profile": "dist",
          "artifact": "syllabix-Linux-x86_64",
          "size_bytes": 42,
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "cargo_lock_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          "source_date_epoch": "1700000000"
        }
    "#;
    let sidecar: ReproSidecar = serde_yaml::from_str(json).expect("json is valid yaml");
    assert_eq!(sidecar.profile, DIST_PROFILE);
    assert_eq!(
        sidecar.artifact,
        artifact_name_for_target(&sidecar.target).unwrap()
    );
    assert_eq!(sidecar.sha256.len(), 64);
    assert_eq!(sidecar.cargo_lock_sha256.len(), 64);
    assert!(sidecar.size_bytes < MAX_DIST_BINARY_BYTES);
    assert!(!sidecar.git_sha.is_empty());
    assert!(sidecar.rustc.contains("rustc"));
    assert!(sidecar.cargo.contains("cargo"));
    assert!(!sidecar.source_date_epoch.is_empty());
}
