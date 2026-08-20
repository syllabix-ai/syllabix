//! Sidecar written next to a `dist/` executable by `scripts/package-release.sh`.

use serde::Deserialize;
use syllabix_core::{
    artifact_name_for_target, ARTIFACT_FILE_NAMES, DIST_PROFILE, DIST_TARGETS,
    MAX_DIST_BINARY_BYTES,
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
