//! Bake the checkout's short git SHA into the binary for ledger fingerprints.
//!
//! Release tarballs and out-of-tree builds fall back to `"unknown"`; the
//! crate version still identifies published artifacts.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=SYLLABIX_GIT_SHA");
    let sha = std::env::var("SYLLABIX_GIT_SHA").ok().or_else(read_git_sha);
    if let Some(sha) = sha {
        println!("cargo:rustc-env=SYLLABIX_GIT_SHA={sha}");
    }
}

/// Walk up from this crate to `.git` (workspace root or above), then ask git
/// for a short SHA. Missing git or a shallow/foreign checkout yields `None`.
fn read_git_sha() -> Option<String> {
    let mut dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    loop {
        dir.push(".git");
        let found = dir.exists();
        dir.pop();
        if found {
            break;
        }
        if !dir.pop() {
            return None;
        }
    }
    let output = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(&dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let trimmed = text.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}
