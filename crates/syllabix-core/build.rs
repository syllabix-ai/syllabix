//! Bake the checkout's short git SHA into the binary for ledger fingerprints.
//!
//! Release tarballs and out-of-tree builds fall back to `"unknown"`; the
//! crate version still identifies published artifacts.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=SYLLABIX_GIT_SHA");
    if let Some(root) = repository_root() {
        emit_git_rerun_paths(&root);
    }
    let sha = std::env::var("SYLLABIX_GIT_SHA").ok().or_else(read_git_sha);
    if let Some(sha) = sha {
        println!("cargo:rustc-env=SYLLABIX_GIT_SHA={sha}");
    }
}

/// Re-run the build script when the current branch advances. Watching only
/// `.git/HEAD` is insufficient because its contents stay `ref: ...` while the
/// referenced branch file changes on fetch/merge/commit.
fn emit_git_rerun_paths(root: &Path) {
    let Some(git_dir) = git_directory(root) else {
        return;
    };
    let head = git_dir.join("HEAD");
    println!("cargo:rerun-if-changed={}", head.display());
    println!(
        "cargo:rerun-if-changed={}",
        git_dir.join("packed-refs").display()
    );
    if let Ok(text) = fs::read_to_string(&head) {
        if let Some(reference) = text.trim().strip_prefix("ref: ") {
            println!(
                "cargo:rerun-if-changed={}",
                git_dir.join(reference).display()
            );
        }
    }
}

fn git_directory(root: &Path) -> Option<PathBuf> {
    let dot_git = root.join(".git");
    if dot_git.is_dir() {
        return Some(dot_git);
    }
    let text = fs::read_to_string(dot_git).ok()?;
    let path = text.trim().strip_prefix("gitdir: ")?;
    let path = PathBuf::from(path);
    Some(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

fn repository_root() -> Option<PathBuf> {
    let mut dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    loop {
        if dir.join(".git").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Walk up from this crate to `.git` (workspace root or above), then ask git
/// for a short SHA. Missing git or a shallow/foreign checkout yields `None`.
fn read_git_sha() -> Option<String> {
    let dir = repository_root()?;
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
