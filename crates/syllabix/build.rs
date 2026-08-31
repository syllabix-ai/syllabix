use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=SYLLABIX_GIT_SHA");
    let sha = std::env::var("SYLLABIX_GIT_SHA")
        .ok()
        .or_else(|| {
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|sha| sha.trim().to_owned())
        })
        .filter(|sha| !sha.is_empty())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=SYLLABIX_GIT_SHA={sha}");
}
