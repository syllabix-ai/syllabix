//! `syllabix-riley-eval` binary (#329 phase A3): refuses to call anything
//! without `--allow-hosted`, even when keys are set.

use std::process::Command;

#[test]
fn refuses_without_opt_in_even_with_keys() {
    let out = std::env::temp_dir().join(format!("riley-eval-bin-{}.jsonl", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_syllabix-riley-eval"))
        .args(["--mode", "text", "--agent-model", "m", "--out"])
        .arg(&out)
        .env("SYLLABIX_LLM_API_KEY", "bin-test-key")
        .env("SYLLABIX_RILEY_DATABASE_URL", "postgres://unused")
        .output()
        .expect("run binary");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--allow-hosted"), "{stderr}");
    assert!(!stderr.contains("bin-test-key"));
    assert!(output.stdout.is_empty());
    assert!(!out.exists());
}

#[test]
fn help_discloses_the_approximation() {
    let output = Command::new(env!("CARGO_BIN_EXE_syllabix-riley-eval"))
        .arg("--help")
        .output()
        .expect("run binary");
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("not comparable with Veris"), "{help}");
    assert!(help.contains("--allow-hosted"), "{help}");
}
