use assert_cmd::cargo::cargo_bin;
use assert_cmd::Command;
use predicates::str;
use syllabix_core::{AgentConfig, CONFIG_FILE_NAME};

fn syllabix() -> Command {
    Command::new(cargo_bin("syllabix"))
}

#[test]
fn help_exits_zero_and_lists_commands() {
    syllabix()
        .arg("--help")
        .assert()
        .success()
        .stdout(str::contains("Usage:"))
        .stdout(str::contains("run"))
        .stdout(str::contains("init"));
}

#[test]
fn version_exits_zero() {
    syllabix()
        .arg("--version")
        .assert()
        .success()
        .stdout(str::contains("syllabix"));
}

#[test]
fn init_writes_yaml() {
    let dir = std::env::temp_dir().join(format!(
        "syllabix-cli-bin-init-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    syllabix()
        .arg("init")
        .arg(&dir)
        .assert()
        .success()
        .stdout(str::contains("wrote"));
    let yaml = std::fs::read_to_string(dir.join(CONFIG_FILE_NAME)).expect("yaml");
    assert_eq!(AgentConfig::parse_yaml(&yaml).unwrap(), AgentConfig::v0());
    syllabix()
        .arg("init")
        .arg(&dir)
        .assert()
        .failure()
        .code(1)
        .stderr(str::contains("already exists"));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn no_args_fails_with_help() {
    syllabix()
        .assert()
        .failure()
        .stderr(str::contains("Usage:"));
}
