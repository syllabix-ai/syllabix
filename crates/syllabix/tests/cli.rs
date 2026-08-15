use assert_cmd::cargo::cargo_bin;
use assert_cmd::Command;
use predicates::str;

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
fn run_is_not_implemented() {
    syllabix()
        .arg("run")
        .assert()
        .failure()
        .code(2)
        .stderr(str::contains("run is not implemented yet"));
}

#[test]
fn init_is_not_implemented() {
    syllabix()
        .arg("init")
        .assert()
        .failure()
        .code(2)
        .stderr(str::contains("init is not implemented yet"));
}

#[test]
fn init_with_dir_is_not_implemented() {
    syllabix()
        .arg("init")
        .arg("demo-agent")
        .assert()
        .failure()
        .code(2)
        .stderr(str::contains("init is not implemented yet"));
}

#[test]
fn no_args_fails_with_help() {
    syllabix()
        .assert()
        .failure()
        .stderr(str::contains("Usage:"));
}
