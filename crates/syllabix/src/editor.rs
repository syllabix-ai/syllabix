//! Open `syllabix.yaml` in a cross-platform text editor.

use std::env;
use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

use syllabix_core::{Error, Result};

/// Open `path` in `$VISUAL`, then `$EDITOR`, then a platform default.
///
/// Blocks until the editor process exits. The TUI must leave raw mode first.
pub fn open_path_in_editor(path: &Path) -> Result<()> {
    let (program, args) = editor_argv()?;
    let program_label = program.to_string_lossy().into_owned();
    let status = Command::new(&program)
        .args(&args)
        .arg(path)
        .status()
        .map_err(|err| Error::Config {
            field: path.display().to_string(),
            message: format!("failed to launch editor `{program_label}`: {err}"),
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(Error::Config {
            field: path.display().to_string(),
            message: format!("editor `{program_label}` exited with {status}"),
        })
    }
}

/// Resolve the editor program and any extra argv from the environment.
fn editor_argv() -> Result<(OsString, Vec<OsString>)> {
    if let Some(value) = env::var_os("VISUAL").filter(|v| !v.is_empty()) {
        return Ok(split_command(value));
    }
    if let Some(value) = env::var_os("EDITOR").filter(|v| !v.is_empty()) {
        return Ok(split_command(value));
    }
    Ok((default_editor(), Vec::new()))
}

fn split_command(value: OsString) -> (OsString, Vec<OsString>) {
    let text = value.to_string_lossy();
    let mut parts = text.split_whitespace();
    let program = parts
        .next()
        .map(OsString::from)
        .unwrap_or_else(default_editor);
    let args = parts.map(OsString::from).collect();
    (program, args)
}

fn default_editor() -> OsString {
    if cfg!(windows) {
        OsString::from("notepad")
    } else {
        OsString::from("nano")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn default_editor_is_platform_native() {
        let editor = default_editor();
        if cfg!(windows) {
            assert_eq!(editor, "notepad");
        } else {
            assert_eq!(editor, "nano");
        }
    }

    #[test]
    fn visual_beats_editor_and_splits_extra_args() {
        let _guard = env_lock().lock().unwrap();
        env::set_var("VISUAL", "code -w");
        env::set_var("EDITOR", "vim");
        let (program, args) = editor_argv().unwrap();
        assert_eq!(program, "code");
        assert_eq!(args, vec![OsString::from("-w")]);
        env::remove_var("VISUAL");
        env::remove_var("EDITOR");
    }

    #[test]
    fn editor_is_used_when_visual_is_unset() {
        let _guard = env_lock().lock().unwrap();
        env::remove_var("VISUAL");
        env::set_var("EDITOR", "vim");
        let (program, args) = editor_argv().unwrap();
        assert_eq!(program, "vim");
        assert!(args.is_empty());
        env::remove_var("EDITOR");
    }
}
