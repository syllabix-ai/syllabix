//! macOS Seatbelt sandbox provider.

use std::path::Path;
use std::process::Command;

use crate::policy::{Enforcement, FilesystemMode, NetworkMode};

#[path = "helpers.rs"]
mod helpers;

use super::{SandboxError, SandboxProvider, SandboxRequest};
use helpers::{outside_probe_path, shell_quote_single, validate_request};

/// macOS Seatbelt provider. `sandbox-exec` is deprecated by Apple, so every
/// session must probe the runner before claiming enforcement.
#[derive(Debug, Clone, Copy, Default)]
pub struct MacOsSandboxProvider;

fn sbpl_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

fn profile(request: &SandboxRequest) -> Result<String, SandboxError> {
    validate_request(request)?;
    let workspace = sbpl_path(&request.workspace);
    let temp = sbpl_path(&request.temp_dir);
    let network = match request.network {
        NetworkMode::None => "(deny network*)",
        NetworkMode::Allow => "",
    };
    let writes = match request.filesystem {
        FilesystemMode::ReadOnly => String::new(),
        FilesystemMode::WorkspaceWrite => {
            format!("(allow file-write* (subpath \"{workspace}\") (subpath \"{temp}\"))")
        }
        FilesystemMode::DangerFullAccess => unreachable!("validated above"),
    };
    Ok(format!(
        "(version 1) (allow default) (deny file-write*) {writes} (allow file-write* (literal \"/dev/null\")) {network}"
    ))
}

impl SandboxProvider for MacOsSandboxProvider {
    fn probe(&self, request: &SandboxRequest) -> Result<Enforcement, SandboxError> {
        let profile = profile(request)?;
        std::fs::create_dir_all(&request.temp_dir)
            .map_err(|e| SandboxError::unavailable(format!("probe temp dir: {e}")))?;
        let marker = request.temp_dir.join(".syllabix-sandbox-probe");
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command.args(["-p", &profile, "/usr/bin/true"]);
        let status = command
            .status()
            .map_err(|e| SandboxError::unavailable(format!("sandbox-exec probe: {e}")))?;
        if !status.success() || !Path::new("/usr/bin/sandbox-exec").exists() {
            return Err(SandboxError::unavailable("sandbox-exec probe failed"));
        }

        let script = format!("touch '{}'", shell_quote_single(&marker));
        let mut write = Command::new("/usr/bin/sandbox-exec");
        write.args(["-p", &profile, "/bin/sh", "-c", &script]);
        let write_status = write
            .status()
            .map_err(|e| SandboxError::unavailable(format!("sandbox write probe: {e}")))?;
        let created = marker.exists();
        let _ = std::fs::remove_file(&marker);
        let expected = request.filesystem == FilesystemMode::WorkspaceWrite;
        if created != expected || (expected && !write_status.success()) {
            return Err(SandboxError::unavailable(
                "sandbox write probe did not match the requested filesystem mode",
            ));
        }
        let outside = outside_probe_path(request)?;
        let outside_script = format!("touch '{}'", shell_quote_single(&outside));
        let mut outside_write = Command::new("/usr/bin/sandbox-exec");
        outside_write.args(["-p", &profile, "/bin/sh", "-c", &outside_script]);
        let _ = outside_write
            .status()
            .map_err(|e| SandboxError::unavailable(format!("sandbox outside-write probe: {e}")))?;
        let outside_created = outside.exists();
        let _ = std::fs::remove_file(&outside);
        if outside_created {
            return Err(SandboxError::unavailable(
                "sandbox outside-workspace write probe was not denied",
            ));
        }
        Ok(Enforcement::Full)
    }

    fn command(
        &self,
        request: &SandboxRequest,
        program: &Path,
        argv: &[String],
    ) -> Result<Command, SandboxError> {
        let profile = profile(request)?;
        let mut command = Command::new("/usr/bin/sandbox-exec");
        command.arg("-p").arg(profile).arg(program);
        command.args(argv);
        Ok(command)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::FilesystemMode;

    #[test]
    #[ignore = "requires a functional Seatbelt runner; managed hosts may deny sandbox_apply"]
    fn macos_probe_reports_full_for_both_supported_modes() {
        let root =
            std::env::temp_dir().join(format!("syllabix-sandbox-test-{}", std::process::id()));
        let workspace = root.join("workspace");
        let temp = root.join("temp");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&temp).unwrap();
        let workspace = workspace.canonicalize().unwrap();
        let temp = temp.canonicalize().unwrap();
        let provider = MacOsSandboxProvider;
        for mode in [FilesystemMode::ReadOnly, FilesystemMode::WorkspaceWrite] {
            let request = SandboxRequest::new(&workspace, &temp, mode, NetworkMode::None);
            assert_eq!(provider.probe(&request).unwrap(), Enforcement::Full);
        }
        let _ = std::fs::remove_dir_all(root);
    }
}
