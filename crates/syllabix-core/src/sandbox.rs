//! Capability sandbox providers for host-owned developer-harness commands.
//!
//! The provider is intentionally separate from the current argv allowlist.  It
//! gives the generic shell and future skill entrypoints one seam for asking the
//! host to enforce a policy, while the allowlist remains the live default until
//! the generic-shell phase.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::policy::{Enforcement, FilesystemMode, NetworkMode};

/// Inputs needed to construct an OS sandbox for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxRequest {
    /// Canonical project root. Workspace-write is confined to this directory.
    pub workspace: PathBuf,
    /// Fresh, private temporary directory owned by this invocation.
    pub temp_dir: PathBuf,
    /// Filesystem capability requested by the invocation.
    pub filesystem: FilesystemMode,
    /// Network policy. Network denial is enforced by the macOS profile.
    pub network: NetworkMode,
}

impl SandboxRequest {
    pub fn new(
        workspace: impl Into<PathBuf>,
        temp_dir: impl Into<PathBuf>,
        filesystem: FilesystemMode,
        network: NetworkMode,
    ) -> Self {
        Self {
            workspace: workspace.into(),
            temp_dir: temp_dir.into(),
            filesystem,
            network,
        }
    }
}

/// A host error from preparing or probing a sandbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxError {
    pub code: &'static str,
    pub message: String,
}

impl SandboxError {
    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            code: "SANDBOX_UNAVAILABLE",
            message: message.into(),
        }
    }

    fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "SANDBOX_INVALID_REQUEST",
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for SandboxError {}

/// One platform implementation of filesystem sandboxing.
pub trait SandboxProvider: Send + Sync {
    /// Run a harmless, functional probe. A successful probe is the only basis
    /// for reporting [`Enforcement::Full`].
    fn probe(&self, request: &SandboxRequest) -> Result<Enforcement, SandboxError>;

    /// Wrap a program and argv in the provider's confinement.
    fn command(
        &self,
        request: &SandboxRequest,
        program: &Path,
        argv: &[String],
    ) -> Result<Command, SandboxError>;
}

/// Provider selected for the current host. Unsupported hosts fail closed.
pub fn current_provider() -> Box<dyn SandboxProvider> {
    #[cfg(target_os = "macos")]
    {
        Box::new(MacOsSandboxProvider)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Box::new(UnavailableSandboxProvider)
    }
}

/// macOS Seatbelt provider. `sandbox-exec` is deprecated by Apple, so every
/// session must probe the runner before claiming enforcement.
#[derive(Debug, Clone, Copy, Default)]
pub struct MacOsSandboxProvider;

/// Explicit fail-closed provider for hosts without a Phase 2 backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableSandboxProvider;

fn validate_request(request: &SandboxRequest) -> Result<(), SandboxError> {
    if !request.workspace.is_absolute() || !request.temp_dir.is_absolute() {
        return Err(SandboxError::invalid(
            "workspace and temp_dir must be absolute paths",
        ));
    }
    if request.workspace == Path::new("/") {
        return Err(SandboxError::invalid(
            "workspace cannot be the filesystem root",
        ));
    }
    if request.filesystem == FilesystemMode::DangerFullAccess {
        return Err(SandboxError::unavailable(
            "danger-full-access has no sandbox provider",
        ));
    }
    if !request.workspace.is_dir() || !request.temp_dir.is_dir() {
        return Err(SandboxError::invalid(
            "workspace and temp_dir must be existing directories",
        ));
    }
    if request.workspace.canonicalize().ok().as_deref() != Some(request.workspace.as_path())
        || request.temp_dir.canonicalize().ok().as_deref() != Some(request.temp_dir.as_path())
    {
        return Err(SandboxError::invalid(
            "workspace and temp_dir must be canonical directories",
        ));
    }
    Ok(())
}

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

#[cfg(target_os = "macos")]
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

        // Probe the actual write promise, not only whether the binary exists.
        let script = format!(
            "touch '{}'",
            marker.to_string_lossy().replace('\'', "'\\''")
        );
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
        let outside = request
            .workspace
            .parent()
            .ok_or_else(|| SandboxError::invalid("workspace has no parent"))?
            .join(".syllabix-sandbox-outside-probe");
        let outside_script = format!(
            "touch '{}'",
            outside.to_string_lossy().replace('\'', "'\\''")
        );
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

impl SandboxProvider for UnavailableSandboxProvider {
    fn probe(&self, _: &SandboxRequest) -> Result<Enforcement, SandboxError> {
        Err(SandboxError::unavailable(
            "no sandbox provider is available on this platform",
        ))
    }

    fn command(&self, _: &SandboxRequest, _: &Path, _: &[String]) -> Result<Command, SandboxError> {
        Err(SandboxError::unavailable(
            "no sandbox provider is available on this platform",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(mode: FilesystemMode) -> SandboxRequest {
        SandboxRequest::new(
            "/private/tmp/syllabix-workspace",
            "/private/tmp/syllabix-temp",
            mode,
            NetworkMode::None,
        )
    }

    #[test]
    fn non_macos_provider_fails_closed() {
        #[cfg(not(target_os = "macos"))]
        assert_eq!(
            current_provider()
                .probe(&request(FilesystemMode::ReadOnly))
                .unwrap_err()
                .code,
            "SANDBOX_UNAVAILABLE"
        );
    }

    #[test]
    fn danger_full_access_is_not_a_sandbox_profile() {
        let err = profile(&request(FilesystemMode::DangerFullAccess)).unwrap_err();
        assert_eq!(err.code, "SANDBOX_UNAVAILABLE");
    }

    #[test]
    fn relative_paths_are_rejected() {
        let mut request = request(FilesystemMode::ReadOnly);
        request.workspace = PathBuf::from("workspace");
        assert_eq!(
            profile(&request).unwrap_err().code,
            "SANDBOX_INVALID_REQUEST"
        );
    }

    #[cfg(target_os = "macos")]
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
