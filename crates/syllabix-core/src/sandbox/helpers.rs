use std::path::{Path, PathBuf};

use super::super::{SandboxError, SandboxRequest};
use crate::policy::FilesystemMode;

impl SandboxError {
    pub(super) fn invalid(message: impl Into<String>) -> Self {
        Self {
            code: "SANDBOX_INVALID_REQUEST",
            message: message.into(),
        }
    }
}

pub(crate) fn validate_request(request: &SandboxRequest) -> Result<(), SandboxError> {
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

/// Host path used to prove outside-workspace writes are denied.
pub(crate) fn outside_probe_path(request: &SandboxRequest) -> Result<PathBuf, SandboxError> {
    // Prefer a sibling of the workspace when that parent is not shadowed by a
    // sandbox remount (for example bwrap `--tmpfs /tmp`). `/var/tmp` stays on
    // the real host filesystem for Linux probes.
    #[cfg(target_os = "linux")]
    {
        use std::sync::atomic::{AtomicU64, Ordering};
        static OUTSIDE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = OUTSIDE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let _ = request;
        Ok(PathBuf::from("/var/tmp").join(format!(
            ".syllabix-sandbox-outside-probe-{}-{}",
            std::process::id(),
            n
        )))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(request
            .workspace
            .parent()
            .ok_or_else(|| SandboxError::invalid("workspace has no parent"))?
            .join(".syllabix-sandbox-outside-probe"))
    }
}

pub(crate) fn shell_quote_single(path: &Path) -> String {
    path.to_string_lossy().replace('\'', "'\\''")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::NetworkMode;

    #[test]
    fn validation_and_path_helpers_cover_normal_requests() {
        let root = std::env::temp_dir().join(format!(
            "syllabix-sandbox-validation-{}",
            std::process::id()
        ));
        let workspace = root.join("workspace");
        let temp = root.join("temp");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&temp).unwrap();
        let workspace = workspace.canonicalize().unwrap();
        let temp = temp.canonicalize().unwrap();
        let request = SandboxRequest::new(
            &workspace,
            &temp,
            FilesystemMode::ReadOnly,
            NetworkMode::None,
        );

        assert!(validate_request(&request).is_ok());
        assert!(outside_probe_path(&request).is_ok());
        assert_eq!(shell_quote_single(Path::new("/tmp/a'b")), "/tmp/a'\\''b");
        assert_eq!(
            SandboxError::invalid("bad").to_string(),
            "SANDBOX_INVALID_REQUEST: bad"
        );

        let mut root_request = request.clone();
        root_request.workspace = PathBuf::from("/");
        assert_eq!(
            validate_request(&root_request).unwrap_err().code,
            "SANDBOX_INVALID_REQUEST"
        );

        let mut missing_request = request.clone();
        missing_request.workspace = root.join("missing");
        assert_eq!(
            validate_request(&missing_request).unwrap_err().code,
            "SANDBOX_INVALID_REQUEST"
        );

        let mut relative_request = request.clone();
        relative_request.workspace = PathBuf::from("workspace");
        assert_eq!(
            validate_request(&relative_request).unwrap_err().code,
            "SANDBOX_INVALID_REQUEST"
        );

        let mut noncanonical_request = request.clone();
        noncanonical_request.workspace = root.join("workspace/../workspace");
        assert_eq!(
            validate_request(&noncanonical_request).unwrap_err().code,
            "SANDBOX_INVALID_REQUEST"
        );

        let mut full_access_request = request;
        full_access_request.filesystem = FilesystemMode::DangerFullAccess;
        assert_eq!(
            validate_request(&full_access_request).unwrap_err().code,
            "SANDBOX_UNAVAILABLE"
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
