//! Sandboxed shell environment policy (inherit launcher env vs fresh session).

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

static INHERIT_CURRENT_SESSION: AtomicBool = AtomicBool::new(true);

/// Install whether sandboxed shell commands inherit the launcher environment.
pub fn set_inherit_current_session(inherit: bool) {
    INHERIT_CURRENT_SESSION.store(inherit, Ordering::Relaxed);
}

/// Apply [`AgentConfig::inherit_current_session`] for every live-loop entry point.
pub fn install_from_config(config: &crate::config::AgentConfig) {
    set_inherit_current_session(config.inherit_current_session);
}

pub(crate) fn inherit_current_session() -> bool {
    INHERIT_CURRENT_SESSION.load(Ordering::Relaxed)
}

/// Apply the configured shell environment policy to a sandboxed child command.
pub(crate) fn apply_sandbox_shell_env(command: &mut Command) {
    if inherit_current_session() {
        return;
    }
    command
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_PAGER", "cat")
        .env("GIT_EXTERNAL_DIFF", "")
        .env("GIT_TERMINAL_PROMPT", "0");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    struct InheritGuard(bool);

    impl InheritGuard {
        fn set(inherit: bool) -> Self {
            let previous = inherit_current_session();
            set_inherit_current_session(inherit);
            Self(previous)
        }
    }

    impl Drop for InheritGuard {
        fn drop(&mut self) {
            set_inherit_current_session(self.0);
        }
    }

    #[test]
    fn fresh_session_clears_path_and_git_env() {
        let _guard = InheritGuard::set(false);
        let mut command = Command::new("/bin/true");
        command.env("SYLLABIX_LLM_API_KEY", "secret");
        apply_sandbox_shell_env(&mut command);
        let envs = command.get_envs().collect::<Vec<_>>();
        assert!(!envs.iter().any(|(key, _)| *key == "SYLLABIX_LLM_API_KEY"));
        assert!(envs.iter().any(|(key, _)| *key == "PATH"));
    }

    #[test]
    fn inherit_session_keeps_launcher_variables() {
        let _guard = InheritGuard::set(true);
        let mut command = Command::new("/bin/true");
        command.env("CUSTOM_SKILL_KEY", "present");
        apply_sandbox_shell_env(&mut command);
        assert!(command
            .get_envs()
            .any(|(key, _)| key == std::ffi::OsStr::new("CUSTOM_SKILL_KEY")));
    }
}
