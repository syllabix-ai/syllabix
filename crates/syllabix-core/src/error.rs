//! Recoverable Syllabix failures with stable, user-facing messages.

/// Process-level result used by the CLI and later runtime crates.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Recoverable Syllabix failures with stable, user-facing messages.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A command exists in the CLI but is not wired to a working implementation yet.
    #[error("{command} is not implemented yet")]
    NotImplemented {
        /// Subcommand name as the user typed it (`run`, `init`).
        command: &'static str,
    },

    /// Filesystem or stdio failure.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Convenience constructor for a not-yet-implemented CLI command.
    pub fn not_implemented(command: &'static str) -> Self {
        Self::NotImplemented { command }
    }

    /// Process exit code for this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NotImplemented { .. } => 2,
            Self::Io(_) => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn not_implemented_message_names_the_command() {
        let err = Error::not_implemented("run");
        assert_eq!(err.to_string(), "run is not implemented yet");
        assert_eq!(err.exit_code(), 2);
    }

    #[test]
    fn init_not_implemented_message_names_the_command() {
        let err = Error::not_implemented("init");
        assert_eq!(err.to_string(), "init is not implemented yet");
    }

    #[test]
    fn io_errors_wrap_transparently() {
        let err = Error::from(io::Error::new(io::ErrorKind::NotFound, "missing"));
        assert_eq!(err.to_string(), "missing");
        assert_eq!(err.exit_code(), 1);
    }
}
