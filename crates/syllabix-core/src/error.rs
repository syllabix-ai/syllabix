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

    /// Audio frame does not match the v0 PCM contract.
    #[error("invalid audio: {message}")]
    InvalidAudio {
        /// Human-readable reason.
        message: String,
    },

    /// A provider returned a recoverable failure.
    #[error("{provider} failed: {message}")]
    Provider {
        /// Provider label (`silero`, `whisper.cpp`, …).
        provider: &'static str,
        /// Human-readable reason.
        message: String,
    },

    /// Shutdown or generation cancel interrupted in-flight work.
    #[error("conversation cancelled")]
    Cancelled,

    /// A pipeline stage exited while another stage still needed it.
    #[error("pipeline stage {stage} disconnected")]
    Disconnected {
        /// Stage name (`vad`, `stt`, `llm`, `tts`, `sink`, `frames`).
        stage: &'static str,
    },

    /// A worker thread panicked.
    #[error("pipeline worker panicked: {message}")]
    WorkerPanic {
        /// Panic payload, stringified.
        message: String,
    },

    /// Microphone or speaker is missing, busy, or refused by the OS.
    #[error("{message}")]
    AudioDevice {
        /// Actionable, user-facing explanation (what failed and what to try).
        message: String,
    },

    /// Model download, checksum, or cache lookup failed.
    #[error("model cache: {message}")]
    ModelCache {
        /// Human-readable reason.
        message: String,
    },

    /// `syllabix.yaml` failed to parse or validate.
    #[error("{field}: {message}")]
    Config {
        /// Field path (`pipeline.stt.language`) or file path.
        field: String,
        /// Human-readable reason.
        message: String,
    },
}

impl Error {
    /// Convenience constructor for a not-yet-implemented CLI command.
    pub fn not_implemented(command: &'static str) -> Self {
        Self::NotImplemented { command }
    }

    /// Provider failures skip the current turn and keep the loop running.
    ///
    /// Invalid audio, disconnects, panics, cache errors, and device errors
    /// still shut the conversation down.
    pub fn is_turn_recoverable(&self) -> bool {
        matches!(self, Self::Provider { .. })
    }

    /// Process exit code for this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NotImplemented { .. } => 2,
            Self::Io(_)
            | Self::InvalidAudio { .. }
            | Self::Provider { .. }
            | Self::Cancelled
            | Self::Disconnected { .. }
            | Self::WorkerPanic { .. }
            | Self::AudioDevice { .. }
            | Self::ModelCache { .. }
            | Self::Config { .. } => 1,
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

    #[test]
    fn cancelled_and_disconnected_are_process_failures() {
        assert_eq!(Error::Cancelled.to_string(), "conversation cancelled");
        assert_eq!(Error::Cancelled.exit_code(), 1);
        let err = Error::Disconnected { stage: "tts" };
        assert_eq!(err.to_string(), "pipeline stage tts disconnected");
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn audio_device_errors_are_the_message() {
        let err = Error::AudioDevice {
            message: "No microphone found. Connect a mic and allow microphone access.".into(),
        };
        assert!(err.to_string().contains("No microphone found"));
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn model_cache_errors_name_the_layer() {
        let err = Error::ModelCache {
            message: "checksum mismatch for silero".into(),
        };
        assert_eq!(err.to_string(), "model cache: checksum mismatch for silero");
        assert_eq!(err.exit_code(), 1);
    }

    #[test]
    fn provider_errors_are_turn_recoverable() {
        let err = Error::Provider {
            provider: "whisper.cpp",
            message: "decode failed".into(),
        };
        assert!(err.is_turn_recoverable());
        assert!(!Error::Cancelled.is_turn_recoverable());
        assert!(!Error::InvalidAudio {
            message: "bad frame".into(),
        }
        .is_turn_recoverable());
        assert!(!Error::ModelCache {
            message: "missing".into(),
        }
        .is_turn_recoverable());
        assert!(!Error::Config {
            field: "pipeline.stt.language".into(),
            message: "missing field".into(),
        }
        .is_turn_recoverable());
    }

    #[test]
    fn config_errors_name_the_field() {
        let err = Error::Config {
            field: "pipeline.stt.language".into(),
            message: "missing field".into(),
        };
        assert_eq!(err.to_string(), "pipeline.stt.language: missing field");
        assert_eq!(err.exit_code(), 1);
    }
}
