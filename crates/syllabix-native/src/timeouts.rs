//! Timeouts for OpenAI-compatible HTTP speech and LLM clients.

use std::time::Duration;

/// Hard connect timeout. A dead endpoint fails the turn instead of hanging.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Socket read granularity inside the stream worker. Also bounds how long a
/// cancelled connection lingers before the worker observes the abandon flag.
pub const READ_POLL_TIMEOUT: Duration = Duration::from_secs(2);

/// No bytes for this long aborts the turn as a provider failure.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(20);

/// Timeouts, injectable so tests can exercise failure paths in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenAiTimeouts {
    /// TCP + TLS connect budget.
    pub connect: Duration,
    /// Socket read granularity inside the stream worker.
    pub read_poll: Duration,
    /// Silence budget between bytes before the turn fails.
    pub idle: Duration,
}

impl Default for OpenAiTimeouts {
    fn default() -> Self {
        Self {
            connect: CONNECT_TIMEOUT,
            read_poll: READ_POLL_TIMEOUT,
            idle: IDLE_TIMEOUT,
        }
    }
}
