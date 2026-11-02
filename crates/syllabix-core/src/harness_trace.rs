//! Opt-in stderr traces for built-in harness tool loops (hidden from the TUI by default).

use std::sync::OnceLock;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SYLLABIX_HARNESS_TRACE")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"))
    })
}

/// Print a harness debug line to stderr when `SYLLABIX_HARNESS_TRACE=1`.
#[macro_export]
macro_rules! harness_trace_eprintln {
    ($($arg:tt)*) => {{
        if $crate::harness_trace::is_enabled() {
            eprintln!($($arg)*);
        }
    }};
}

pub(crate) fn is_enabled() -> bool {
    enabled()
}
