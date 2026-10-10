//! Lane 2 — embed the live loop in-process.
//!
//! Copy-paste starting point for other repos: resolve `syllabix.yaml` (or the
//! built-in stack when the file is missing), subscribe to `LoopEvent`, and run
//! the shipped conversation loop. Uses only the supported crate-root surface
//! listed in `docs/embed.md`.
//!
//! Host requirements: CMake plus a C++ compiler (vendored ggml), Linux ALSA
//! headers (`libasound2-dev`), and a shared `SYLLABIX_CACHE_DIR`. Ctrl-C stops
//! the process; the first `run` fetches about 2.2 GB into the cache.

use std::sync::mpsc::channel;

use syllabix_core::{run_live, AgentConfig, Cancel};

fn main() -> syllabix_core::Result<()> {
    let config = AgentConfig::resolve_for_run(&std::env::current_dir()?)?;
    let cancel = Cancel::new();
    let (events_tx, events_rx) = channel();
    let _printer = std::thread::spawn(move || {
        while let Ok(event) = events_rx.recv() {
            println!("event: {event:?}");
        }
    });
    let report = run_live(&config, cancel, Some(events_tx), false)?;
    println!(
        "conversation ended: {} turn(s), {} skipped",
        report.turns.len(),
        report.skipped_turns
    );
    Ok(())
}
