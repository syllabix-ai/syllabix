//! Score Riley calls against the scenario pack (#329 phase A3). Needs
//! `--allow-hosted`; see the crate README. Our own approximation, not
//! comparable with Veris's VAmoS leaderboard.

use clap::Parser;
use syllabix_riley_eval::runner::{execute, Args, VadChoice};

fn main() {
    let args = Args::parse();
    let code = execute(
        &args,
        &|name| std::env::var(name).ok(),
        VadChoice::Silero,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    );
    std::process::exit(code);
}
