#[cfg(not(coverage))]
mod bench;
mod cli;
mod tui;

use clap::Parser;
use syllabix_core::Error;
use tracing_subscriber::EnvFilter;

fn main() {
    let _linked = syllabix_core::ensure_shared_ggml_frontends();
    let cli = cli::Cli::parse();
    let default_filter = match cli.command {
        cli::Commands::Run { .. } => "warn",
        cli::Commands::Init { .. } => "info",
        cli::Commands::Bench { .. } => "info",
        cli::Commands::BenchAsrWorker { .. } => "info",
        cli::Commands::BenchTtsWorker { .. } => "info",
    };
    init_tracing(default_filter);

    if let Err(err) = cli::execute(cli) {
        if !matches!(err, Error::Cancelled) {
            tracing::error!("{err}");
            eprintln!("error: {err}");
        }
        std::process::exit(err.exit_code());
    }
}

fn init_tracing(default_filter: &str) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}
