mod cli;

use clap::Parser;
use tracing_subscriber::EnvFilter;

fn main() {
    let cli = cli::Cli::parse();
    init_tracing();

    if let Err(err) = cli::execute(cli) {
        tracing::error!("{err}");
        eprintln!("error: {err}");
        std::process::exit(err.exit_code());
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}
