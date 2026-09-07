#[cfg(not(coverage))]
mod bench;
mod cli;
pub mod tui;

use clap::Parser;
use syllabix_core::Error;
use tracing_subscriber::EnvFilter;

fn main() {
    std::process::exit(run_cli());
}

/// Parse, configure tracing, and dispatch. Separated from [`main`] so unit
/// tests can cover argument handling without exiting the test process.
fn run_cli() -> i32 {
    let _linked = syllabix_core::ensure_shared_ggml_frontends();
    let cli = cli::Cli::parse();
    run_parsed(cli)
}

fn run_parsed(cli: cli::Cli) -> i32 {
    init_tracing(default_filter_for(&cli.command));
    if let Err(err) = cli::execute(cli) {
        if !matches!(err, Error::Cancelled) {
            tracing::error!("{err}");
            eprintln!("error: {err}");
        }
        return err.exit_code();
    }
    0
}

/// Tracing default per subcommand: chatty for setup, quiet for conversation.
fn default_filter_for(command: &cli::Commands) -> &'static str {
    match command {
        cli::Commands::Run { .. } => "warn",
        cli::Commands::Init { .. } => "info",
        cli::Commands::Bench { .. } => "info",
        cli::Commands::BenchAsrWorker { .. } => "info",
        cli::Commands::BenchTtsWorker { .. } => "info",
        cli::Commands::BenchLlmWorker { .. } => "info",
    }
}

fn init_tracing(default_filter: &str) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_filter));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn run_is_quiet_and_setup_is_info() {
        assert_eq!(
            default_filter_for(&cli::Commands::Run { barge_in: false }),
            "warn"
        );
        assert_eq!(
            default_filter_for(&cli::Commands::Run { barge_in: true }),
            "warn"
        );
        assert_eq!(
            default_filter_for(&cli::Commands::Init { dir: None }),
            "info"
        );
        assert_eq!(
            default_filter_for(&cli::Commands::Bench {
                out: PathBuf::from("run.jsonl")
            }),
            "info"
        );
        assert_eq!(
            default_filter_for(&cli::Commands::BenchAsrWorker {
                out: PathBuf::from("worker.jsonl"),
                model: String::from("whisper-small"),
            }),
            "info"
        );
        assert_eq!(
            default_filter_for(&cli::Commands::BenchTtsWorker {
                out: PathBuf::from("worker.jsonl"),
                model: String::from("pocket-tts"),
            }),
            "info"
        );
        assert_eq!(
            default_filter_for(&cli::Commands::BenchLlmWorker {
                out: PathBuf::from("worker.jsonl"),
                model: String::from("lfm2.5-2.6b"),
            }),
            "info"
        );
    }

    #[test]
    fn init_tracing_is_idempotent() {
        init_tracing("info");
        init_tracing("warn");
    }

    #[test]
    fn run_parsed_init_reports_success_or_existing_file() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-main-init-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let code = run_parsed(cli::Cli {
            command: cli::Commands::Init { dir: Some(dir) },
        });
        assert_eq!(code, 0);
    }

    #[cfg(coverage)]
    #[test]
    fn run_parsed_rejects_bench_under_coverage() {
        // Under the coverage build bench workers are stubbed to
        // `not_implemented`, so this exercises the error path (non-zero
        // exit) without touching native weights.
        let code = run_parsed(cli::Cli {
            command: cli::Commands::BenchLlmWorker {
                out: PathBuf::from("missing.jsonl"),
                model: String::from("definitely-not-a-model"),
            },
        });
        assert_eq!(code, 2);
    }

    #[test]
    fn cli_parses_in_binary_context() {
        let cli = cli::Cli::try_parse_from(["syllabix", "run"]).expect("parse run");
        assert!(matches!(cli.command, cli::Commands::Run { .. }));
    }
}
