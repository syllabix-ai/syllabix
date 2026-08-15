//! `syllabix` command-line interface.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use syllabix_core::{Error, Result};

/// Local voice agent.
///
/// Download one binary, then `syllabix run`. No Python, pip, or API key.
#[derive(Debug, Parser)]
#[command(
    name = "syllabix",
    version,
    about = "Local voice agent",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// Top-level commands. Launch CLI surface is `run` and optional `init` only.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Start a local voice conversation (zero config).
    Run,
    /// Scaffold an optional project folder with syllabix.yaml.
    Init {
        /// Directory to create. Defaults to the current directory.
        #[arg(value_name = "DIR")]
        dir: Option<PathBuf>,
    },
}

/// Dispatch a parsed CLI invocation.
pub fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Run => run(),
        Commands::Init { dir } => init(dir),
    }
}

fn run() -> Result<()> {
    tracing::info!("syllabix run is not implemented yet");
    Err(Error::not_implemented("run"))
}

fn init(dir: Option<PathBuf>) -> Result<()> {
    let target = dir
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| ".".to_string());
    tracing::info!(%target, "syllabix init is not implemented yet");
    Err(Error::not_implemented("init"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_run() {
        let cli = Cli::try_parse_from(["syllabix", "run"]).expect("parse run");
        assert!(matches!(cli.command, Commands::Run));
    }

    #[test]
    fn parses_init_without_dir() {
        let cli = Cli::try_parse_from(["syllabix", "init"]).expect("parse init");
        match cli.command {
            Commands::Init { dir } => assert!(dir.is_none()),
            Commands::Run => panic!("expected init"),
        }
    }

    #[test]
    fn parses_init_with_dir() {
        let cli = Cli::try_parse_from(["syllabix", "init", "demo-agent"]).expect("parse init dir");
        match cli.command {
            Commands::Init { dir } => {
                assert_eq!(dir.as_deref(), Some(std::path::Path::new("demo-agent")));
            }
            Commands::Run => panic!("expected init"),
        }
    }

    #[test]
    fn help_lists_run_and_init() {
        let mut command = Cli::command();
        let help = command.render_help().to_string();
        assert!(help.contains("run"), "{help}");
        assert!(help.contains("init"), "{help}");
        assert!(!help.contains("serve"), "{help}");
        assert!(!help.contains("bench"), "{help}");
    }

    #[test]
    fn execute_run_is_not_implemented() {
        let err = execute(Cli {
            command: Commands::Run,
        })
        .expect_err("run should not be implemented");
        assert!(matches!(err, Error::NotImplemented { command: "run" }));
    }

    #[test]
    fn execute_init_is_not_implemented() {
        let err = execute(Cli {
            command: Commands::Init { dir: None },
        })
        .expect_err("init should not be implemented");
        assert!(matches!(err, Error::NotImplemented { command: "init" }));
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        let err =
            Cli::try_parse_from(["syllabix", "serve"]).expect_err("serve is not a v0 command");
        let message = err.to_string();
        assert!(
            message.contains("unrecognized subcommand") || message.contains("unexpected"),
            "{message}"
        );
    }
}
