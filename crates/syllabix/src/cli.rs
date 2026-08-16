//! `syllabix` command-line interface.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use syllabix_core::{run_live, AgentConfig, Cancel, Result};

#[cfg(not(coverage))]
use crate::tui;
#[cfg(not(coverage))]
use std::io::{self, IsTerminal};

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
    let config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    let cancel = Cancel::new();
    #[cfg(coverage)]
    {
        let report = run_live(&config, cancel, None)?;
        tracing::info!(turns = report.turns.len(), "conversation ended");
        Ok(())
    }
    #[cfg(not(coverage))]
    {
        if io::stdout().is_terminal() {
            tui::run_conversation_tui(config, cancel)
        } else {
            let report = run_live(&config, cancel, None)?;
            tracing::info!(turns = report.turns.len(), "conversation ended");
            Ok(())
        }
    }
}

fn init(dir: Option<PathBuf>) -> Result<()> {
    let target = dir.unwrap_or_else(|| PathBuf::from("."));
    let path = AgentConfig::write_init(&target)?;
    println!("wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use syllabix_core::{AgentConfig, Error, CONFIG_FILE_NAME};

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
    fn execute_init_writes_yaml_and_round_trips() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-cli-init-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        execute(Cli {
            command: Commands::Init {
                dir: Some(dir.clone()),
            },
        })
        .expect("init");
        let yaml = dir.join(CONFIG_FILE_NAME);
        assert!(yaml.is_file());
        let loaded = AgentConfig::load_path(&yaml).expect("load");
        assert_eq!(loaded, AgentConfig::v0());
        let err = execute(Cli {
            command: Commands::Init {
                dir: Some(dir.clone()),
            },
        })
        .expect_err("overwrite");
        assert!(matches!(err, Error::Config { .. }));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(coverage)]
    #[test]
    fn execute_run_uses_fake_loop_under_coverage() {
        execute(Cli {
            command: Commands::Run,
        })
        .expect("coverage run");
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
