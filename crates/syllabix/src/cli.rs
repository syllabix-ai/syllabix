//! `syllabix` command-line interface.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use syllabix_core::{resolve_turn_debug_dir, run_live, AgentConfig, Cancel, Result, TurnDebug};

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
    Run {
        /// Write per-turn WAVs and a sidecar. Off by default.
        #[arg(long, value_name = "DIR", num_args = 0..=1)]
        turn_debug: Option<Option<PathBuf>>,
    },
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
        Commands::Run { turn_debug } => run(turn_debug),
        Commands::Init { dir } => init(dir),
    }
}

fn open_turn_debug(flag: Option<Option<PathBuf>>) -> Result<Option<TurnDebug>> {
    match flag {
        None => Ok(None),
        Some(path) => {
            let dir = resolve_turn_debug_dir(path);
            Ok(Some(TurnDebug::open(dir)?))
        }
    }
}

fn run(turn_debug: Option<Option<PathBuf>>) -> Result<()> {
    let debug = open_turn_debug(turn_debug)?;
    let config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    let cancel = Cancel::new();
    #[cfg(coverage)]
    {
        let report = run_live(&config, cancel, None, debug)?;
        tracing::info!(turns = report.turns.len(), "conversation ended");
        Ok(())
    }
    #[cfg(not(coverage))]
    {
        if io::stdout().is_terminal() {
            tui::run_conversation_tui(config, cancel, debug)
        } else {
            let report = run_live(&config, cancel, None, debug)?;
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
    use std::time::{SystemTime, UNIX_EPOCH};
    use syllabix_core::{AgentConfig, Error, CONFIG_FILE_NAME};

    fn unique_dir(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn clap_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_run() {
        let cli = Cli::try_parse_from(["syllabix", "run"]).expect("parse run");
        match cli.command {
            Commands::Run { turn_debug } => assert!(turn_debug.is_none()),
            Commands::Init { .. } => panic!("expected run"),
        }
    }

    #[test]
    fn parses_run_turn_debug_without_dir() {
        let cli = Cli::try_parse_from(["syllabix", "run", "--turn-debug"]).expect("parse");
        match cli.command {
            Commands::Run { turn_debug } => assert_eq!(turn_debug, Some(None)),
            Commands::Init { .. } => panic!("expected run"),
        }
    }

    #[test]
    fn parses_run_turn_debug_with_dir() {
        let cli = Cli::try_parse_from(["syllabix", "run", "--turn-debug", "/tmp/turns"])
            .expect("parse dir");
        match cli.command {
            Commands::Run { turn_debug } => {
                assert_eq!(turn_debug, Some(Some(PathBuf::from("/tmp/turns"))));
            }
            Commands::Init { .. } => panic!("expected run"),
        }
    }

    #[test]
    fn parses_init_without_dir() {
        let cli = Cli::try_parse_from(["syllabix", "init"]).expect("parse init");
        match cli.command {
            Commands::Init { dir } => assert!(dir.is_none()),
            Commands::Run { .. } => panic!("expected init"),
        }
    }

    #[test]
    fn parses_init_with_dir() {
        let cli = Cli::try_parse_from(["syllabix", "init", "demo-agent"]).expect("parse init dir");
        match cli.command {
            Commands::Init { dir } => {
                assert_eq!(dir.as_deref(), Some(std::path::Path::new("demo-agent")));
            }
            Commands::Run { .. } => panic!("expected init"),
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
    fn run_help_lists_turn_debug() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("run")
            .expect("run")
            .render_help()
            .to_string();
        assert!(help.contains("--turn-debug"), "{help}");
    }

    #[test]
    fn execute_init_writes_yaml_and_round_trips() {
        let dir = unique_dir("syllabix-cli-init");
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

    #[test]
    fn turn_debug_missing_dir_fails_before_loop() {
        let dir = unique_dir("syllabix-cli-turn-debug-bad");
        std::fs::create_dir_all(&dir).unwrap();
        let blocker = dir.join("file");
        std::fs::write(&blocker, b"x").unwrap();
        let err = execute(Cli {
            command: Commands::Run {
                turn_debug: Some(Some(blocker.join("nested"))),
            },
        })
        .expect_err("unwritable");
        assert!(matches!(err, Error::Config { .. }));
        assert!(err.to_string().contains("cannot write"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(coverage)]
    #[test]
    fn execute_run_uses_fake_loop_under_coverage() {
        execute(Cli {
            command: Commands::Run { turn_debug: None },
        })
        .expect("coverage run");
    }

    #[cfg(coverage)]
    #[test]
    fn execute_run_turn_debug_writes_fixture_under_coverage() {
        let dir = unique_dir("syllabix-cli-turn-debug-cov");
        execute(Cli {
            command: Commands::Run {
                turn_debug: Some(Some(dir.clone())),
            },
        })
        .expect("coverage turn debug");
        assert!(dir.join("turn-000").join("turn.json").is_file());
        std::fs::remove_dir_all(&dir).unwrap();
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
