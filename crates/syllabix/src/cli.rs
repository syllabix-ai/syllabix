//! `syllabix` command-line interface.

use clap::{Parser, Subcommand};
use std::path::PathBuf;
#[cfg(coverage)]
use syllabix_core::Error;
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
///
/// Diagnostics (turn timeline + WAVs) are yaml-only since row 34:
/// `diagnostics: {timestamps, audio, directory}` in `syllabix.yaml`.
#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Start a local voice conversation (zero config).
    Run {
        /// Keep VAD running during TTS and cancel playback on user speech.
        #[arg(long)]
        barge_in: bool,
    },
    /// Scaffold an optional project folder with syllabix.yaml.
    Init {
        /// Directory to create. Defaults to the current directory.
        #[arg(value_name = "DIR")]
        dir: Option<PathBuf>,
    },
    /// Run contributor-only, component-level native performance measurements.
    #[command(hide = true)]
    Bench {
        /// Destination JSONL file. The command refuses to overwrite it.
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
    /// Private worker used to isolate one ASR model's host-memory measurement.
    #[command(hide = true)]
    BenchAsrWorker {
        /// Temporary JSONL file written for the parent `bench` process.
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
    /// Private worker used to isolate one TTS model's host-memory measurement.
    #[command(hide = true)]
    BenchTtsWorker {
        /// Temporary JSONL file written for the parent `bench` process.
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
    /// Private worker used to isolate one local LLM's host-memory measurement.
    #[command(hide = true)]
    BenchLlmWorker {
        /// Temporary JSONL file written for the parent `bench` process.
        #[arg(long, value_name = "FILE")]
        out: PathBuf,
    },
}

/// Dispatch a parsed CLI invocation.
pub fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Run { barge_in } => run(barge_in),
        Commands::Init { dir } => init(dir),
        #[cfg(not(coverage))]
        Commands::Bench { out } => crate::bench::run(out),
        #[cfg(not(coverage))]
        Commands::BenchAsrWorker { out } => crate::bench::run_asr_worker(out),
        #[cfg(not(coverage))]
        Commands::BenchTtsWorker { out } => crate::bench::run_tts_worker(out),
        #[cfg(not(coverage))]
        Commands::BenchLlmWorker { out } => crate::bench::run_llm_worker(out),
        // Bench is contributor-only native work. Coverage must neither load
        // benchmark models nor execute its corpus/writer tests.
        #[cfg(coverage)]
        Commands::Bench { .. } => Err(Error::not_implemented("bench")),
        #[cfg(coverage)]
        Commands::BenchAsrWorker { .. } => Err(Error::not_implemented("bench")),
        #[cfg(coverage)]
        Commands::BenchTtsWorker { .. } => Err(Error::not_implemented("bench")),
        #[cfg(coverage)]
        Commands::BenchLlmWorker { .. } => Err(Error::not_implemented("bench")),
    }
}

fn run(barge_in: bool) -> Result<()> {
    let config = AgentConfig::resolve_for_run(std::env::current_dir()?.as_path())?;
    let cancel = Cancel::new();
    #[cfg(coverage)]
    {
        let report = run_live(&config, cancel, None, barge_in)?;
        tracing::info!(turns = report.turns.len(), "conversation ended");
        Ok(())
    }
    #[cfg(not(coverage))]
    {
        if io::stdout().is_terminal() {
            tui::run_conversation_tui(config, cancel, barge_in)
        } else {
            let report = run_live(&config, cancel, None, barge_in)?;
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
            Commands::Run { barge_in } => assert!(!barge_in),
            Commands::Init { .. } => panic!("expected run"),
            Commands::Bench { .. } => panic!("expected run"),
            Commands::BenchAsrWorker { .. } => panic!("expected run"),
            Commands::BenchTtsWorker { .. } => panic!("expected run"),
            Commands::BenchLlmWorker { .. } => panic!("expected run"),
        }
    }

    #[test]
    fn turn_debug_flag_is_gone() {
        // Row 34: diagnostics are yaml-only; the flag must be rejected.
        for args in [
            vec!["syllabix", "run", "--turn-debug"],
            vec!["syllabix", "run", "--turn-debug", "/tmp/turns"],
        ] {
            assert!(
                Cli::try_parse_from(args.clone()).is_err(),
                "{args:?} must not parse"
            );
        }
    }

    #[test]
    fn parses_init_without_dir() {
        let cli = Cli::try_parse_from(["syllabix", "init"]).expect("parse init");
        match cli.command {
            Commands::Init { dir } => assert!(dir.is_none()),
            Commands::Run { .. } => panic!("expected init"),
            Commands::Bench { .. } => panic!("expected init"),
            Commands::BenchAsrWorker { .. } => panic!("expected init"),
            Commands::BenchTtsWorker { .. } => panic!("expected init"),
            Commands::BenchLlmWorker { .. } => panic!("expected init"),
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
            Commands::Bench { .. } => panic!("expected init"),
            Commands::BenchAsrWorker { .. } => panic!("expected init"),
            Commands::BenchTtsWorker { .. } => panic!("expected init"),
            Commands::BenchLlmWorker { .. } => panic!("expected init"),
        }
    }

    #[test]
    fn parses_hidden_bench_with_an_output_file() {
        let cli =
            Cli::try_parse_from(["syllabix", "bench", "--out", "run.jsonl"]).expect("parse bench");
        match cli.command {
            Commands::Bench { out } => assert_eq!(out, PathBuf::from("run.jsonl")),
            _ => panic!("expected bench"),
        }
    }

    #[test]
    fn parses_private_asr_benchmark_worker() {
        let cli = Cli::try_parse_from(["syllabix", "bench-asr-worker", "--out", "worker.jsonl"])
            .expect("parse worker");
        match cli.command {
            Commands::BenchAsrWorker { out } => assert_eq!(out, PathBuf::from("worker.jsonl")),
            _ => panic!("expected ASR worker"),
        }
    }

    #[test]
    fn parses_private_tts_benchmark_worker() {
        let cli = Cli::try_parse_from(["syllabix", "bench-tts-worker", "--out", "worker.jsonl"])
            .expect("parse worker");
        match cli.command {
            Commands::BenchTtsWorker { out } => assert_eq!(out, PathBuf::from("worker.jsonl")),
            _ => panic!("expected TTS worker"),
        }
    }

    #[test]
    fn parses_private_llm_benchmark_worker() {
        let cli = Cli::try_parse_from(["syllabix", "bench-llm-worker", "--out", "worker.jsonl"])
            .expect("parse worker");
        match cli.command {
            Commands::BenchLlmWorker { out } => assert_eq!(out, PathBuf::from("worker.jsonl")),
            _ => panic!("expected LLM worker"),
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
    fn run_help_lists_barge_in_only() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("run")
            .expect("run")
            .render_help()
            .to_string();
        assert!(help.contains("--barge-in"), "{help}");
        assert!(!help.contains("--turn-debug"), "{help}");
    }

    #[test]
    fn parses_run_barge_in() {
        let cli = Cli::try_parse_from(["syllabix", "run", "--barge-in"]).expect("parse");
        match cli.command {
            Commands::Run { barge_in } => assert!(barge_in),
            Commands::Init { .. } => panic!("expected run"),
            Commands::Bench { .. } => panic!("expected run"),
            Commands::BenchAsrWorker { .. } => panic!("expected run"),
            Commands::BenchTtsWorker { .. } => panic!("expected run"),
            Commands::BenchLlmWorker { .. } => panic!("expected run"),
        }
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

    #[cfg(coverage)]
    #[test]
    fn execute_run_uses_fake_loop_under_coverage() {
        execute(Cli {
            command: Commands::Run { barge_in: false },
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
