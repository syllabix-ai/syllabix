//! `syllabix` command-line interface.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use syllabix_core::{run_live, AgentConfig, Cancel, Result};

use crate::bench;
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

/// Top-level commands. The stranger surface is `run` plus optional `init`.
///
/// Diagnostics (turn timeline + WAVs) are yaml-only since row 34:
/// `diagnostics: {timestamps, audio, directory}` in `syllabix.yaml`.
/// `bench` is a contributor command (issue #45) hidden from this help.
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
    /// Contributor performance harness: component fixtures → JSONL ledger.
    #[command(hide = true)]
    Bench {
        /// Component to measure (default: all independent components).
        #[arg(value_enum, default_value_t = BenchComponent::All)]
        component: BenchComponent,
        /// Output JSONL path (default: docs/eval/runs/<profile>-<os>-<arch>-<sha>.jsonl).
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
        /// STT yaml id (default: small).
        #[arg(long, value_name = "ID")]
        stt: Option<String>,
        /// LLM yaml id (default: llama-3.2-1b).
        #[arg(long, value_name = "ID")]
        llm: Option<String>,
        /// TTS yaml id (default: kokoro).
        #[arg(long, value_name = "ID")]
        tts: Option<String>,
    },
}

/// Dispatch a parsed CLI invocation.
pub fn execute(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Run { barge_in } => run(barge_in),
        Commands::Init { dir } => init(dir),
        Commands::Bench {
            component,
            out,
            stt,
            llm,
            tts,
        } => bench_command(component, out, stt, llm, tts),
    }
}

fn bench_command(
    component: BenchComponent,
    out: Option<PathBuf>,
    stt: Option<String>,
    llm: Option<String>,
    tts: Option<String>,
) -> Result<()> {
    bench::run(bench::BenchArgs {
        component: component.into(),
        out,
        stt,
        llm,
        tts,
    })
}

/// Independently measurable contributors to the local voice stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BenchComponent {
    All,
    Stt,
    Llm,
    Tts,
    TtsAsr,
}

impl From<BenchComponent> for bench::Component {
    fn from(value: BenchComponent) -> Self {
        match value {
            BenchComponent::All => bench::Component::All,
            BenchComponent::Stt => bench::Component::Stt,
            BenchComponent::Llm => bench::Component::Llm,
            BenchComponent::Tts => bench::Component::Tts,
            BenchComponent::TtsAsr => bench::Component::TtsAsr,
        }
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
        }
    }

    #[test]
    fn help_lists_run_and_init() {
        let mut command = Cli::command();
        let help = command.render_help().to_string();
        assert!(help.contains("run"), "{help}");
        assert!(help.contains("init"), "{help}");
        assert!(!help.contains("serve"), "{help}");
        // `bench` is contributor-only (issue #45): hidden from stranger help.
        assert!(!help.contains("bench"), "{help}");
    }

    #[test]
    fn bench_is_hidden_but_parseable() {
        let cli = Cli::try_parse_from(["syllabix", "bench"]).expect("parse bare bench");
        match cli.command {
            Commands::Bench {
                component,
                out,
                stt,
                llm,
                tts,
            } => {
                assert_eq!(component, BenchComponent::All);
                assert!(out.is_none() && stt.is_none() && llm.is_none() && tts.is_none());
            }
            _ => panic!("expected bench"),
        }
        let cli = Cli::try_parse_from([
            "syllabix",
            "bench",
            "--out",
            "/tmp/run.jsonl",
            "--llm",
            "qwen3.5-0.8b",
        ])
        .expect("parse full bench");
        match cli.command {
            Commands::Bench {
                component,
                out,
                stt,
                llm,
                tts,
            } => {
                assert_eq!(component, BenchComponent::All);
                assert_eq!(out.as_deref(), Some(std::path::Path::new("/tmp/run.jsonl")));
                assert!(stt.is_none());
                assert_eq!(llm.as_deref(), Some("qwen3.5-0.8b"));
                assert!(tts.is_none());
            }
            _ => panic!("expected bench"),
        }
        // The hidden subcommand still documents its own flags on demand.
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("bench")
            .expect("hidden bench exists")
            .render_help()
            .to_string();
        assert!(help.contains("--out"), "{help}");
        assert!(help.contains("--stt"), "{help}");
    }

    #[test]
    fn bench_accepts_component_and_independent_model_overrides() {
        let cli = Cli::try_parse_from([
            "syllabix",
            "bench",
            "llm",
            "--stt",
            "medium",
            "--llm",
            "qwen3.5-0.8b",
            "--tts",
            "qwen3-0.6",
        ])
        .expect("component profile parses");
        match cli.command {
            Commands::Bench {
                component,
                stt,
                llm,
                tts,
                ..
            } => {
                assert_eq!(component, BenchComponent::Llm);
                assert_eq!(stt.as_deref(), Some("medium"));
                assert_eq!(llm.as_deref(), Some("qwen3.5-0.8b"));
                assert_eq!(tts.as_deref(), Some("qwen3-0.6"));
            }
            _ => panic!("expected bench"),
        }
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

    #[cfg(coverage)]
    #[test]
    fn execute_bench_uses_fake_providers_and_writes_jsonl() {
        let dir = unique_dir("syllabix-coverage-bench");
        std::fs::create_dir_all(&dir).expect("create output directory");
        let out = dir.join("run.jsonl");
        execute(Cli {
            command: Commands::Bench {
                component: BenchComponent::All,
                out: Some(out.clone()),
                stt: Some("medium".into()),
                llm: None,
                tts: None,
            },
        })
        .expect("coverage bench");
        let rows = std::fs::read_to_string(&out).expect("JSONL output");
        assert_eq!(rows.lines().count(), 6);
        std::fs::remove_dir_all(dir).expect("remove output directory");
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
