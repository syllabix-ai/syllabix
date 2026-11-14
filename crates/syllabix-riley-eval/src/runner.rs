//! Scored Riley runs (#329 phase A3): for each scenario, reset the database,
//! apply the overlay, let the simulated caller talk to the cascade, then
//! score the call with the deterministic checks and the LLM judge. Writes
//! one JSON line per call and a summary line.
//!
//! Every run calls hosted endpoints (agent, caller, judge, and in voice mode
//! STT and TTS), so it needs `--allow-hosted`: a key in the environment is
//! not consent. Each stage reads only its own key (`SYLLABIX_LLM_API_KEY`
//! for agent, caller, and judge; `SYLLABIX_STT_API_KEY`;
//! `SYLLABIX_TTS_API_KEY`). Scores are our own approximation and are not
//! comparable with Veris's VAmoS leaderboard.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use serde_json::{json, Value};
use syllabix_core::models::{HttpFetcher, NoProgress};
use syllabix_core::openai::{self, DEFAULT_LLM_BASE_URL};
use syllabix_core::remote_stt::{self, RemoteStt, RemoteSttSettings};
use syllabix_core::remote_tts::{self, RemoteTts, RemoteTtsSettings, DEFAULT_VOICE};
use syllabix_core::testing::{FakeTts, FakeVad};
use syllabix_core::vad::SileroVad;
use syllabix_core::{Cancel, ModelCache, Stt, Tts, Vad};
use zeroize::Zeroizing;

use crate::caller::{CallerSim, QueueStt, SimCaller, TextQueue, TextVoice, TtsVoice};
use crate::chat::ChatClient;
use crate::checks::evaluate;
use crate::db::CardDb;
use crate::judge::judge;
use crate::llm::{RileyLlm, RileyLlmSettings, PROVIDER_NAME};
use crate::scenario::{self, Scenario};
use crate::session::{run_session_with, SessionConfig, SessionReport, Stages};
use crate::voice::CallerScript;
use crate::{DATABASE_URL_ENV, DEFAULT_SCHEMA, SCHEMA_ENV};

/// Printed with every summary.
pub const LABEL: &str = "Our own offline approximation of a Riley/VAmoS-style eval; \
                         not comparable with Veris's VAmoS leaderboard.";

/// How the caller reaches the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Caller lines are synthesized with TTS and heard through VAD and STT.
    Voice,
    /// Caller text goes straight to the agent. Smoke check, not a voice score.
    Text,
}

impl Mode {
    /// Name in the results file.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Voice => "voice",
            Mode::Text => "text-smoke",
        }
    }
}

/// VAD for voice runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadChoice {
    /// Silero from the model cache, real-time pacing (the binary).
    Silero,
    /// Energy VAD with fast pacing (tests with synthetic audio).
    Energy,
}

/// Command line of `syllabix-riley-eval`.
#[derive(Debug, Clone, Parser)]
#[command(
    name = "syllabix-riley-eval",
    about = "Score Riley calls against the scenario pack (our own approximation, not comparable with Veris's leaderboard)"
)]
pub struct Args {
    /// Allow calls to the hosted endpoints below. Required: keys alone are not consent.
    #[arg(long)]
    pub allow_hosted: bool,
    /// `voice` scores calls through TTS, VAD, and STT; `text` is a smoke check only.
    #[arg(long, value_enum, default_value_t = Mode::Voice)]
    pub mode: Mode,
    /// Scenario id to run (repeatable; default: the whole pack).
    #[arg(long = "scenario")]
    pub scenarios: Vec<String>,
    /// JSONL results file (overwritten).
    #[arg(long)]
    pub out: PathBuf,
    /// OpenAI-compatible base URL for agent, caller, and judge.
    #[arg(long, default_value = DEFAULT_LLM_BASE_URL)]
    pub llm_base_url: String,
    /// Agent model.
    #[arg(long)]
    pub agent_model: String,
    /// Caller model (default: the agent model).
    #[arg(long)]
    pub caller_model: Option<String>,
    /// Judge model (default: the agent model).
    #[arg(long)]
    pub judge_model: Option<String>,
    /// STT base URL (voice mode).
    #[arg(long)]
    pub stt_base_url: Option<String>,
    /// STT model (voice mode).
    #[arg(long)]
    pub stt_model: Option<String>,
    /// STT language.
    #[arg(long, default_value = "en")]
    pub language: String,
    /// TTS base URL (voice mode).
    #[arg(long)]
    pub tts_base_url: Option<String>,
    /// TTS model (voice mode).
    #[arg(long)]
    pub tts_model: Option<String>,
    /// Agent TTS voice.
    #[arg(long, default_value = DEFAULT_VOICE)]
    pub agent_voice: String,
    /// Caller TTS voice.
    #[arg(long, default_value = DEFAULT_VOICE)]
    pub caller_voice: String,
    /// Seconds a caller turn waits for the agent.
    #[arg(long, default_value_t = 60)]
    pub turn_timeout_secs: u64,
    /// Seconds per call.
    #[arg(long, default_value_t = 600)]
    pub call_timeout_secs: u64,
    /// Seconds per chat-completions request.
    #[arg(long, default_value_t = 60)]
    pub request_timeout_secs: u64,
}

/// Why a run stopped before scoring every scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// `--allow-hosted` was not given. Nothing was called.
    NoConsent,
    /// Bad flags, keys, database, or output file.
    Setup(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::NoConsent => write!(
                f,
                "this run calls hosted endpoints; pass --allow-hosted to confirm (keys alone are not consent)"
            ),
            RunError::Setup(message) => f.write_str(message),
        }
    }
}

fn setup(message: impl std::fmt::Display) -> RunError {
    RunError::Setup(message.to_string())
}

/// Host (and port) of `url`, without credentials, path, or query.
pub fn endpoint_host(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host)
        .to_string()
}

struct Voice {
    stt_base: String,
    stt_model: String,
    tts_base: String,
    tts_model: String,
    stt_key: Zeroizing<String>,
    tts_key: Zeroizing<String>,
}

struct Plan {
    scenarios: Vec<Scenario>,
    llm_key: Zeroizing<String>,
    voice: Option<Voice>,
    database_url: String,
    schema: String,
    caller_model: String,
    judge_model: String,
}

fn required(value: &Option<String>, flag: &str) -> Result<String, RunError> {
    value
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| setup(format!("--mode voice needs {flag}")))
}

fn plan(args: &Args, getenv: &dyn Fn(&str) -> Option<String>) -> Result<Plan, RunError> {
    if !args.allow_hosted {
        return Err(RunError::NoConsent);
    }
    let scenarios = scenario::select(&args.scenarios).map_err(setup)?;
    let llm_key = openai::resolve_api_key(getenv).map_err(setup)?;
    let voice = match args.mode {
        Mode::Text => None,
        Mode::Voice => Some(Voice {
            stt_base: required(&args.stt_base_url, "--stt-base-url")?,
            stt_model: required(&args.stt_model, "--stt-model")?,
            tts_base: required(&args.tts_base_url, "--tts-base-url")?,
            tts_model: required(&args.tts_model, "--tts-model")?,
            stt_key: remote_stt::resolve_api_key(getenv).map_err(setup)?,
            tts_key: remote_tts::resolve_api_key(getenv).map_err(setup)?,
        }),
    };
    let database_url = getenv(DATABASE_URL_ENV)
        .filter(|url| !url.trim().is_empty())
        .ok_or_else(|| setup(format!("{DATABASE_URL_ENV} is not set")))?;
    let schema = getenv(SCHEMA_ENV).unwrap_or_else(|| DEFAULT_SCHEMA.to_string());
    crate::db::validate_schema(&schema).map_err(setup)?;
    Ok(Plan {
        scenarios,
        llm_key,
        voice,
        database_url,
        schema,
        caller_model: args
            .caller_model
            .clone()
            .unwrap_or_else(|| args.agent_model.clone()),
        judge_model: args
            .judge_model
            .clone()
            .unwrap_or_else(|| args.agent_model.clone()),
    })
}

fn hosted_stages(args: &Args, plan: &Plan) -> Vec<Value> {
    let llm_host = endpoint_host(&args.llm_base_url);
    let mut stages = vec![
        json!({"stage": "agent", "provider": PROVIDER_NAME, "host": llm_host, "model": args.agent_model}),
        json!({"stage": "caller", "host": llm_host, "model": plan.caller_model}),
        json!({"stage": "judge", "host": llm_host, "model": plan.judge_model}),
    ];
    if let Some(voice) = &plan.voice {
        stages.push(json!({"stage": "stt", "host": endpoint_host(&voice.stt_base), "model": voice.stt_model}));
        stages.push(json!({"stage": "tts", "host": endpoint_host(&voice.tts_base), "model": voice.tts_model, "voice": args.agent_voice}));
        stages.push(json!({"stage": "caller-tts", "host": endpoint_host(&voice.tts_base), "model": voice.tts_model, "voice": args.caller_voice}));
    }
    stages
}

fn session_config(args: &Args, vad: VadChoice) -> SessionConfig {
    let mut config = if args.mode == Mode::Voice && vad == VadChoice::Silero {
        SessionConfig::default()
    } else {
        SessionConfig::fast()
    };
    config.turn_timeout = Duration::from_secs(args.turn_timeout_secs);
    config.call_timeout = Duration::from_secs(args.call_timeout_secs);
    config
}

fn load_silero() -> Result<SileroVad, String> {
    SileroVad::from_cache(
        &ModelCache::v0(),
        &HttpFetcher::new(),
        &mut NoProgress,
        &Cancel::new(),
    )
    .map_err(|err| format!("could not load Silero VAD: {err}"))
}

fn session<V, S, T>(
    config: &SessionConfig,
    vad: V,
    stt: S,
    llm: RileyLlm,
    tts: T,
    caller: Box<dyn CallerScript>,
    db: &mut CardDb,
) -> SessionReport
where
    V: Vad + 'static,
    S: Stt + 'static,
    T: Tts + 'static,
{
    run_session_with(config, Stages { vad, stt, llm, tts }, caller, db)
}

struct CallSetup<'a> {
    args: &'a Args,
    plan: &'a Plan,
    config: &'a SessionConfig,
    vad: VadChoice,
    judge: &'a ChatClient,
}

fn run_call(ctx: &CallSetup, scenario: &Scenario, schema: &str) -> Result<Value, RunError> {
    let (args, plan) = (ctx.args, ctx.plan);
    let mut tools_db = CardDb::connect(&plan.database_url, schema).map_err(setup)?;
    tools_db.reset().map_err(setup)?;
    tools_db
        .apply_overlay(&scenario.db_overlay)
        .map_err(|err| setup(format!("{}: overlay failed: {err}", scenario.id)))?;
    let mut check_db = CardDb::connect(&plan.database_url, schema).map_err(setup)?;
    let request_timeout = Duration::from_secs(args.request_timeout_secs);
    let mut settings = RileyLlmSettings::new(&args.llm_base_url, &args.agent_model);
    settings.request_timeout = request_timeout;
    let llm = RileyLlm::with_card_db(settings, plan.llm_key.clone(), tools_db);
    let sim = CallerSim::new(
        ChatClient::new(
            &args.llm_base_url,
            &plan.caller_model,
            plan.llm_key.clone(),
            request_timeout,
        ),
        &scenario.caller,
        scenario.max_caller_turns,
    );
    let started = Instant::now();
    let report = match &plan.voice {
        None => {
            let queue = TextQueue::default();
            let caller = Box::new(SimCaller::new(sim, TextVoice(queue.clone())));
            session(
                ctx.config,
                FakeVad::new(),
                QueueStt(queue),
                llm,
                FakeTts,
                caller,
                &mut check_db,
            )
        }
        Some(voice) => {
            let stt = RemoteStt::new(
                RemoteSttSettings {
                    endpoint: remote_stt::join_transcriptions_endpoint(&voice.stt_base),
                    model: voice.stt_model.clone(),
                    language: args.language.clone(),
                },
                voice.stt_key.clone(),
            );
            let tts_with = |voice_name: &str| {
                RemoteTts::new(
                    RemoteTtsSettings {
                        endpoint: remote_tts::join_speech_endpoint(&voice.tts_base),
                        model: voice.tts_model.clone(),
                        voice: voice_name.to_string(),
                    },
                    voice.tts_key.clone(),
                )
            };
            let caller = Box::new(SimCaller::new(sim, TtsVoice(tts_with(&args.caller_voice))));
            let tts = tts_with(&args.agent_voice);
            match ctx.vad {
                VadChoice::Silero => session(
                    ctx.config,
                    load_silero().map_err(setup)?,
                    stt,
                    llm,
                    tts,
                    caller,
                    &mut check_db,
                ),
                VadChoice::Energy => session(
                    ctx.config,
                    FakeVad::new(),
                    stt,
                    llm,
                    tts,
                    caller,
                    &mut check_db,
                ),
            }
        }
    };
    let call_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    let checks = evaluate(&report, &scenario.expect);
    let verdict = judge(ctx.judge, scenario, &report, &checks.failures);
    let _ = check_db.drop_schema();
    Ok(json!({
        "kind": "call",
        "scenario": scenario.id,
        "category": scenario.category.as_str(),
        "mode": args.mode.label(),
        "passed": checks.passed && verdict.passed,
        "checks": {"passed": checks.passed, "failures": checks.failures},
        "judge": verdict.to_json(),
        "latency_ms": {"call": call_ms, "turns": report.to_json()["turn_latency_ms"]},
        "session": report.to_json(),
    }))
}

/// Summary of the calls in `records` (the last line of the results file).
pub fn summarize(args: &Args, stages: Vec<Value>, records: &[Value]) -> Value {
    let passed = records.iter().filter(|r| r["passed"] == true).count();
    let mut by_category: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for record in records {
        let entry = by_category
            .entry(record["category"].as_str().unwrap_or_default().to_string())
            .or_default();
        entry.0 += 1;
        entry.1 += usize::from(record["passed"] == true);
    }
    let rate = |passed: usize, calls: usize| {
        if calls == 0 {
            0.0
        } else {
            passed as f64 / calls as f64
        }
    };
    let pass_rate = rate(passed, records.len());
    json!({
        "kind": "summary",
        "label": LABEL,
        "agent_stage": PROVIDER_NAME,
        "mode": args.mode.label(),
        "score": match args.mode {
            Mode::Voice => "voice",
            Mode::Text => "text smoke only (not a voice score)",
        },
        "hosted_stages": stages,
        "calls": records.len(),
        "passed": passed,
        "pass_rate": pass_rate,
        "voice_pass_rate": (args.mode == Mode::Voice).then_some(pass_rate),
        "by_category": by_category
            .into_iter()
            .map(|(category, (calls, passed))| {
                (category, json!({"calls": calls, "passed": passed, "pass_rate": rate(passed, calls)}))
            })
            .collect::<serde_json::Map<_, _>>(),
    })
}

/// Run every selected scenario and return the summary.
pub fn run(
    args: &Args,
    getenv: &dyn Fn(&str) -> Option<String>,
    vad: VadChoice,
) -> Result<Value, RunError> {
    let plan = plan(args, getenv)?;
    let file = File::create(&args.out)
        .map_err(|err| setup(format!("cannot write {}: {err}", args.out.display())))?;
    let mut out = BufWriter::new(file);
    let config = session_config(args, vad);
    let judge_client = ChatClient::new(
        &args.llm_base_url,
        &plan.judge_model,
        plan.llm_key.clone(),
        Duration::from_secs(args.request_timeout_secs),
    );
    let ctx = CallSetup {
        args,
        plan: &plan,
        config: &config,
        vad,
        judge: &judge_client,
    };
    let mut records = Vec::new();
    for (index, scenario) in plan.scenarios.iter().enumerate() {
        let schema = format!("{}_call{index}", plan.schema);
        let record = run_call(&ctx, scenario, &schema)?;
        writeln!(out, "{record}")
            .and_then(|()| out.flush())
            .map_err(setup)?;
        records.push(record);
    }
    let summary = summarize(args, hosted_stages(args, &plan), &records);
    writeln!(out, "{summary}")
        .and_then(|()| out.flush())
        .map_err(setup)?;
    Ok(summary)
}

/// Binary entry point: exit 0 after a full run (whatever the pass rate),
/// 2 without `--allow-hosted`, 1 on setup errors.
pub fn execute(
    args: &Args,
    getenv: &dyn Fn(&str) -> Option<String>,
    vad: VadChoice,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    match run(args, getenv, vad) {
        Ok(summary) => {
            let text = serde_json::to_string_pretty(&summary).unwrap_or_default();
            let _ = writeln!(stdout, "{text}");
            0
        }
        Err(err) => {
            let _ = writeln!(stderr, "syllabix-riley-eval: {err}");
            if err == RunError::NoConsent {
                2
            } else {
                1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(extra: &[&str]) -> Args {
        let mut argv = vec![
            "syllabix-riley-eval",
            "--out",
            "/tmp/unused.jsonl",
            "--agent-model",
            "m",
        ];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).unwrap()
    }

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn host_drops_scheme_credentials_path_and_query() {
        assert_eq!(endpoint_host("https://api.openai.com/v1"), "api.openai.com");
        assert_eq!(
            endpoint_host("http://u:p@127.0.0.1:8080/v1?k=x"),
            "127.0.0.1:8080"
        );
        assert_eq!(endpoint_host("localhost:9000"), "localhost:9000");
    }

    #[test]
    fn no_consent_is_checked_before_keys() {
        let err = plan(&args(&[]), &|_| None).err().unwrap();
        assert_eq!(err, RunError::NoConsent);
        assert!(err.to_string().contains("--allow-hosted"));
    }

    #[test]
    fn each_stage_needs_its_own_key() {
        let all = [
            ("SYLLABIX_LLM_API_KEY", "l"),
            ("SYLLABIX_STT_API_KEY", "s"),
            ("SYLLABIX_TTS_API_KEY", "t"),
            ("SYLLABIX_RILEY_DATABASE_URL", "postgres://x"),
        ];
        let voice = args(&[
            "--allow-hosted",
            "--stt-base-url",
            "http://s/v1",
            "--stt-model",
            "sm",
            "--tts-base-url",
            "http://t/v1",
            "--tts-model",
            "tm",
        ]);
        for missing in [
            "SYLLABIX_LLM_API_KEY",
            "SYLLABIX_STT_API_KEY",
            "SYLLABIX_TTS_API_KEY",
        ] {
            let getenv = |name: &str| {
                (name != missing)
                    .then(|| {
                        all.iter()
                            .find(|(k, _)| *k == name)
                            .map(|(_, v)| v.to_string())
                    })
                    .flatten()
            };
            let err = plan(&voice, &getenv).err().unwrap().to_string();
            assert!(err.contains(missing), "{missing}: {err}");
        }
        let text = args(&["--allow-hosted", "--mode", "text"]);
        let only_llm = env(&[
            ("SYLLABIX_LLM_API_KEY", "l"),
            ("SYLLABIX_RILEY_DATABASE_URL", "postgres://x"),
        ]);
        let plan = plan(&text, &only_llm).unwrap();
        assert!(plan.voice.is_none());
        assert_eq!(
            (plan.caller_model.as_str(), plan.judge_model.as_str()),
            ("m", "m")
        );
        assert_eq!(plan.schema, DEFAULT_SCHEMA);
        let stages = hosted_stages(&text, &plan);
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0]["provider"], PROVIDER_NAME);
    }

    #[test]
    fn voice_flags_database_and_schema_are_checked() {
        let keys = env(&[
            ("SYLLABIX_LLM_API_KEY", "l"),
            ("SYLLABIX_STT_API_KEY", "s"),
            ("SYLLABIX_TTS_API_KEY", "t"),
        ]);
        let err = plan(&args(&["--allow-hosted"]), &keys).err().unwrap();
        assert_eq!(err.to_string(), "--mode voice needs --stt-base-url");
        let err = plan(&args(&["--allow-hosted", "--mode", "text"]), &keys)
            .err()
            .unwrap();
        assert_eq!(err.to_string(), format!("{DATABASE_URL_ENV} is not set"));
        let bad_schema = env(&[
            ("SYLLABIX_LLM_API_KEY", "l"),
            ("SYLLABIX_RILEY_DATABASE_URL", "postgres://x"),
            ("SYLLABIX_RILEY_SCHEMA", "Bad-Name"),
        ]);
        let err = plan(&args(&["--allow-hosted", "--mode", "text"]), &bad_schema)
            .err()
            .unwrap();
        assert!(err.to_string().contains("invalid schema name"), "{err}");
        let err = plan(
            &args(&["--allow-hosted", "--mode", "text", "--scenario", "x"]),
            &bad_schema,
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("unknown scenario"), "{err}");
    }

    #[test]
    fn summary_separates_text_smoke_from_voice_score() {
        let records = [
            json!({"category": "simple", "passed": true}),
            json!({"category": "adversarial", "passed": false}),
        ];
        let text = summarize(&args(&["--mode", "text"]), vec![], &records);
        assert_eq!(text["label"], LABEL);
        assert_eq!(text["agent_stage"], "riley-openai");
        assert_eq!(text["mode"], "text-smoke");
        assert_eq!(text["pass_rate"], 0.5);
        assert_eq!(text["voice_pass_rate"], Value::Null);
        assert_eq!(text["by_category"]["simple"]["passed"], 1);
        assert_eq!(text["by_category"]["adversarial"]["pass_rate"], 0.0);
        let voice = summarize(&args(&[]), vec![], &records);
        assert_eq!(voice["score"], "voice");
        assert_eq!(voice["voice_pass_rate"], 0.5);
        assert_eq!(summarize(&args(&[]), vec![], &[])["pass_rate"], 0.0);
    }

    #[test]
    fn session_pacing_follows_mode_and_vad() {
        let voice = args(&["--turn-timeout-secs", "5", "--call-timeout-secs", "9"]);
        let live = session_config(&voice, VadChoice::Silero);
        assert!(live.realtime_playback);
        assert_eq!(live.trailing_silence_frames, 32);
        assert_eq!(live.turn_timeout, Duration::from_secs(5));
        assert_eq!(live.call_timeout, Duration::from_secs(9));
        assert!(!session_config(&voice, VadChoice::Energy).realtime_playback);
        assert!(!session_config(&args(&["--mode", "text"]), VadChoice::Silero).realtime_playback);
    }

    #[test]
    fn execute_maps_errors_to_exit_codes() {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        assert_eq!(
            execute(&args(&[]), &|_| None, VadChoice::Energy, &mut out, &mut err),
            2
        );
        assert!(String::from_utf8_lossy(&err).contains("--allow-hosted"));
        err.clear();
        let code = execute(
            &args(&["--allow-hosted"]),
            &|_| None,
            VadChoice::Energy,
            &mut out,
            &mut err,
        );
        assert_eq!(code, 1);
        assert!(String::from_utf8_lossy(&err).contains("SYLLABIX_LLM_API_KEY"));
        assert!(out.is_empty());
    }
}
