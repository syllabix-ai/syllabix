//! Native inference suite (whisper.cpp, llama.cpp, Pocket TTS, six-turn loop).
//!
//! One integration binary so Whisper/Llama/Pocket TTS load once per `cargo test`
//! process. `cargo llvm-cov` sets `--cfg coverage` and skips weight loads;
//! a fake loop still runs so this binary is not a coverage hole.
//!
//! Default `cargo test` loads Silero, Whisper `small`, LFM2.5-350M, and Pocket TTS.
//! Set `SYLLABIX_NATIVE_MODELS` to a comma-separated list of YAML
//! `pipeline.*.model` identifiers to run only those native suites. Unknown
//! identifiers fail fast, and identifiers without a native suite fail
//! with `no native suite for <id>`. Whisper `small` may still load as a
//! TTS→ASR scorer when a TTS id is selected.

#[cfg(not(coverage))]
mod kokoro;
#[cfg(not(coverage))]
mod lfm;
#[cfg(not(coverage))]
mod llama;
#[cfg(not(coverage))]
mod pocket_tts;
#[cfg(not(coverage))]
mod qwen;
#[cfg(not(coverage))]
mod qwen_asr;
#[cfg(not(coverage))]
mod real_loop;
#[cfg(not(coverage))]
mod whisper;

use std::collections::BTreeSet;

#[cfg(not(coverage))]
use std::sync::{Mutex, MutexGuard, OnceLock};

use syllabix_core::{
    SttModel, TtsModel, LFM25_230M_ASSET, LFM25_2_6B_ASSET, LFM25_350M_ASSET, LLAMA_32_1B_ASSET,
    QWEN35_08B_ASSET, QWEN35_2B_ASSET,
};

#[cfg(not(coverage))]
use syllabix_core::{
    build_tts, AgentConfig, Cancel, HttpFetcher, LiveTts, LlamaLlm, ModelCache, StderrProgress,
    WhisperStt,
};

/// YAML identifiers covered by the default native test set.
const LAUNCH_NATIVE_IDS: [&str; 3] = ["whisper-small", LFM25_350M_ASSET, "pocket-tts"];

/// Native-test ids that currently have a suite.
const SUITED_NATIVE_IDS: [&str; 11] = [
    "whisper-small",
    LLAMA_32_1B_ASSET,
    "kokoro",
    "qwen3-0.6",
    "qwen3-1.7",
    "qwen3-asr-0.6",
    QWEN35_08B_ASSET,
    "pocket-tts",
    LFM25_2_6B_ASSET,
    LFM25_350M_ASSET,
    LFM25_230M_ASSET,
];

/// Every selectable local TTS model uses this fixed latency corpus.
/// Keep it spoken and ordinary so this captures reproducible native
/// synthesis latency and throughput.
#[cfg(not(coverage))]
pub(crate) const TTS_LATENCY_SENTENCES: [&str; 20] = [
    "The weather looks clear today.",
    "Remind me to call the dentist tomorrow.",
    "That restaurant opens at six.",
    "A short reply is a good reply.",
    "Please summarize the article in two sentences.",
    "The train leaves before noon.",
    "Coffee first, questions later.",
    "The meeting moved to Thursday afternoon.",
    "Turn left at the next intersection.",
    "This podcast episode runs about an hour.",
    "She finished the marathon in four hours.",
    "The package arrives sometime next week.",
    "Backup files live in the cloud folder.",
    "He plays guitar in a local band.",
    "Dinner smells almost ready.",
    "The report needs one more revision.",
    "Their flight landed late last night.",
    "Sunrise happens earlier in the summer.",
    "The library closes at eight.",
    "Write the note before you forget it.",
];

fn all_yaml_model_ids() -> BTreeSet<&'static str> {
    let mut ids = BTreeSet::new();
    for model in SttModel::ALL {
        ids.insert(model.as_str());
    }
    ids.insert(LLAMA_32_1B_ASSET);
    ids.insert(QWEN35_08B_ASSET);
    ids.insert(QWEN35_2B_ASSET);
    ids.insert(LFM25_2_6B_ASSET);
    ids.insert(LFM25_350M_ASSET);
    ids.insert(LFM25_230M_ASSET);
    for model in TtsModel::ALL {
        ids.insert(model.as_str());
    }
    ids
}

fn all_native_model_ids() -> BTreeSet<&'static str> {
    all_yaml_model_ids()
}

/// Parse `SYLLABIX_NATIVE_MODELS`. Missing or blank input selects the default stack.
fn parse_native_models(raw: Option<&str>) -> Result<BTreeSet<String>, String> {
    let known = all_native_model_ids();
    let suited: BTreeSet<&str> = SUITED_NATIVE_IDS.into_iter().collect();
    let Some(raw) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(LAUNCH_NATIVE_IDS.into_iter().map(str::to_string).collect());
    };
    let mut selected = BTreeSet::new();
    for part in raw.split(',') {
        let id = part.trim();
        if id.is_empty() {
            continue;
        }
        if !known.contains(id) {
            return Err(format!(
                "unknown SYLLABIX_NATIVE_MODELS id {id:?} (allowed: {})",
                known.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
        if !suited.contains(id) {
            return Err(format!(
                "no native suite for {id:?} (suited ids: {})",
                SUITED_NATIVE_IDS.join(", ")
            ));
        }
        selected.insert(id.to_string());
    }
    if selected.is_empty() {
        return Ok(LAUNCH_NATIVE_IDS.into_iter().map(str::to_string).collect());
    }
    Ok(selected)
}

/// `SYLLABIX_NATIVE_LATENCY=1` is exclusive to selected local TTS ids.
fn latency_requires_tts(selected: &BTreeSet<String>, latency: bool) -> Result<(), String> {
    if latency
        && !(selected.contains("kokoro")
            || selected.contains("pocket-tts")
            || selected.contains("qwen3-0.6")
            || selected.contains("qwen3-1.7"))
    {
        return Err(
            "SYLLABIX_NATIVE_LATENCY=1 requires a local TTS id (kokoro, pocket-tts, qwen3-0.6, or qwen3-1.7) in SYLLABIX_NATIVE_MODELS"
                .into(),
        );
    }
    Ok(())
}

fn native_models_from_env() -> Result<BTreeSet<String>, String> {
    let selected = parse_native_models(std::env::var("SYLLABIX_NATIVE_MODELS").ok().as_deref())?;
    latency_requires_tts(&selected, native_latency_env_set())?;
    Ok(selected)
}

fn native_latency_env_set() -> bool {
    std::env::var_os("SYLLABIX_NATIVE_LATENCY").is_some()
}

#[cfg(not(coverage))]
fn selected_native_models() -> &'static BTreeSet<String> {
    static CELL: OnceLock<BTreeSet<String>> = OnceLock::new();
    CELL.get_or_init(|| native_models_from_env().unwrap_or_else(|err| panic!("{err}")))
}

#[cfg(not(coverage))]
pub(crate) fn native_model_selected(id: &str) -> bool {
    selected_native_models().contains(id)
}

#[cfg(not(coverage))]
pub(crate) fn launch_stack_selected() -> bool {
    LAUNCH_NATIVE_IDS.iter().all(|id| native_model_selected(id))
}

#[cfg(not(coverage))]
pub(crate) fn native_latency_enabled() -> bool {
    native_latency_env_set()
}

#[cfg(not(coverage))]
macro_rules! skip_unless_model {
    ($id:expr) => {
        if !crate::native_model_selected($id) {
            eprintln!(
                "skipping native test (set SYLLABIX_NATIVE_MODELS to include {})",
                $id
            );
            return;
        }
    };
}
#[cfg(not(coverage))]
pub(crate) use skip_unless_model;

#[cfg(not(coverage))]
macro_rules! skip_unless_any_model {
    ($($id:expr),+ $(,)?) => {
        if ![$($id),+].iter().any(|id| crate::native_model_selected(id)) {
            eprintln!(
                "skipping native test (set SYLLABIX_NATIVE_MODELS to include one of: {})",
                [$($id),+].join(", ")
            );
            return;
        }
    };
}
#[cfg(not(coverage))]
pub(crate) use skip_unless_any_model;

#[cfg(not(coverage))]
macro_rules! skip_unless_launch_stack {
    () => {
        if !crate::launch_stack_selected() {
            eprintln!(
                "skipping launch-stack native test (needs small, lfm2.5-350m, pocket-tts in SYLLABIX_NATIVE_MODELS)"
            );
            return;
        }
    };
}
#[cfg(not(coverage))]
pub(crate) use skip_unless_launch_stack;

/// Load TTS through the same builder production uses, driven by
/// [`AgentConfig::v0`] (today: Pocket TTS). Suites that need another TTS id
/// (Kokoro, Qwen) construct that engine themselves.
#[cfg(not(coverage))]
fn load_launch_tts() -> LiveTts {
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    build_tts(
        &cache,
        &HttpFetcher,
        &mut progress,
        &cancel,
        &AgentConfig::v0(),
    )
    .expect("load launch-default TTS from AgentConfig::v0()")
}

#[cfg(not(coverage))]
pub(crate) struct Native {
    stt: Option<WhisperStt>,
    llm: Option<LlamaLlm>,
    lfm: Option<LlamaLlm>,
    lfm_asset_id: Option<&'static str>,
    tts: Option<LiveTts>,
}

#[cfg(not(coverage))]
impl Native {
    fn empty() -> Self {
        Self {
            stt: None,
            llm: None,
            lfm: None,
            lfm_asset_id: None,
            tts: None,
        }
    }

    pub(crate) fn stt(&mut self) -> &WhisperStt {
        self.ensure_stt();
        self.stt.as_ref().expect("whisper loaded")
    }

    pub(crate) fn stt_mut(&mut self) -> &mut WhisperStt {
        self.ensure_stt();
        self.stt.as_mut().expect("whisper loaded")
    }

    pub(crate) fn llm(&mut self) -> &LlamaLlm {
        self.ensure_llm();
        self.llm.as_ref().expect("lfm loaded")
    }

    pub(crate) fn llm_mut(&mut self) -> &mut LlamaLlm {
        self.ensure_llm();
        self.llm.as_mut().expect("lfm loaded")
    }

    /// Dedicated LFM handle for tools-dialect tests. The regular `llm` slot
    /// loads the launch default for ordinary generation.
    pub(crate) fn lfm_mut(&mut self) -> &mut LlamaLlm {
        self.lfm_asset_mut(LFM25_2_6B_ASSET)
    }

    /// Load any yaml LFM id into the dedicated LFM slot (replacing a prior id).
    pub(crate) fn lfm_asset_mut(&mut self, asset_id: &'static str) -> &mut LlamaLlm {
        if self.lfm_asset_id != Some(asset_id) {
            self.lfm = None;
            self.lfm_asset_id = None;
        }
        if self.lfm.is_none() {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            let cancel = Cancel::new();
            self.lfm = Some(
                LlamaLlm::from_cached_model(
                    &cache,
                    &HttpFetcher,
                    &mut progress,
                    &cancel,
                    asset_id,
                    false,
                )
                .unwrap_or_else(|err| panic!("load {asset_id} QAD Q4_0 without segfault: {err}")),
            );
            self.lfm_asset_id = Some(asset_id);
        }
        self.lfm.as_mut().expect("LFM loaded")
    }

    fn ensure_llm(&mut self) {
        if self.llm.is_none() {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            let cancel = Cancel::new();
            let llm = LlamaLlm::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)
                .expect("load default LFM QAD Q4_0 once");
            assert!(!llm.thinking(), "v0 thinking is off until yaml enables it");
            self.llm = Some(llm);
        }
    }

    /// Borrow the launch-default TTS (`AgentConfig::v0` → `build_tts`).
    pub(crate) fn tts(&mut self) -> &LiveTts {
        if self.tts.is_none() {
            self.tts = Some(load_launch_tts());
        }
        self.tts.as_ref().expect("launch TTS loaded")
    }

    /// Move the launch-default TTS into a pipeline stage (single weight load).
    pub(crate) fn take_tts(&mut self) -> LiveTts {
        if self.tts.is_none() {
            self.tts = Some(load_launch_tts());
        }
        self.tts.take().expect("launch TTS loaded")
    }

    fn ensure_stt(&mut self) {
        if self.stt.is_none() {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            let cancel = Cancel::new();
            self.stt = Some(
                WhisperStt::from_cache(
                    &cache,
                    &HttpFetcher,
                    &mut progress,
                    &cancel,
                    SttModel::Small,
                )
                .expect("load whisper.cpp small once"),
            );
        }
    }
}

#[cfg(not(coverage))]
pub(crate) fn hex(bytes: impl AsRef<[u8]>) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Shared ggml is process-global; serialize every native test.
/// Weights load lazily so an exclusive `SYLLABIX_NATIVE_MODELS` list does not
/// fetch unlisted GGUFs.
#[cfg(not(coverage))]
pub(crate) fn native() -> MutexGuard<'static, Native> {
    static CELL: OnceLock<Mutex<Native>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(Native::empty()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn native_models_env_is_valid() {
    native_models_from_env().unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn unset_native_models_selects_the_launch_stack() {
    let ids = parse_native_models(None).expect("parse");
    assert_eq!(
        ids,
        BTreeSet::from([
            "whisper-small".into(),
            "lfm2.5-350m".into(),
            "pocket-tts".into()
        ])
    );
}

#[test]
fn launch_tts_config_matches_builtin_defaults() {
    use syllabix_core::BuiltinDefaults;
    let cfg = syllabix_core::AgentConfig::v0();
    let defaults = BuiltinDefaults::v0();
    assert_eq!(cfg.tts, defaults.tts);
    assert_eq!(cfg.tts_model, defaults.tts_model);
    assert_eq!(cfg.tts_model.as_str(), "pocket-tts");
}

#[test]
fn native_models_list_is_exclusive() {
    let ids = parse_native_models(Some("qwen3-0.6")).expect("parse");
    assert_eq!(ids, BTreeSet::from(["qwen3-0.6".into()]));
}

#[test]
fn unknown_native_model_id_fails() {
    let err = parse_native_models(Some("nope")).expect_err("unknown");
    assert!(err.contains("unknown"), "{err}");
    assert!(err.contains("nope"), "{err}");
}

#[test]
fn native_model_without_a_suite_fails() {
    let err = parse_native_models(Some("qwen3.5-2b")).expect_err("no suite");
    assert!(err.contains("no native suite for \"qwen3.5-2b\""), "{err}");
    let err = parse_native_models(Some("whisper-medium")).expect_err("no suite");
    assert!(
        err.contains("no native suite for \"whisper-medium\""),
        "{err}"
    );
}

#[test]
fn suited_native_ids_are_recognized_by_the_native_harness() {
    let known = all_native_model_ids();
    for id in SUITED_NATIVE_IDS {
        assert!(known.contains(id), "{id} must be a native test id");
    }
}

#[test]
fn lfm_is_a_yaml_and_native_model_id() {
    for id in [LFM25_2_6B_ASSET, LFM25_350M_ASSET, LFM25_230M_ASSET] {
        assert!(all_native_model_ids().contains(id), "{id}");
        assert!(all_yaml_model_ids().contains(id), "{id}");
        assert!(SUITED_NATIVE_IDS.contains(&id), "{id}");
    }
}

#[test]
fn latency_without_tts_fails() {
    let launch = parse_native_models(None).expect("launch");
    latency_requires_tts(&launch, true).expect("Pocket TTS is a TTS model");
    let stt = parse_native_models(Some("whisper-small")).expect("stt");
    let err = latency_requires_tts(&stt, true).expect_err("no TTS");
    assert!(err.contains("SYLLABIX_NATIVE_LATENCY"), "{err}");
    let qwen = parse_native_models(Some("qwen3-0.6")).expect("qwen");
    latency_requires_tts(&qwen, true).expect("qwen + latency");
    let pocket = parse_native_models(Some("pocket-tts")).expect("pocket");
    latency_requires_tts(&pocket, true).expect("pocket + latency");
    latency_requires_tts(&stt, false).expect("no latency");
}

#[cfg(coverage)]
#[test]
fn native_weights_are_not_loaded_under_llvm_cov() {
    use syllabix_core::{
        run_loop, scripted_frames, Cancel, CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad,
        LoopConfig, PipelineStages,
    };

    let report = run_loop(
        LoopConfig::default(),
        PipelineStages {
            vad: FakeVad::new(),
            stt: FakeStt,
            llm: FakeLlm::new(),
            tts: FakeTts,
            sink: CollectingSink::default(),
        },
        scripted_frames(1, 2, 1),
        Cancel::new(),
    )
    .expect("coverage fake loop");
    assert_eq!(report.turns.len(), 1);
    assert_eq!(report.tasks_still_running, 0);
}
