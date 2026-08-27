//! Native inference suite (whisper.cpp, llama.cpp, Kokoro, six-turn loop).
//!
//! One integration binary so Whisper/Llama/Kokoro load once per `cargo test`
//! process. `cargo llvm-cov` sets `--cfg coverage` and skips weight loads;
//! a fake loop still runs so this binary is not a coverage hole.
//!
//! Default `cargo test` is the launch stack only (Silero, Whisper `small`,
//! Llama 3.2 1B, Kokoro). Set `SYLLABIX_NATIVE_MODELS` to a comma-separated
//! list of yaml `pipeline.*.model` ids to run **only** those native suites
//! (exclusive). Unknown ids fail fast. Ids with no native suite yet fail
//! with `no native suite for <id>`. Whisper `small` may still load as a
//! TTS→ASR scorer when a TTS id is selected.

#[cfg(not(coverage))]
mod audio8;
#[cfg(not(coverage))]
mod kokoro;
#[cfg(not(coverage))]
mod llama;
#[cfg(not(coverage))]
mod qwen;
#[cfg(not(coverage))]
mod real_loop;
#[cfg(not(coverage))]
mod whisper;

use std::collections::BTreeSet;

#[cfg(not(coverage))]
use std::sync::{Mutex, MutexGuard, OnceLock};

use syllabix_core::{SttModel, TtsModel, LLAMA_32_1B_ASSET, QWEN35_08B_ASSET, QWEN35_2B_ASSET};

#[cfg(not(coverage))]
use syllabix_core::{
    Cancel, HttpFetcher, KokoroTts, LlamaLlm, ModelCache, StderrProgress, WhisperStt,
};

/// Yaml ids the launch stack native tests cover.
const LAUNCH_NATIVE_IDS: [&str; 3] = ["small", LLAMA_32_1B_ASSET, "kokoro"];

/// Yaml ids that currently have a native suite.
const SUITED_NATIVE_IDS: [&str; 7] = [
    "small",
    LLAMA_32_1B_ASSET,
    "kokoro",
    "qwen3-0.6",
    "qwen3-1.7",
    QWEN35_08B_ASSET,
    "audio8",
];

fn all_yaml_model_ids() -> BTreeSet<&'static str> {
    let mut ids = BTreeSet::new();
    for model in SttModel::ALL {
        ids.insert(model.as_str());
    }
    ids.insert(LLAMA_32_1B_ASSET);
    ids.insert(QWEN35_08B_ASSET);
    ids.insert(QWEN35_2B_ASSET);
    for model in TtsModel::ALL {
        ids.insert(model.as_str());
    }
    // A1 is deliberately not YAML-selectable yet; this private native-suite
    // id is the cross-platform feasibility proof before A2 exposes it.
    ids.insert("audio8");
    ids
}

/// Parse `SYLLABIX_NATIVE_MODELS`. `None` / blank ⇒ launch stack.
fn parse_native_models(raw: Option<&str>) -> Result<BTreeSet<String>, String> {
    let known = all_yaml_model_ids();
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

/// `SYLLABIX_NATIVE_LATENCY=1` is exclusive to selected Qwen TTS ids.
fn latency_requires_qwen_tts(selected: &BTreeSet<String>, latency: bool) -> Result<(), String> {
    if latency && !(selected.contains("qwen3-0.6") || selected.contains("qwen3-1.7")) {
        return Err(
            "SYLLABIX_NATIVE_LATENCY=1 requires qwen3-0.6 and/or qwen3-1.7 in SYLLABIX_NATIVE_MODELS"
                .into(),
        );
    }
    Ok(())
}

fn native_models_from_env() -> Result<BTreeSet<String>, String> {
    let selected = parse_native_models(std::env::var("SYLLABIX_NATIVE_MODELS").ok().as_deref())?;
    latency_requires_qwen_tts(&selected, native_latency_env_set())?;
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
                "skipping launch-stack native test (needs small, llama-3.2-1b, kokoro in SYLLABIX_NATIVE_MODELS)"
            );
            return;
        }
    };
}
#[cfg(not(coverage))]
pub(crate) use skip_unless_launch_stack;

#[cfg(not(coverage))]
pub(crate) struct Native {
    stt: Option<WhisperStt>,
    llm: Option<LlamaLlm>,
    tts: Option<KokoroTts>,
}

#[cfg(not(coverage))]
impl Native {
    fn empty() -> Self {
        Self {
            stt: None,
            llm: None,
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
        self.llm.as_ref().expect("llama loaded")
    }

    pub(crate) fn llm_mut(&mut self) -> &mut LlamaLlm {
        self.ensure_llm();
        self.llm.as_mut().expect("llama loaded")
    }

    fn ensure_llm(&mut self) {
        if self.llm.is_none() {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            let cancel = Cancel::new();
            let llm = LlamaLlm::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)
                .expect("load Llama 3.2 1B Q4_K_M once");
            assert!(!llm.thinking(), "v0 thinking is off until yaml enables it");
            self.llm = Some(llm);
        }
    }

    pub(crate) fn tts(&mut self) -> &KokoroTts {
        if self.tts.is_none() {
            let cache = ModelCache::v0();
            let mut progress = StderrProgress::new();
            let cancel = Cancel::new();
            self.tts = Some(
                KokoroTts::from_cache(&cache, &HttpFetcher, &mut progress, &cancel)
                    .expect("load Kokoro ONNX once"),
            );
        }
        self.tts.as_ref().expect("kokoro loaded")
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
        BTreeSet::from(["small".into(), "llama-3.2-1b".into(), "kokoro".into()])
    );
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
    let err = parse_native_models(Some("medium")).expect_err("no suite");
    assert!(err.contains("no native suite for \"medium\""), "{err}");
}

#[test]
fn suited_native_ids_are_yaml_ids() {
    let known = all_yaml_model_ids();
    for id in SUITED_NATIVE_IDS {
        assert!(known.contains(id), "{id} must be a yaml model id");
    }
}

#[test]
fn latency_without_qwen_tts_fails() {
    let launch = parse_native_models(None).expect("launch");
    let err = latency_requires_qwen_tts(&launch, true).expect_err("no qwen");
    assert!(err.contains("SYLLABIX_NATIVE_LATENCY"), "{err}");
    let qwen = parse_native_models(Some("qwen3-0.6")).expect("qwen");
    latency_requires_qwen_tts(&qwen, true).expect("qwen + latency");
    latency_requires_qwen_tts(&launch, false).expect("no latency");
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
