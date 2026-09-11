//! Mock-session coverage for the real Pocket TTS adapter. No weights, no Hub.
//!
//! The four ONNX graphs and the SentencePiece tokenizer are injected through
//! the [`PocketTts::from_parts`] seam; package/voice fixtures are synthetic
//! temp files exercising the same validation as the pinned export.

use std::path::{Path, PathBuf};

use super::*;
use crate::onnx::{MockSession, OnnxSession, OnnxTensor};
use crate::{Cancel, Error, GenerationId, TokenChunk, Tts, TurnId};

fn names(words: &[&str]) -> Vec<String> {
    words.iter().map(|word| (*word).to_string()).collect()
}

fn spec(
    input: &str,
    output: &str,
    module: &str,
    key: &str,
    dtype: &str,
    fill: &str,
    shape: Vec<i64>,
) -> StateSpec {
    StateSpec {
        input_name: input.to_string(),
        output_name: output.to_string(),
        module: module.to_string(),
        key: key.to_string(),
        dtype: dtype.to_string(),
        fill: fill.to_string(),
        shape,
    }
}

fn flow_specs() -> Vec<StateSpec> {
    vec![
        spec(
            "s0_in",
            "s0_out",
            "flow0",
            "w0",
            "f32",
            "zeros",
            vec![1, 1, 2, 1, 2],
        ),
        spec("s1_in", "s1_out", "m", "step", "int64", "zeros", vec![1]),
    ]
}

fn mimi_specs() -> Vec<StateSpec> {
    vec![
        spec("m0_in", "m0_out", "mx", "miss", "f32", "zeros", vec![1, 4]),
        spec("m1_in", "m1_out", "b", "b0", "bool", "ones", vec![2]),
        spec("m2_in", "m2_out", "mx", "miss2", "f32", "nan", vec![1]),
    ]
}

fn bundle_json() -> String {
    serde_json::json!({
        "flow_lm_state_manifest": [
            {"input_name": "s0_in", "output_name": "s0_out", "module": "flow0", "key": "w0", "dtype": "f32", "fill": "zeros", "shape": [1, 1, 2, 1, 2]},
            {"input_name": "s1_in", "output_name": "s1_out", "module": "m", "key": "step", "dtype": "int64", "fill": "zeros", "shape": [1]},
        ],
        "mimi_state_manifest": [
            {"input_name": "m0_in", "output_name": "m0_out", "module": "mx", "key": "miss", "dtype": "f32", "fill": "zeros", "shape": [1, 4]},
            {"input_name": "m1_in", "output_name": "m1_out", "module": "b", "key": "b0", "dtype": "bool", "fill": "ones", "shape": [2]},
            {"input_name": "m2_in", "output_name": "m2_out", "module": "mx", "key": "miss2", "dtype": "f32", "fill": "nan", "shape": [1]},
        ],
    })
    .to_string()
}

/// Synthetic safetensors voice: a 5-D F32 tensor, an I64 `offset` backing the
/// `step` fallback, and an F32 entry behind a bool target (dtype mismatch).
fn voice_bytes() -> Vec<u8> {
    let mut tensor_data = Vec::new();
    let mut entries = serde_json::Map::new();
    let mut entry = |entries: &mut serde_json::Map<String, serde_json::Value>,
                     name: &str,
                     dtype: &str,
                     shape: Vec<usize>,
                     bytes: &[u8]| {
        let start = tensor_data.len();
        tensor_data.extend_from_slice(bytes);
        let end = tensor_data.len();
        entries.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, end]}),
        );
    };
    let f32s: Vec<f32> = (0..4).map(|i| i as f32 * 0.25).collect();
    let mut raw = Vec::new();
    for value in &f32s {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    entry(&mut entries, "flow0/w0", "F32", vec![1, 1, 2, 1, 2], &raw);
    entry(
        &mut entries,
        "m/offset",
        "I64",
        vec![1],
        &3_i64.to_le_bytes(),
    );
    entry(&mut entries, "b/b0", "F32", vec![2], &0.0_f32.to_le_bytes());
    let header = serde_json::Value::Object(entries).to_string();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&tensor_data);
    out
}

struct Package {
    paths: Vec<PathBuf>,
}

fn package() -> Package {
    let dir = std::env::temp_dir().join(format!(
        "syllabix-pocket-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, bytes: &[u8]| {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    };
    let bundle = write("bundle.json", bundle_json().as_bytes());
    let bos = write("bos.npy", b"\x93NUMPYv1");
    let tokenizer = write("tokenizer.model", b"not a real model, only non-empty");
    let text = write("text.onnx", b"placeholder");
    let flow_main = write("flow_main.onnx", b"placeholder");
    let flow = write("flow.onnx", b"placeholder");
    let decoder = write("decoder.onnx", b"placeholder");
    let voice = write("voice.safetensors", &voice_bytes());
    Package {
        paths: vec![
            bundle, bos, tokenizer, text, flow_main, flow, decoder, voice,
        ],
    }
}

/// Canned-ids tokenizer.
struct MockTokenizer {
    ids: Vec<i64>,
    fail: Option<String>,
}

impl MockTokenizer {
    fn ids(ids: Vec<i64>) -> Self {
        Self { ids, fail: None }
    }

    fn failing(message: &str) -> Self {
        Self {
            ids: vec![],
            fail: Some(message.to_string()),
        }
    }
}

impl PocketTokenizer for MockTokenizer {
    fn encode_to_ids(&self, _text: &str) -> Result<Vec<i64>> {
        match &self.fail {
            Some(message) => Err(provider(message)),
            None => Ok(self.ids.clone()),
        }
    }
}

fn text_conditioner(tokens: usize) -> MockSession {
    MockSession::with_handler(
        names(&["token_ids"]),
        names(&["embeddings"]),
        move |inputs, _| {
            let count = inputs[0].1.shape[1] as usize;
            assert_eq!(count, tokens, "conditioner sees every token id");
            Ok(vec![OnnxTensor::f32(
                vec![1, count as i64, 1024],
                vec![0.1; count * 1024],
            )])
        },
    )
}

fn conditioning() -> OnnxTensor {
    OnnxTensor::f32(vec![1, 32], vec![0.2; 32])
}

fn flow_state_tensors() -> Vec<OnnxTensor> {
    vec![
        OnnxTensor::f32(vec![1], vec![0.0]),
        OnnxTensor::i64(vec![1], vec![0]),
    ]
}

fn mimi_state_tensors() -> Vec<OnnxTensor> {
    vec![
        OnnxTensor::f32(vec![1, 4], vec![0.0; 4]),
        OnnxTensor::bool(vec![2], vec![0, 0]),
        OnnxTensor::f32(vec![1], vec![0.0]),
    ]
}

/// Flow-LM main: the initial call returns fresh state; frame calls answer
/// below-threshold EOS `hold` times, then trip the EOS window.
fn flow_main(hold: usize) -> MockSession {
    let frames = std::cell::Cell::new(0_usize);
    MockSession::with_handler(vec![], vec![], move |_, outputs| {
        if outputs.contains(&"eos_logit") {
            let frame = frames.get();
            frames.set(frame + 1);
            let eos = if frame < hold { -10.0 } else { 10.0 };
            let mut out = vec![conditioning(), OnnxTensor::f32(vec![1], vec![eos])];
            out.extend(flow_state_tensors());
            Ok(out)
        } else {
            Ok(flow_state_tensors())
        }
    })
}

fn flow() -> MockSession {
    MockSession::with_handler(
        names(&["c", "s", "t", "x"]),
        names(&["flow_dir"]),
        |_, _| Ok(vec![OnnxTensor::f32(vec![1, 32], vec![0.0; 32])]),
    )
}

fn decoder() -> MockSession {
    MockSession::with_handler(names(&["latent"]), names(&["audio_frame"]), |_, _| {
        let mut out = vec![OnnxTensor::f32(vec![1, 2400, 1], vec![0.1; 2400])];
        out.extend(mimi_state_tensors());
        Ok(out)
    })
}

fn engine(tokenizer: MockTokenizer, hold: usize, tokens: usize) -> PocketTts {
    // Every engine owns a synthetic on-disk package: sentence synthesis
    // parses the fixed voice for real.
    let package = package();
    PocketTts::from_parts(
        Box::new(text_conditioner(tokens)),
        Box::new(flow_main(hold)),
        Box::new(flow()),
        Box::new(decoder()),
        Box::new(tokenizer),
        flow_specs(),
        mimi_specs(),
        package.paths[7].clone(),
    )
    .expect("mock contract is valid")
}

fn chunk(turn: u64, text: &str, is_last: bool) -> TokenChunk {
    TokenChunk {
        turn: TurnId(turn),
        generation: GenerationId(0),
        index: 0,
        text: text.to_string(),
        is_last,
    }
}

#[test]
fn mock_engine_reports_provider_identity() {
    let tts = engine(MockTokenizer::ids(vec![1, 2, 3]), 99, 3);
    assert_eq!(tts.name(), "local");
    assert_eq!(tts.model_id(), Some("pocket-tts"));
}

#[test]
fn from_parts_rejects_a_wrong_text_contract() {
    let parts_error = |text: Box<dyn OnnxSession>| match PocketTts::from_parts(
        text,
        Box::new(flow_main(99)),
        Box::new(flow()),
        Box::new(decoder()),
        Box::new(MockTokenizer::ids(vec![1])),
        flow_specs(),
        mimi_specs(),
        PathBuf::from("voice"),
    ) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("bad contract should fail"),
    };
    let err = parts_error(Box::new(MockSession::script(
        names(&["wrong"]),
        names(&["embeddings"]),
        vec![],
    )));
    assert!(err.contains("unexpected text conditioner contract"));
    let err = parts_error(Box::new(MockSession::script(
        names(&["token_ids"]),
        names(&["wrong"]),
        vec![],
    )));
    assert!(err.contains("unexpected text conditioner contract"));
}

#[test]
fn synthesize_chunk_speaks_a_sentence_end_to_end() {
    let mut tts = engine(MockTokenizer::ids(vec![10, 20, 30]), 2, 3);
    let chunks = tts
        .synthesize_chunk(
            &chunk(1, "Hello world, this is a longer sentence.", true),
            &Cancel::new(),
        )
        .expect("mock synthesis");
    assert!(!chunks.is_empty(), "decoder audio reaches playback");
    assert!(chunks.last().unwrap().is_last);
    assert!(chunks.iter().enumerate().all(|(i, c)| c.index == i as u32));
    assert!(chunks.iter().all(|c| c.turn == TurnId(1)));
    assert!(chunks.iter().all(|c| !c.samples.is_empty()));
}

#[test]
fn buffered_text_waits_for_the_sentence_end() {
    let mut tts = engine(MockTokenizer::ids(vec![10, 20, 30]), 99, 3);
    let chunks = tts
        .synthesize_chunk(
            &chunk(1, "Hello world, no period yet", false),
            &Cancel::new(),
        )
        .expect("buffering is not an error");
    assert!(chunks.is_empty());
}

#[test]
fn empty_tokenizer_output_still_emits_one_terminal_marker() {
    let mut tts = engine(MockTokenizer::ids(vec![]), 99, 0);
    let chunks = tts
        .synthesize_chunk(&chunk(1, "Hello world.", true), &Cancel::new())
        .expect("empty ids are not an error");
    assert_eq!(chunks.len(), 1, "finalization must release the turn");
    assert_eq!(chunks[0].samples, vec![0; 16]);
    assert!(chunks[0].is_last);
}

#[test]
fn trailing_markup_only_fragment_keeps_the_terminal_marker() {
    let mut tts = engine(MockTokenizer::ids(vec![10, 20, 30]), 2, 3);
    let chunks = tts
        .synthesize_chunk(&chunk(1, "Hello. **", true), &Cancel::new())
        .expect("trailing markup must not swallow is_last");
    assert!(!chunks.is_empty(), "Hello. must reach synthesis");
    assert_eq!(
        chunks.iter().filter(|c| c.is_last).count(),
        1,
        "exactly one terminal marker, got {chunks:?}"
    );
    assert!(
        chunks.last().unwrap().is_last,
        "terminal marker releases the assistant turn"
    );
}

#[test]
fn think_only_final_chunk_emits_one_terminal_marker() {
    let mut tts = engine(MockTokenizer::ids(vec![1]), 99, 1);
    let chunks = tts
        .synthesize_chunk(
            &chunk(1, "<think>hidden plan.</think>", true),
            &Cancel::new(),
        )
        .expect("think-only finalization must close");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].samples, vec![0; 16]);
    assert!(chunks[0].is_last);
}

#[test]
fn cancelled_final_fallback_does_not_emit_a_terminal_marker() {
    let mut tts = engine(MockTokenizer::ids(vec![]), 99, 0);
    let cancel = Cancel::new();
    cancel.cancel_generation();
    assert!(matches!(
        tts.synthesize_chunk(&chunk(1, "Hello world.", true), &cancel),
        Err(Error::Cancelled)
    ));
}

#[test]
fn whitespace_only_final_chunk_emits_the_silence_beep() {
    let mut tts = engine(MockTokenizer::ids(vec![1]), 99, 1);
    let chunks = tts
        .synthesize_chunk(&chunk(1, "   ", true), &Cancel::new())
        .expect("fallback beep");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].samples, vec![0; 16]);
    assert!(chunks[0].is_last);
}

#[test]
fn tokenizer_failure_aborts_the_chunk() {
    let mut tts = engine(MockTokenizer::failing("tokenizer down"), 99, 0);
    let err = tts
        .synthesize_chunk(&chunk(1, "Hello world.", true), &Cancel::new())
        .unwrap_err();
    assert!(err.to_string().contains("tokenizer down"));
}

#[test]
fn stale_generations_cancel_before_inference() {
    let mut tts = engine(MockTokenizer::ids(vec![1, 2, 3]), 99, 3);
    let cancel = Cancel::new();
    cancel.cancel_generation();
    assert!(matches!(
        tts.synthesize_chunk(&chunk(1, "Hello world.", true), &cancel),
        Err(Error::Cancelled)
    ));
}

#[test]
fn shutdown_during_synthesis_cancels_the_turn() {
    let cancel = Cancel::new();
    let worker = cancel.clone();
    let calls = std::cell::Cell::new(0_usize);
    let flow_main = MockSession::with_handler(vec![], vec![], move |_, outputs| {
        if outputs.contains(&"eos_logit") {
            let frame = calls.get();
            calls.set(frame + 1);
            if frame == 1 {
                worker.shutdown();
            }
            let mut out = vec![conditioning(), OnnxTensor::f32(vec![1], vec![-10.0])];
            out.extend(flow_state_tensors());
            Ok(out)
        } else {
            Ok(flow_state_tensors())
        }
    });
    let mut tts = PocketTts::from_parts(
        Box::new(text_conditioner(3)),
        Box::new(flow_main),
        Box::new(flow()),
        Box::new(decoder()),
        Box::new(MockTokenizer::ids(vec![10, 20, 30])),
        flow_specs(),
        mimi_specs(),
        package().paths[7].clone(),
    )
    .unwrap();
    assert!(matches!(
        tts.synthesize_chunk(
            &chunk(1, "Hello world, this is a longer sentence.", true),
            &cancel
        ),
        Err(Error::Cancelled)
    ));
}

#[test]
fn shutdown_between_sentences_cancels_the_turn() {
    let cancel = Cancel::new();
    let worker = cancel.clone();
    let decoder =
        MockSession::with_handler(names(&["latent"]), names(&["audio_frame"]), move |_, _| {
            worker.shutdown();
            let mut out = vec![OnnxTensor::f32(vec![1, 2400, 1], vec![0.1; 2400])];
            out.extend(mimi_state_tensors());
            Ok(out)
        });
    let mut tts = PocketTts::from_parts(
        Box::new(text_conditioner(3)),
        Box::new(flow_main(99)),
        Box::new(flow()),
        Box::new(decoder),
        Box::new(MockTokenizer::ids(vec![10, 20, 30])),
        flow_specs(),
        mimi_specs(),
        package().paths[7].clone(),
    )
    .unwrap();
    assert!(matches!(
        tts.synthesize_chunk(
            &chunk(
                1,
                "First longer sentence here. Second longer sentence here.",
                true
            ),
            &cancel
        ),
        Err(Error::Cancelled)
    ));
}

#[test]
fn audio_callback_errors_abort_synthesis() {
    let mut tts = engine(MockTokenizer::ids(vec![10, 20, 30]), 2, 3);
    let err = tts
        .synthesize_chunk_into(
            &chunk(1, "Hello world, this is a longer sentence.", true),
            &Cancel::new(),
            &mut |_| Err(Error::Cancelled),
        )
        .unwrap_err();
    assert!(matches!(err, Error::Cancelled));
}

#[test]
fn text_fixture_validates_embeddings() {
    let mut tts = engine(MockTokenizer::ids(vec![1]), 99, 5);
    let embeddings = tts.text_fixture().expect("mock embeddings");
    assert_eq!(embeddings.len(), 5 * 1024);
}

#[test]
fn text_fixture_rejects_empty_and_bad_outputs() {
    let mut tts = PocketTts::from_parts(
        Box::new(MockSession::script(
            names(&["token_ids"]),
            names(&["embeddings"]),
            vec![Ok(vec![])],
        )),
        Box::new(flow_main(99)),
        Box::new(flow()),
        Box::new(decoder()),
        Box::new(MockTokenizer::ids(vec![1])),
        flow_specs(),
        mimi_specs(),
        PathBuf::from("voice"),
    )
    .unwrap();
    let err = tts.text_fixture().unwrap_err();
    assert!(err.to_string().contains("no embeddings"));

    let mut tts = PocketTts::from_parts(
        Box::new(MockSession::with_handler(
            names(&["token_ids"]),
            names(&["embeddings"]),
            |_, _| Ok(vec![OnnxTensor::i64(vec![1, 5], vec![0; 5])]),
        )),
        Box::new(flow_main(99)),
        Box::new(flow()),
        Box::new(decoder()),
        Box::new(MockTokenizer::ids(vec![1])),
        flow_specs(),
        mimi_specs(),
        PathBuf::from("voice"),
    )
    .unwrap();
    let err = tts.text_fixture().unwrap_err();
    assert!(err.to_string().contains("float32"));

    let mut tts = PocketTts::from_parts(
        Box::new(MockSession::with_handler(
            names(&["token_ids"]),
            names(&["embeddings"]),
            |_, _| {
                Ok(vec![OnnxTensor::f32(
                    vec![1, 5, 1024],
                    vec![f32::NAN; 5 * 1024],
                )])
            },
        )),
        Box::new(flow_main(99)),
        Box::new(flow()),
        Box::new(decoder()),
        Box::new(MockTokenizer::ids(vec![1])),
        flow_specs(),
        mimi_specs(),
        PathBuf::from("voice"),
    )
    .unwrap();
    let err = tts.text_fixture().unwrap_err();
    assert!(err.to_string().contains("invalid embeddings"));
}

#[test]
fn c_api_fixture_returns_decoder_audio() {
    let mut tts = engine(MockTokenizer::ids(vec![1]), 99, 1);
    let audio = tts.c_api_fixture().expect("mock decoder frame");
    assert_eq!(audio.len(), 2400);
}

#[test]
fn c_api_fixture_rejects_non_float_audio() {
    let mut tts = PocketTts::from_parts(
        Box::new(text_conditioner(1)),
        Box::new(flow_main(99)),
        Box::new(flow()),
        Box::new(MockSession::with_handler(
            names(&["latent"]),
            names(&["audio_frame"]),
            |_, _| Ok(vec![OnnxTensor::i64(vec![1], vec![0])]),
        )),
        Box::new(MockTokenizer::ids(vec![1])),
        flow_specs(),
        mimi_specs(),
        PathBuf::from("voice"),
    )
    .unwrap();
    let err = tts.c_api_fixture().unwrap_err();
    assert!(err.to_string().contains("float32"));
}

#[test]
fn synthesize_fixture_chains_every_graph() {
    let mut tts = engine(MockTokenizer::ids(vec![1]), 99, 5);
    let audio = tts.synthesize_fixture().expect("mock fixture chain");
    assert_eq!(audio.len(), 2400);
}

fn paths_error(paths: &[PathBuf]) -> String {
    match PocketTts::from_paths(paths) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("bad package should fail"),
    }
}

#[test]
fn from_paths_rejects_a_short_asset_list() {
    let err = paths_error(&[PathBuf::from("only-one")]);
    assert!(err.contains("eight pinned P1 assets"));
}

#[test]
fn from_paths_surfaces_missing_graphs() {
    let package = package();
    let mut paths = package.paths.clone();
    // Keep the inspected package files real; point the graphs at nothing.
    for index in [3, 4, 5, 6] {
        paths[index] = PathBuf::from(format!("/no/such/graph-{index}.onnx"));
    }
    let err = paths_error(&paths);
    assert!(err.contains("could not load text conditioner"));
}

#[test]
fn sentencepiece_open_rejects_garbage() {
    let err = match OrtPocketTokenizer::open(Path::new("/no/such/tokenizer.model")) {
        Err(err) => err,
        Ok(_) => panic!("missing tokenizer should fail"),
    };
    assert!(err.to_string().contains("could not load tokenizer.model"));
}

#[test]
fn inspect_package_rejects_each_broken_asset() {
    let package = package();
    let good = &package.paths;
    let rewrite = |index: usize, bytes: &[u8]| {
        std::fs::write(&good[index], bytes).unwrap();
    };

    rewrite(0, b"{nope");
    assert!(paths_error(good).contains("invalid bundle.json"));
    rewrite(0, b"[1, 2]");
    assert!(paths_error(good).contains("bundle.json must be an object"));
    rewrite(0, bundle_json().as_bytes());
    rewrite(1, b"nope");
    assert!(paths_error(good).contains("not a NumPy asset"));
    rewrite(1, b"\x93NUMPYv1");
    rewrite(2, b"");
    assert!(paths_error(good).contains("tokenizer.model is empty"));
    rewrite(2, b"placeholder");
    std::fs::write(&good[7], b"short").unwrap();
    assert!(paths_error(good).contains("not a safetensors file"));
    std::fs::write(&good[7], voice_bytes()).unwrap();
    let mut broken = serde_json::from_str::<serde_json::Value>(&bundle_json()).unwrap();
    broken
        .as_object_mut()
        .unwrap()
        .remove("mimi_state_manifest");
    rewrite(0, broken.to_string().as_bytes());
    assert!(paths_error(good).contains("bundle lacks mimi_state_manifest"));
}

#[test]
fn voice_state_rejects_an_unreadable_tensor() {
    let dir = std::env::temp_dir().join(format!(
        "syllabix-pocket-voice-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("voice.safetensors");
    let mut header = serde_json::Map::new();
    header.insert(
        "flow0/w0".to_string(),
        serde_json::json!({"dtype": "F32", "shape": [1], "data_offsets": [500, 600]}),
    );
    let header_text = serde_json::Value::Object(header).to_string();
    let mut bytes = (header_text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header_text.as_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = voice_state(&flow_specs(), &path).unwrap_err();
    assert!(err.to_string().contains("fixed voice tensor exceeds file"));

    std::fs::write(&path, b"tiny").unwrap();
    let err = voice_state(&flow_specs(), &path).unwrap_err();
    assert!(err.to_string().contains("invalid fixed voice"));

    let mut bad = vec![8_u8, 0, 0, 0, 0, 0, 0, 0];
    bad.extend_from_slice(b"{bad!!!!");
    std::fs::write(&path, &bad).unwrap();
    let err = voice_state(&flow_specs(), &path).unwrap_err();
    assert!(err.to_string().contains("invalid fixed voice header"));
}

#[test]
fn from_cache_reports_unresolvable_assets() {
    use crate::models::{BlockedFetcher, Manifest, ModelAsset, ModelCache, ModelLayer, NoProgress};
    let root = std::env::temp_dir().join(format!(
        "syllabix-pocket-cache-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let assets = ASSETS
        .iter()
        .map(|id| ModelAsset {
            id: id.to_string(),
            layer: ModelLayer::Tts,
            file_name: format!("{id}.bin"),
            url: "https://example.invalid/model.bin".to_string(),
            sha256: "0".repeat(64),
            size_bytes: 1,
        })
        .collect();
    let cache = ModelCache::new(root, Manifest { version: 1, assets });
    let err = match PocketTts::from_cache(
        &cache,
        &BlockedFetcher::default(),
        &mut NoProgress,
        &Cancel::new(),
    ) {
        Err(err) => err,
        Ok(_) => panic!("blocked fetcher should fail"),
    };
    assert!(!err.to_string().is_empty());
}
