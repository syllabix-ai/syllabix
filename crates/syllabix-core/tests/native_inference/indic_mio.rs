//! Indic-Mio native gates. Opt-in via `SYLLABIX_NATIVE_MODELS=indic-mio`
//! with `--features native-inference`. Downloads the GGUF and
//! decoder ONNX (~1.02 GB) into the model cache.

use std::path::PathBuf;

use syllabix_core::defaults::TtsCompute;
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::tts::{
    IndicMioTts, MioDecoder, INDIC_MIO_ASSET, INDIC_MIO_TEST_SEED, MIOCODEC_DECODER_ASSET,
};
use syllabix_core::types::GenerationId;
use syllabix_core::{Cancel, ModelCache, TokenChunk, Tts, TurnId};
use syllabix_native::audio::istft::SAMPLES_PER_TOKEN;

use crate::skip_unless_model;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../syllabix-native/tests/fixtures/miocodec_istft")
}

fn read_f32(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(fixture_dir().join(name)).expect(name);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_tokens() -> Vec<i32> {
    let text = std::fs::read_to_string(fixture_dir().join("tokens.txt")).expect("tokens.txt");
    text.split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("token"))
        .collect()
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let a = &a[..n];
    let b = &b[..n];
    let mean_a = a.iter().sum::<f32>() / n as f32;
    let mean_b = b.iter().sum::<f32>() / n as f32;
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    num / (den_a.sqrt() * den_b.sqrt())
}

fn load_indic_mio() -> IndicMioTts {
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    let backbone = cache
        .manifest()
        .asset(INDIC_MIO_ASSET)
        .expect("indic-mio asset")
        .clone();
    let decoder = cache
        .manifest()
        .asset(MIOCODEC_DECODER_ASSET)
        .expect("miocodec-decoder asset")
        .clone();
    let model_path = cache
        .resolve(&backbone, &HttpFetcher, &mut progress, &cancel)
        .expect("indic-mio GGUF");
    let decoder_path = cache
        .resolve(&decoder, &HttpFetcher, &mut progress, &cancel)
        .expect("miocodec decoder");
    IndicMioTts::from_paths_with_seed(
        model_path,
        decoder_path,
        INDIC_MIO_TEST_SEED,
        TtsCompute::Cpu,
        "hi-female",
    )
    .expect("load Indic-Mio")
}

fn token(text: &str, is_last: bool) -> TokenChunk {
    TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index: 0,
        text: text.into(),
        is_last,
    }
}

#[test]
fn decoder_onnx_loads_under_pinned_ort_and_matches_fixture() {
    skip_unless_model!("indic-mio");
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    let decoder_asset = cache
        .manifest()
        .asset(MIOCODEC_DECODER_ASSET)
        .expect("miocodec-decoder asset")
        .clone();
    let decoder_path = cache
        .resolve(&decoder_asset, &HttpFetcher, &mut progress, &cancel)
        .expect("decoder onnx");
    let mut decoder = MioDecoder::load(&decoder_path).expect("ort load");
    let codes = read_tokens();
    let embedding = read_f32("voice.f32");
    let mut voice = [0.0f32; 128];
    voice.copy_from_slice(&embedding[..128]);
    let pcm = decoder.decode(&codes, &voice).expect("decode");
    let expected = read_f32("decoder_pcm.f32");
    let corr = correlation(&pcm, &expected);
    assert!(
        corr >= 0.98,
        "decoder vs committed ONNX+iSTFT waveform correlation {corr} (want >= 0.98)"
    );
    assert_eq!(expected.len(), codes.len() * SAMPLES_PER_TOKEN);
}

#[test]
fn one_sentence_with_fixed_seed_emits_pcm() {
    skip_unless_model!("indic-mio");
    let mut tts = load_indic_mio();
    assert_eq!(tts.name(), "local");
    assert_eq!(tts.model_id(), Some("indic-mio-q8"));
    let chunks = tts
        .synthesize_chunk(&token("नमस्ते, आप कैसे हैं।", true), &Cancel::new())
        .expect("synth");
    assert!(!chunks.is_empty());
    assert!(chunks[0].samples.iter().any(|s| s.abs() > 32));
}

/// When `SYLLABIX_INDIC_MIO_WAV_DIR` is set, write 16 kHz WAVs for the
/// round-trip set (tooling; not part of default CI).
#[test]
fn write_roundtrip_wavs_when_dir_set() {
    skip_unless_model!("indic-mio");
    let Ok(dir) = std::env::var("SYLLABIX_INDIC_MIO_WAV_DIR") else {
        return;
    };
    let out = PathBuf::from(dir);
    std::fs::create_dir_all(&out).expect("wav dir");
    let cache = ModelCache::v0();
    let mut progress = StderrProgress::new();
    let cancel = Cancel::new();
    let backbone = cache
        .manifest()
        .asset(INDIC_MIO_ASSET)
        .expect("indic-mio asset")
        .clone();
    let decoder = cache
        .manifest()
        .asset(MIOCODEC_DECODER_ASSET)
        .expect("miocodec-decoder asset")
        .clone();
    let model_path = cache
        .resolve(&backbone, &HttpFetcher, &mut progress, &cancel)
        .expect("indic-mio GGUF");
    let decoder_path = cache
        .resolve(&decoder, &HttpFetcher, &mut progress, &cancel)
        .expect("miocodec decoder");
    let cases: &[(&str, &str)] = &[
        ("hi-01", "नमस्ते, आप कैसे हैं।"),
        ("hi-02", "आज मौसम बहुत अच्छा है।"),
        ("hi-03", "कृपया दरवाज़ा बंद कर दीजिए।"),
        ("hi-04", "मैं बाज़ार जा रहा हूँ।"),
        ("hi-05", "यह किताब मेरी है।"),
        ("hi-06", "गाड़ी आठ बजे छूटेगी।"),
        ("hi-07", "पानी गरम है, सावधान रहिए।"),
        ("hi-08", "बच्चे बगीचे में खेल रहे हैं।"),
        ("hi-09", "कृपया अपना नाम बताइए।"),
        ("hi-10", "कल सुबह मिलते हैं।"),
        ("ta-01", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),
        ("ta-02", "இன்று வானிலை நன்றாக உள்ளது."),
        ("ta-03", "தயவு செய்து கதவை மூடுங்கள்."),
        ("ta-04", "நான் கடைக்குப் போகிறேன்."),
        ("ta-05", "இந்தப் புத்தகம் என்னுடையது."),
        ("ta-06", "ரயில் எட்டு மணிக்குப் புறப்படும்."),
        ("ta-07", "தண்ணீர் சூடாக இருக்கிறது."),
        ("ta-08", "குழந்தைகள் தோட்டத்தில் விளையாடுகிறார்கள்."),
        ("ta-09", "தயவு செய்து உங்கள் பெயரைச் சொல்லுங்கள்."),
        ("ta-10", "நாளை காலை சந்திப்போம்."),
        ("bn-01", "নমস্কার, আপনি কেমন আছেন?"),
        ("bn-02", "আজ আবহাওয়া খুব ভালো।"),
        ("bn-03", "দয়া করে দরজাটা বন্ধ করুন।"),
        ("bn-04", "আমি বাজারে যাচ্ছি।"),
        ("bn-05", "এই বইটি আমার।"),
        ("bn-06", "ট্রেন আটটায় ছাড়বে।"),
        ("bn-07", "জল গরম, সাবধান।"),
        ("bn-08", "শিশুরা বাগানে খেলছে।"),
        ("mix-hi", "Please wait, मैं अभी आ रहा हूँ।"),
        ("mix-ta", "This is okay, நன்றி."),
        ("mix-bn", "Thank you, আবার দেখা হবে।"),
        ("mix-hi2", "Call me later, मैं व्यस्त हूँ।"),
        ("mix-ta2", "See you tomorrow, நன்றி நண்பா."),
    ];
    fn voice_for_case(name: &str) -> &'static str {
        if name.starts_with("ta-") || name.starts_with("mix-ta") {
            "ta-female"
        } else {
            "hi-female"
        }
    }
    for (name, text) in cases {
        let voice = voice_for_case(name);
        let mut tts = IndicMioTts::from_paths_with_seed(
            &model_path,
            &decoder_path,
            INDIC_MIO_TEST_SEED,
            TtsCompute::Cpu,
            voice,
        )
        .unwrap_or_else(|err| panic!("{name}: {err}"));
        let started = std::time::Instant::now();
        let chunks = tts
            .synthesize_chunk(&token(text, true), &Cancel::new())
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        let samples: Vec<i16> = chunks.into_iter().flat_map(|c| c.samples).collect();
        let path = out.join(format!("{name}.wav"));
        let n_samples = samples.len();
        let wav = syllabix_native::audio::WavPcm {
            format: syllabix_native::audio::PcmFormat::v0(),
            samples,
        };
        let file = std::fs::File::create(&path).expect("create wav");
        syllabix_native::audio::write_wav(file, &wav).expect("wav");
        eprintln!(
            "indic-mio wav {name}: {n_samples} samples in {:.3}s ({})",
            started.elapsed().as_secs_f64(),
            path.display()
        );
    }
}
