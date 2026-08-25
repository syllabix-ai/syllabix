//! Frozen question audio for the deterministic performance corpus.
//!
//! Every clip was rendered once with Qwen3-TTS 1.7B, sampler seed 42,
//! English, and the runtime's fixed self-voice anchor. Benchmark profiles
//! consume these exact bytes; the selected pipeline TTS never generates its
//! own microphone input.

use std::io::Cursor;

use sha2::{Digest, Sha256};

use crate::audio::{read_wav, WavPcm};
use crate::error::{Error, Result};

/// Generator model used for every frozen question clip.
pub const GENERATOR_MODEL: &str = "qwen3-1.7";
/// Pinned semantic-token sampler seed used for every clip.
pub const GENERATOR_SEED: u32 = 42;
/// Voice conditioning used by the shipped Qwen runtime.
pub const VOICE_ANCHOR: &str = "built-in fixed self-voice anchor";

/// One embedded, hash-pinned question recording.
#[derive(Debug, Clone, Copy)]
pub struct AudioFixture {
    /// Stable corpus id (`scenario_turn_N`).
    pub id: &'static str,
    /// Exact text used to render the clip.
    pub prompt: &'static str,
    /// SHA-256 of the canonical PCM16 WAV bytes.
    pub sha256: &'static str,
    /// Embedded WAV bytes; no checkout or model download is needed to read them.
    pub wav: &'static [u8],
}

macro_rules! fixture {
    ($id:literal, $prompt:literal, $sha256:literal, $file:literal) => {
        AudioFixture {
            id: $id,
            prompt: $prompt,
            sha256: $sha256,
            wav: include_bytes!(concat!("fixtures/", $file)),
        }
    };
}

/// Complete spoken-input corpus: nine questions across five scenarios.
pub const AUDIO_FIXTURES: &[AudioFixture] = &[
    fixture!(
        "greeting_001_turn_1",
        "Hi, let us start with something simple: what can you do?",
        "2cc5cdc204f54597320ba5f0410c7ce26bbd1f965fb1daaa9bf698d7591eb139",
        "greeting_001_turn_1.wav"
    ),
    fixture!(
        "greeting_001_turn_2",
        "Tell me something interesting about yourself.",
        "514ae34bcce73d5945ead463a497c6e7aa5c96884cd96f557f3ce2d7cb274efe",
        "greeting_001_turn_2.wav"
    ),
    fixture!(
        "space_fact_001_turn_1",
        "Tell me a short fun fact about space please.",
        "65ad19697b793f7cd68b4e0dc163ee29015613db8e2c60b056965cccdf1215cd",
        "space_fact_001_turn_1.wav"
    ),
    fixture!(
        "space_fact_001_turn_2",
        "Tell me which planet is the biggest one in our solar system.",
        "9800b9d284b4748478d1bb95f98a218eff90840c87fc58db571282d0d2f0fd71",
        "space_fact_001_turn_2.wav"
    ),
    fixture!(
        "arithmetic_001_turn_1",
        "Hey, what is twenty plus thirty exactly?",
        "9d9b10150521238e98fdd38b5276836eba2a8a3a81510175e7d54947d5ae86d7",
        "arithmetic_001_turn_1.wav"
    ),
    fixture!(
        "arithmetic_001_turn_2",
        "What is the answer when you multiply five by five?",
        "16b95b8d7e9b3d7e8593183d3871624e4e850353d111023b30994cb017112821",
        "arithmetic_001_turn_2.wav"
    ),
    fixture!(
        "history_001_turn_1",
        "My favorite color is blue. Please remember that forever.",
        "70997dba2a57dbdb77fb03463266a93a0de27b1ecc79d3095e95c71bd6aaedc7",
        "history_001_turn_1.wav"
    ),
    fixture!(
        "history_001_turn_2",
        "So tell me, what is my favorite color again?",
        "dc8423d22f950a3f323a89c6a65da904bef853ff888246eaf463238f12a1cf85",
        "history_001_turn_2.wav"
    ),
    fixture!(
        "long_sentence_001_turn_1",
        "I am testing how you handle a longer sentence so here is one that keeps going for quite a while before asking how you are doing today",
        "1941e0a4ee6710e1f11e0aae1fa5aeba0de3e4d2b243503297c0dd9929250184",
        "long_sentence_001_turn_1.wav"
    ),
];

/// Find one recording by its stable corpus id.
pub fn audio_fixture(id: &str) -> Result<&'static AudioFixture> {
    AUDIO_FIXTURES
        .iter()
        .find(|fixture| fixture.id == id)
        .ok_or_else(|| Error::Config {
            field: "eval.scenarios.audio".into(),
            message: format!("unknown audio fixture {id:?}"),
        })
}

/// Verify the embedded hash and decode one canonical PCM16 WAV.
pub fn decode_verified(fixture: &AudioFixture) -> Result<WavPcm> {
    let actual = hex(Sha256::digest(fixture.wav));
    if actual != fixture.sha256 {
        return Err(Error::InvalidAudio {
            message: format!(
                "audio fixture {:?} SHA-256 mismatch: expected {}, got {actual}",
                fixture.id, fixture.sha256
            ),
        });
    }
    read_wav(Cursor::new(fixture.wav))
}

fn hex(bytes: impl AsRef<[u8]>) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn corpus_is_complete_unique_and_canonical() {
        assert_eq!(AUDIO_FIXTURES.len(), 9);
        let mut ids = BTreeSet::new();
        for fixture in AUDIO_FIXTURES {
            assert!(ids.insert(fixture.id), "duplicate {}", fixture.id);
            assert!(!fixture.prompt.trim().is_empty());
            let wav = decode_verified(fixture).expect("hash-pinned WAV");
            assert_eq!(wav.format.sample_rate_hz, 16_000, "{}", fixture.id);
            assert_eq!(wav.format.channels, 1, "{}", fixture.id);
            assert!(
                wav.samples.iter().any(|sample| sample.abs() > 32),
                "{} must contain speech energy",
                fixture.id
            );
        }
    }

    #[test]
    fn unknown_fixture_fails_fast() {
        let err = audio_fixture("missing").expect_err("unknown fixture");
        assert!(err.to_string().contains("unknown audio fixture"), "{err}");
    }
}
