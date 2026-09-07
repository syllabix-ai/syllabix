//! English grapheme-to-phoneme for Kokoro phoneme token ids.
//!
//! Uses the Misaki US lexicon and rules ([hexgrad/misaki](https://github.com/hexgrad/misaki))
//! via the in-process `misaki-rs` port. That is the G2P Kokoro-82M was trained
//! on. espeak-ng is not linked (GPL + extra dylib).

use std::collections::HashMap;
use std::sync::OnceLock;

use misaki_rs::{Language, G2P};

use crate::error::{Error, Result};

const APOSTROPHES: &[char] = &[
    '\u{2018}', '\u{2019}', '\u{02BC}', '\u{02B9}', '\u{00B4}', '`',
];

/// Phoneme tokens excluding the two pad ids. Model context is 512.
pub const KOKORO_MAX_PHONEME_TOKENS: usize = 510;

/// hexgrad/Kokoro-82M `vocab` (Apache-2.0).
const KOKORO_VOCAB: &[(char, u16)] = &[
    (';', 1),
    (':', 2),
    (',', 3),
    ('.', 4),
    ('!', 5),
    ('?', 6),
    ('\u{2014}', 9),
    ('\u{2026}', 10),
    ('"', 11),
    ('(', 12),
    (')', 13),
    ('\u{201c}', 14),
    ('\u{201d}', 15),
    (' ', 16),
    ('\u{303}', 17),
    ('\u{2a3}', 18),
    ('\u{2a5}', 19),
    ('\u{2a6}', 20),
    ('\u{2a8}', 21),
    ('\u{1d5d}', 22),
    ('\u{ab67}', 23),
    ('A', 24),
    ('I', 25),
    ('O', 31),
    ('Q', 33),
    ('S', 35),
    ('T', 36),
    ('W', 39),
    ('Y', 41),
    ('\u{1d4a}', 42),
    ('a', 43),
    ('b', 44),
    ('c', 45),
    ('d', 46),
    ('e', 47),
    ('f', 48),
    ('h', 50),
    ('i', 51),
    ('j', 52),
    ('k', 53),
    ('l', 54),
    ('m', 55),
    ('n', 56),
    ('o', 57),
    ('p', 58),
    ('q', 59),
    ('r', 60),
    ('s', 61),
    ('t', 62),
    ('u', 63),
    ('v', 64),
    ('w', 65),
    ('x', 66),
    ('y', 67),
    ('z', 68),
    ('\u{251}', 69),
    ('\u{250}', 70),
    ('\u{252}', 71),
    ('\u{e6}', 72),
    ('\u{3b2}', 75),
    ('\u{254}', 76),
    ('\u{255}', 77),
    ('\u{e7}', 78),
    ('\u{256}', 80),
    ('\u{f0}', 81),
    ('\u{2a4}', 82),
    ('\u{259}', 83),
    ('\u{25a}', 85),
    ('\u{25b}', 86),
    ('\u{25c}', 87),
    ('\u{25f}', 90),
    ('\u{261}', 92),
    ('\u{265}', 99),
    ('\u{268}', 101),
    ('\u{26a}', 102),
    ('\u{29d}', 103),
    ('\u{26f}', 110),
    ('\u{270}', 111),
    ('\u{14b}', 112),
    ('\u{273}', 113),
    ('\u{272}', 114),
    ('\u{274}', 115),
    ('\u{f8}', 116),
    ('\u{278}', 118),
    ('\u{3b8}', 119),
    ('\u{153}', 120),
    ('\u{279}', 123),
    ('\u{27e}', 125),
    ('\u{27b}', 126),
    ('\u{281}', 128),
    ('\u{27d}', 129),
    ('\u{282}', 130),
    ('\u{283}', 131),
    ('\u{288}', 132),
    ('\u{2a7}', 133),
    ('\u{28a}', 135),
    ('\u{28b}', 136),
    ('\u{28c}', 138),
    ('\u{263}', 139),
    ('\u{264}', 140),
    ('\u{3c7}', 142),
    ('\u{28e}', 143),
    ('\u{292}', 147),
    ('\u{294}', 148),
    ('\u{2c8}', 156),
    ('\u{2cc}', 157),
    ('\u{2d0}', 158),
    ('\u{2b0}', 162),
    ('\u{2b2}', 164),
    ('\u{2193}', 169),
    ('\u{2192}', 171),
    ('\u{2197}', 172),
    ('\u{2198}', 173),
    ('\u{1d7b}', 177),
];

fn vocab_map() -> &'static HashMap<char, u16> {
    static MAP: OnceLock<HashMap<char, u16>> = OnceLock::new();
    MAP.get_or_init(|| KOKORO_VOCAB.iter().copied().collect())
}

fn engine() -> &'static G2P {
    static ENGINE: OnceLock<G2P> = OnceLock::new();
    ENGINE.get_or_init(|| G2P::new(Language::EnglishUS))
}

/// Convert spoken English into Kokoro `input_ids` without pad tokens.
pub fn english_to_kokoro_ids(text: &str) -> Result<Vec<i64>> {
    let ipa = english_to_ipa(text);
    let ids = ipa_to_ids(&ipa);
    if ids.len() > KOKORO_MAX_PHONEME_TOKENS {
        return Err(Error::Provider {
            provider: "kokoro",
            message: format!(
                "phoneme sequence length {} exceeds {KOKORO_MAX_PHONEME_TOKENS}",
                ids.len()
            ),
        });
    }
    Ok(ids)
}

/// Misaki phoneme string for tests and debugging.
pub fn english_to_ipa(text: &str) -> String {
    let text = fold_apostrophes(text);
    let mut out = String::new();
    let mut word = String::new();
    let mut digits = String::new();
    for c in text.chars() {
        if c.is_ascii_alphabetic() || c == '\'' {
            flush_digits(&mut out, &mut digits);
            word.push(c);
            continue;
        }
        if c.is_ascii_digit() {
            flush_word(&mut out, &mut word);
            digits.push(c);
            continue;
        }
        flush_digits(&mut out, &mut digits);
        flush_word(&mut out, &mut word);
        if c == '-' {
            continue;
        }
        if vocab_map().contains_key(&c) {
            out.push(c);
        } else if c.is_whitespace() && !out.is_empty() && !out.ends_with(' ') {
            out.push(' ');
        }
    }
    flush_digits(&mut out, &mut digits);
    flush_word(&mut out, &mut word);
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

fn fold_apostrophes(text: &str) -> String {
    text.chars()
        .map(|c| if APOSTROPHES.contains(&c) { '\'' } else { c })
        .collect()
}

fn flush_word(out: &mut String, word: &mut String) {
    if word.is_empty() {
        return;
    }
    push_ps(out, &token_to_ps(word));
    word.clear();
}

fn flush_digits(out: &mut String, digits: &mut String) {
    if digits.is_empty() {
        return;
    }
    push_ps(out, &token_to_ps(digits));
    digits.clear();
}

fn push_ps(out: &mut String, ps: &str) {
    if ps.is_empty() {
        return;
    }
    if !out.is_empty() && !out.ends_with(' ') {
        out.push(' ');
    }
    out.push_str(ps);
}

fn token_to_ps(token: &str) -> String {
    let tag = if token.chars().any(|c| c.is_alphabetic())
        && token
            .chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| c.is_uppercase())
    {
        "NNP"
    } else {
        "NN"
    };
    let stress = if token.chars().any(|c| c.is_uppercase()) {
        Some(
            if token
                .chars()
                .filter(|c| c.is_alphabetic())
                .all(|c| c.is_uppercase())
            {
                2.0
            } else {
                0.5
            },
        )
    } else {
        None
    };
    if let Some((ps, _)) = engine().lexicon.get_word(token, tag, stress, None) {
        return collapse_ws(&to_misaki_us(&ps));
    }
    let lower = token.to_ascii_lowercase();
    if lower != token {
        if let Some((ps, _)) = engine().lexicon.get_word(&lower, "NN", None, None) {
            return collapse_ws(&to_misaki_us(&ps));
        }
    }
    let raw = match engine().g2p(token) {
        Ok((phonemes, _)) => phonemes,
        Err(_) => String::new(),
    };
    collapse_ws(&to_misaki_us(&raw))
}

/// Map espeak-style IPA (including ZWJ ties from `misaki-rs`) onto the
/// hexgrad Misaki alphabet Kokoro-82M was trained on.
fn to_misaki_us(ps: &str) -> String {
    let mut s = ps.replace(['\u{200d}', '\u{361}'], "^");
    const PAIRS: &[(&str, &str)] = &[
        ("a^ɪ", "I"),
        ("aɪ", "I"),
        ("a^ʊ", "W"),
        ("aʊ", "W"),
        ("e^ɪ", "A"),
        ("eɪ", "A"),
        ("ɔ^ɪ", "Y"),
        ("ɔɪ", "Y"),
        ("o^ʊ", "O"),
        ("oʊ", "O"),
        ("d^ʒ", "ʤ"),
        ("dʒ", "ʤ"),
        ("t^ʃ", "ʧ"),
        ("tʃ", "ʧ"),
        ("ə^l", "ᵊl"),
        ("ʔn", "tᵊn"),
        ("ɚ", "əɹ"),
        ("ɐ", "ə"),
        ("ʔ", "t"),
        ("ɾ", "T"),
    ];
    for (old, new) in PAIRS {
        s = s.replace(old, new);
    }
    s = s.replace(['ː', '^', '❓'], "");
    s.replace('r', "ɹ")
}

fn collapse_ws(phonemes: &str) -> String {
    let mut out = String::with_capacity(phonemes.len());
    let mut last_space = true;
    for c in phonemes.chars() {
        if c.is_whitespace() {
            if !last_space && !out.is_empty() {
                out.push(' ');
                last_space = true;
            }
            continue;
        }
        out.push(c);
        last_space = false;
    }
    out
}

fn ipa_to_ids(ipa: &str) -> Vec<i64> {
    let map = vocab_map();
    ipa.chars()
        .filter_map(|c| map.get(&c).copied().map(i64::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_world_maps_to_vocab_ids() {
        let ipa = english_to_ipa("Hello world.");
        assert!(
            ipa.contains('h') || ipa.contains('ə') || ipa.contains('ˈ'),
            "{ipa}"
        );
        assert!(ipa.contains('O'), "Kokoro diphthong O missing: {ipa}");
        let ids = english_to_kokoro_ids("Hello world.").unwrap();
        assert!(!ids.is_empty());
        assert!(ids.len() <= KOKORO_MAX_PHONEME_TOKENS);
        assert!(!ids.is_empty());
    }

    #[test]
    fn unknown_words_still_emit_ids() {
        let ids = english_to_kokoro_ids("xyzzy").unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn contractions_match_misaki_gold() {
        assert_eq!(english_to_ipa("can't").replace(' ', ""), "kˈænt");
        assert_eq!(english_to_ipa("can’t").replace(' ', ""), "kˈænt");
        let dont = english_to_ipa("don't").replace(' ', "");
        assert_eq!(dont, "dˈOnt", "{dont}");
        let im = english_to_ipa("I'm").replace(' ', "");
        assert!(im.contains('I'), "expected Misaki I diphthong: {im}");
        assert!(!im.contains("tˈi"), "must not speak letter T: {im}");
    }

    #[test]
    fn digits_and_apostrophe_s_are_spoken() {
        let ipa = english_to_ipa("cat's 42-piece");
        assert!(!ipa.is_empty(), "{ipa}");
        let ids = english_to_kokoro_ids("cat's 42-piece").unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn dictionary_words_cover_varied_arpa_phones() {
        for word in [
            "choice", "the", "measure", "think", "sing", "out", "boy", "ship", "vision", "quick",
            "zero",
        ] {
            let ipa = english_to_ipa(word);
            assert!(!ipa.is_empty(), "{word} -> {ipa}");
            assert!(!ipa.contains('❓'), "{word} -> {ipa}");
        }
        let boy = english_to_ipa("boy").replace(' ', "");
        assert!(boy.contains('Y'), "expected Misaki Y diphthong: {boy}");
    }

    #[test]
    fn overlong_phoneme_sequence_is_a_provider_error() {
        let long = "hello ".repeat(400);
        let err = english_to_kokoro_ids(&long).unwrap_err();
        assert!(matches!(
            err,
            Error::Provider {
                provider: "kokoro",
                ..
            }
        ));
        assert!(err.to_string().contains("exceeds"));
    }
}
