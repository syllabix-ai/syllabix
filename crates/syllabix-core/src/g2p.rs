//! English grapheme-to-phoneme for Kokoro IPA token ids.
//!
//! Uses a compiled CMUdict table (BSD-style cmusphinx dictionary) plus a small
//! letter-to-sound fallback. espeak-ng is not linked (GPL + extra dylib).

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::error::{Error, Result};

/// Kokoro pad id. Wrapped around every `input_ids` sequence.
pub const KOKORO_PAD_ID: i64 = 0;

/// Phoneme tokens excluding the two pad ids. Model context is 512.
pub const KOKORO_MAX_PHONEME_TOKENS: usize = 510;

const CMU_BIN: &[u8] = include_bytes!("../data/cmudict.bin");

const PHONES: &[&str] = &[
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
    "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH",
    "UW", "V", "W", "Y", "Z", "ZH",
];

/// hexgrad/Kokoro-82M `vocab` (Apache-2.0). `$` pad is [`KOKORO_PAD_ID`].
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

fn cmu_map() -> &'static HashMap<String, Vec<u8>> {
    static MAP: OnceLock<HashMap<String, Vec<u8>>> = OnceLock::new();
    MAP.get_or_init(parse_cmu)
}

fn parse_cmu() -> HashMap<String, Vec<u8>> {
    let mut bytes = CMU_BIN;
    assert!(bytes.starts_with(b"CMU1"), "cmudict magic");
    bytes = &bytes[4..];
    let count = u32::from_le_bytes(bytes[..4].try_into().expect("count")) as usize;
    bytes = &bytes[4..];
    let mut map = HashMap::with_capacity(count);
    for _ in 0..count {
        let word_len = bytes[0] as usize;
        let n_phones = bytes[1] as usize;
        bytes = &bytes[2..];
        let word = std::str::from_utf8(&bytes[..word_len])
            .expect("ascii word")
            .to_string();
        bytes = &bytes[word_len..];
        let phones = bytes[..n_phones].to_vec();
        bytes = &bytes[n_phones..];
        map.insert(word, phones);
    }
    map
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

/// IPA string for tests and debugging.
pub fn english_to_ipa(text: &str) -> String {
    let mut out = String::new();
    let mut word = String::new();
    let mut digits = String::new();
    let flush_word = |out: &mut String, word: &mut String| {
        if word.is_empty() {
            return;
        }
        let ipa = word_to_ipa(word);
        if !out.is_empty() && !out.ends_with(' ') && !ipa.is_empty() {
            out.push(' ');
        }
        out.push_str(&ipa);
        word.clear();
    };
    let flush_digits = |out: &mut String, word: &mut String, digits: &mut String| {
        if digits.is_empty() {
            return;
        }
        flush_word(out, word);
        for (i, d) in digits.chars().enumerate() {
            if (i > 0 || !out.is_empty()) && !out.ends_with(' ') {
                out.push(' ');
            }
            let name = digit_word(d);
            out.push_str(&word_to_ipa(name));
        }
        digits.clear();
    };

    for c in text.chars() {
        if c.is_ascii_alphabetic() || c == '\'' {
            flush_digits(&mut out, &mut word, &mut digits);
            word.push(c.to_ascii_lowercase());
            continue;
        }
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        flush_digits(&mut out, &mut word, &mut digits);
        flush_word(&mut out, &mut word);
        if c == '-' {
            continue;
        }
        if vocab_map().contains_key(&c) {
            out.push(c);
        } else if c.is_whitespace() && !out.ends_with(' ') && !out.is_empty() {
            out.push(' ');
        }
    }
    flush_digits(&mut out, &mut word, &mut digits);
    flush_word(&mut out, &mut word);
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

fn digit_word(d: char) -> &'static str {
    match d {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        '9' => "nine",
        _ => "zero",
    }
}

fn word_to_ipa(word: &str) -> String {
    let key = word.trim_matches('\'').to_ascii_lowercase();
    if key.is_empty() {
        return String::new();
    }
    if let Some(packed) = cmu_map().get(&key) {
        return packed_to_ipa(packed);
    }
    if let Some(stripped) = key.strip_suffix("'s") {
        if let Some(packed) = cmu_map().get(stripped) {
            let mut ipa = packed_to_ipa(packed);
            ipa.push('s');
            return ipa;
        }
    }
    guess_ipa(&key)
}

fn packed_to_ipa(packed: &[u8]) -> String {
    let mut ipa = String::new();
    for byte in packed {
        let idx = (*byte & 0x3f) as usize;
        let stress = byte >> 6;
        let Some(arpa) = PHONES.get(idx).copied() else {
            continue;
        };
        ipa.push_str(&arpa_to_ipa(arpa, stress));
    }
    ipa
}

fn arpa_to_ipa(arpa: &str, stress: u8) -> String {
    let body: String = match arpa {
        "AA" => "ɑ".into(),
        "AE" => "æ".into(),
        "AH" if stress == 0 => "ə".into(),
        "AH" => "ʌ".into(),
        "AO" => "ɔ".into(),
        "AW" => "aʊ".into(),
        "AY" => "aɪ".into(),
        "B" => "b".into(),
        "CH" => "ʧ".into(),
        "D" => "d".into(),
        "DH" => "ð".into(),
        "EH" => "ɛ".into(),
        "ER" => "ɚ".into(),
        "EY" => "eɪ".into(),
        "F" => "f".into(),
        "G" => "ɡ".into(),
        "HH" => "h".into(),
        "IH" => "ɪ".into(),
        "IY" => "i".into(),
        "JH" => "ʤ".into(),
        "K" => "k".into(),
        "L" => "l".into(),
        "M" => "m".into(),
        "N" => "n".into(),
        "NG" => "ŋ".into(),
        "OW" => "oʊ".into(),
        "OY" => "ɔɪ".into(),
        "P" => "p".into(),
        "R" => "ɹ".into(),
        "S" => "s".into(),
        "SH" => "ʃ".into(),
        "T" => "t".into(),
        "TH" => "θ".into(),
        "UH" => "ʊ".into(),
        "UW" => "u".into(),
        "V" => "v".into(),
        "W" => "w".into(),
        "Y" => "j".into(),
        "Z" => "z".into(),
        "ZH" => "ʒ".into(),
        _ => String::new(),
    };
    if body.is_empty() {
        return body;
    }
    let vowel = matches!(
        arpa,
        "AA" | "AE"
            | "AH"
            | "AO"
            | "AW"
            | "AY"
            | "EH"
            | "ER"
            | "EY"
            | "IH"
            | "IY"
            | "OW"
            | "OY"
            | "UH"
            | "UW"
    );
    if !vowel || stress == 0 {
        return body;
    }
    let mark = if stress == 1 { 'ˈ' } else { 'ˌ' };
    let mut out = String::new();
    out.push(mark);
    out.push_str(&body);
    out
}

fn guess_ipa(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    let mut i = 0;
    let mut packed = Vec::new();
    let push = |packed: &mut Vec<u8>, arpa: &str, stress: u8| {
        if let Some(idx) = PHONES.iter().position(|p| *p == arpa) {
            packed.push((idx as u8) | (stress << 6));
        }
    };
    while i < chars.len() {
        let rest: String = chars[i..].iter().collect();
        if rest.starts_with("tion") {
            push(&mut packed, "SH", 0);
            push(&mut packed, "AH", 0);
            push(&mut packed, "N", 0);
            i += 4;
            continue;
        }
        if rest.starts_with("ing") {
            push(&mut packed, "IH", 1);
            push(&mut packed, "NG", 0);
            i += 3;
            continue;
        }
        if rest.starts_with("ch") {
            push(&mut packed, "CH", 0);
            i += 2;
            continue;
        }
        if rest.starts_with("sh") {
            push(&mut packed, "SH", 0);
            i += 2;
            continue;
        }
        if rest.starts_with("th") {
            push(&mut packed, "TH", 0);
            i += 2;
            continue;
        }
        if rest.starts_with("ng") {
            push(&mut packed, "NG", 0);
            i += 2;
            continue;
        }
        if rest.starts_with("ph") {
            push(&mut packed, "F", 0);
            i += 2;
            continue;
        }
        if rest.starts_with("qu") {
            push(&mut packed, "K", 0);
            push(&mut packed, "W", 0);
            i += 2;
            continue;
        }
        if rest.starts_with("ee") {
            push(&mut packed, "IY", 1);
            i += 2;
            continue;
        }
        if rest.starts_with("oo") {
            push(&mut packed, "UW", 1);
            i += 2;
            continue;
        }
        match chars[i] {
            'a' => push(&mut packed, "AE", 1),
            'e' => push(&mut packed, "EH", 1),
            'i' => push(&mut packed, "IH", 1),
            'o' => push(&mut packed, "AA", 1),
            'u' => push(&mut packed, "AH", 1),
            'b' => push(&mut packed, "B", 0),
            'c' | 'k' => push(&mut packed, "K", 0),
            'd' => push(&mut packed, "D", 0),
            'f' => push(&mut packed, "F", 0),
            'g' => push(&mut packed, "G", 0),
            'h' => push(&mut packed, "HH", 0),
            'j' => push(&mut packed, "JH", 0),
            'l' => push(&mut packed, "L", 0),
            'm' => push(&mut packed, "M", 0),
            'n' => push(&mut packed, "N", 0),
            'p' => push(&mut packed, "P", 0),
            'q' => push(&mut packed, "K", 0),
            'r' => push(&mut packed, "R", 0),
            's' => push(&mut packed, "S", 0),
            't' => push(&mut packed, "T", 0),
            'v' => push(&mut packed, "V", 0),
            'w' => push(&mut packed, "W", 0),
            'x' => {
                push(&mut packed, "K", 0);
                push(&mut packed, "S", 0);
            }
            'y' => push(&mut packed, "Y", 0),
            'z' => push(&mut packed, "Z", 0),
            _ => {}
        }
        i += 1;
    }
    packed_to_ipa(&packed)
}

fn ipa_to_ids(ipa: &str) -> Vec<i64> {
    let map = vocab_map();
    ipa.chars()
        .filter_map(|c| map.get(&c).copied().map(i64::from))
        .collect()
}

/// Wrap phoneme ids with pad tokens for the ONNX graph.
pub fn pad_input_ids(ids: &[i64]) -> Vec<i64> {
    let mut out = Vec::with_capacity(ids.len() + 2);
    out.push(KOKORO_PAD_ID);
    out.extend_from_slice(ids);
    out.push(KOKORO_PAD_ID);
    out
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
        assert!(ipa.ends_with('.'), "{ipa}");
        let ids = english_to_kokoro_ids("Hello world.").unwrap();
        assert!(!ids.is_empty());
        assert!(ids.len() <= KOKORO_MAX_PHONEME_TOKENS);
        let padded = pad_input_ids(&ids);
        assert_eq!(padded.first().copied(), Some(0));
        assert_eq!(padded.last().copied(), Some(0));
    }

    #[test]
    fn unknown_words_still_emit_ids() {
        let ids = english_to_kokoro_ids("xyzzy").unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn cmu_table_loads() {
        assert!(cmu_map().len() > 100_000);
        assert!(cmu_map().contains_key("hello"));
        assert!(cmu_map().contains_key("world"));
    }

    #[test]
    fn digits_and_apostrophe_s_are_spoken() {
        let ipa = english_to_ipa("cat's 42-piece");
        assert!(!ipa.is_empty(), "{ipa}");
        let ids = english_to_kokoro_ids("cat's 42-piece").unwrap();
        assert!(!ids.is_empty());
    }

    #[test]
    fn letter_to_sound_covers_common_clusters() {
        let ipa = english_to_ipa("tioning shthphqueeoo xyz");
        assert!(!ipa.is_empty(), "{ipa}");
        assert!(english_to_kokoro_ids("tioning").is_ok());
    }

    #[test]
    fn dictionary_words_cover_varied_arpa_phones() {
        for word in [
            "choice", "the", "measure", "think", "sing", "out", "boy", "ship", "vision", "quick",
            "zero",
        ] {
            let ipa = english_to_ipa(word);
            assert!(!ipa.is_empty(), "{word} -> {ipa}");
        }
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
