//! Deterministic gate evaluation for the performance ledger (issue #45).
//!
//! Gates are recorded verdicts, not exceptions: the harness measures, the
//! ledger records, and humans (or the native merge-gate test) decide what a
//! failed verdict means. Every check here is pure and unit-tested without
//! weights.

use serde::Serialize;

use crate::tts::TTS_ASR_MIN_WORD_MATCH;

/// Verdicts for one completed turn.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TurnGateVerdicts {
    /// Measured in-order word match between prompt and STT transcript.
    pub stt_ratio: f64,
    /// `stt_ratio >= scenario.stt_min_match()`.
    pub stt_pass: bool,
    /// Assistant reply is non-blank.
    pub non_empty_reply: bool,
    /// Raw reply carries no `<think>` / `</think>` leak.
    pub no_think_leak: bool,
    /// Post-strip spoken text is free of Markdown artifacts.
    pub spoken_markdown_free: bool,
}

impl TurnGateVerdicts {
    /// All per-turn gates passed.
    pub fn passed(&self) -> bool {
        self.stt_pass && self.non_empty_reply && self.no_think_leak && self.spoken_markdown_free
    }
}

/// Verdicts that need the whole scenario (turn count, keywords, TTS→ASR).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ScenarioGateVerdicts {
    /// Turns the corpus expects.
    pub expected_turns: usize,
    /// Turns the loop actually completed.
    pub completed_turns: usize,
    /// `completed_turns == expected_turns`.
    pub turn_count_pass: bool,
    /// Keyword groups all matched somewhere in the replies.
    pub keywords_pass: bool,
    /// No forbidden keyword appeared in any reply.
    pub not_contains_pass: bool,
    /// Agent speech round-tripped through ASR: measured ratio.
    /// `None` when the scenario produced no agent audio (silence-only).
    pub tts_asr_ratio: Option<f64>,
    /// `tts_asr_ratio >= 0.8` when measured.
    pub tts_asr_pass: Option<bool>,
}

impl ScenarioGateVerdicts {
    /// All scenario-level gates passed.
    pub fn passed(&self) -> bool {
        self.turn_count_pass
            && self.keywords_pass
            && self.not_contains_pass
            && self.tts_asr_pass.unwrap_or(true)
    }
}

/// In-order word-match ratio between a spoken prompt and its transcript.
pub fn stt_match_ratio(prompt: &str, transcript: &str) -> f64 {
    let expected = normalize_number_words(prompt);
    let got = normalize_number_words(transcript);
    let mut index = 0usize;
    let mut matched = 0usize;
    for word in &expected {
        if let Some(found) = got[index..].iter().position(|candidate| candidate == word) {
            index += found + 1;
            matched += 1;
        }
    }
    if expected.is_empty() {
        1.0
    } else {
        matched as f64 / expected.len() as f64
    }
}

/// Whisper legitimately renders spoken numbers as digits (`twenty` → `20`),
/// and [`transcript_words`] drops non-alphabetic tokens, so an exact numeral
/// capture would score as a miss. Expand number tokens to their spoken words
/// on BOTH sides before scoring. Eval-only; the product STT contract and the
/// shared scorer are untouched.
fn normalize_number_words(text: &str) -> Vec<String> {
    const WORDS: [(&str, &str); 22] = [
        ("0", "zero"),
        ("1", "one"),
        ("2", "two"),
        ("3", "three"),
        ("4", "four"),
        ("5", "five"),
        ("6", "six"),
        ("7", "seven"),
        ("8", "eight"),
        ("9", "nine"),
        ("10", "ten"),
        ("11", "eleven"),
        ("12", "twelve"),
        ("13", "thirteen"),
        ("14", "fourteen"),
        ("15", "fifteen"),
        ("16", "sixteen"),
        ("17", "seventeen"),
        ("18", "eighteen"),
        ("19", "nineteen"),
        ("20", "twenty"),
        ("100", "hundred"),
    ];
    const TENS: [(&str, &str); 8] = [
        ("2", "twenty"),
        ("3", "thirty"),
        ("4", "forty"),
        ("5", "fifty"),
        ("6", "sixty"),
        ("7", "seventy"),
        ("8", "eighty"),
        ("9", "ninety"),
    ];

    fn spoken(token: &str) -> Option<Vec<String>> {
        if !token.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        for (digit, word) in WORDS {
            if token == digit {
                return Some(vec![word.to_string()]);
            }
        }
        for (tens_digit, word) in TENS {
            if token == format!("{tens_digit}0") {
                return Some(vec![word.to_string()]);
            }
        }
        // Compound ("25" → "twenty five"): tens word + ones word.
        if token.len() == 2 {
            let n: i32 = token.parse().ok()?;
            if (21..99).contains(&n) && n % 10 != 0 {
                let mut words = spoken(&format!("{}", (n / 10) * 10))?;
                words.extend(spoken(&format!("{}", n % 10))?);
                return Some(words);
            }
        }
        None
    }

    // Split on non-alphanumerics so digit tokens survive to the mapping
    // (`transcript_words` would drop them before `spoken` could expand them).
    let mut out = Vec::new();
    for token in text.split(|c: char| !c.is_ascii_alphanumeric()) {
        let token = token.to_ascii_lowercase();
        if token.is_empty() {
            continue;
        }
        match spoken(&token) {
            Some(expanded) => out.extend(expanded),
            None => {
                if token
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic())
                {
                    out.push(token);
                }
            }
        }
    }
    out
}

/// Evaluate every per-turn gate from raw pipeline output.
///
/// `assistant_text` is the raw LLM reply; `speak_text` is what TTS actually
/// received (think-strip → markdown-strip), so `spoken_markdown_free` audits
/// the strip contract end to end.
pub fn turn_verdicts(
    prompt: &str,
    assistant_text: &str,
    speak_text: &str,
    stt_text: &str,
    stt_min_match: f64,
) -> TurnGateVerdicts {
    let stt_ratio = stt_match_ratio(prompt, stt_text);
    TurnGateVerdicts {
        stt_ratio,
        stt_pass: stt_ratio + f64::EPSILON >= stt_min_match,
        non_empty_reply: !assistant_text.trim().is_empty(),
        no_think_leak: !contains_think_tag(assistant_text) && !contains_think_tag(speak_text),
        spoken_markdown_free: !spoken_has_markdown(speak_text),
    }
}

/// True when a think tag survives anywhere in the text.
pub fn contains_think_tag(text: &str) -> bool {
    text.contains("<think>") || text.contains("</think>")
}

/// True when Markdown artifacts survive into spoken text (`**`, backticks,
/// heading or bullet lines). This is the HF S2S #338 regression detector.
pub fn spoken_has_markdown(speak_text: &str) -> bool {
    if speak_text.contains("**") || speak_text.contains('`') {
        return true;
    }
    speak_text.lines().any(|line| {
        let line = line.trim_start();
        line.starts_with("# ") || line.starts_with("- ")
    })
}

/// Case-insensitive substring containment over every reply.
pub fn replies_contain_any(replies: &[&str], group: &[String]) -> bool {
    let lowered: Vec<String> = replies.iter().map(|r| r.to_lowercase()).collect();
    group.iter().any(|keyword| {
        lowered
            .iter()
            .any(|reply| reply.contains(&keyword.to_lowercase()))
    })
}

/// True when none of the forbidden keywords appear in any reply.
pub fn replies_contain_none(replies: &[&str], forbidden: &[String]) -> bool {
    forbidden
        .iter()
        .all(|keyword| !replies_contain_any(replies, std::slice::from_ref(keyword)))
}

/// Evaluate scenario-level gates.
///
/// `asr_ratio` is `None` when there was no agent audio to round-trip
/// (silence-only scenarios); the gate passes vacuously in that case.
pub fn scenario_verdicts(
    expected_turns: usize,
    completed_turns: usize,
    replies: &[&str],
    expect: &crate::eval::scenario::Expectation,
    asr_ratio: Option<f64>,
) -> ScenarioGateVerdicts {
    let keywords_pass = expect
        .reply_contains_any
        .iter()
        .all(|group| replies_contain_any(replies, group));
    let not_contains_pass = replies_contain_none(replies, &expect.reply_not_contains);
    ScenarioGateVerdicts {
        expected_turns,
        completed_turns,
        turn_count_pass: completed_turns == expected_turns,
        keywords_pass,
        not_contains_pass,
        tts_asr_ratio: asr_ratio,
        tts_asr_pass: asr_ratio.map(|ratio| ratio + f64::EPSILON >= TTS_ASR_MIN_WORD_MATCH),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_ratio_is_in_order_and_case_insensitive() {
        let ratio = stt_match_ratio("Hello! What can you do?", "hello what can you do?");
        assert!((ratio - 1.0).abs() < f64::EPSILON);

        // Extra hypothesis words are ignored; order still matters.
        assert!(stt_match_ratio("blue sky", "the sky is blue today") < 1.0);
        assert_eq!(stt_match_ratio("", "anything"), 1.0);
    }

    #[test]
    fn digit_transcripts_match_spoken_numbers() {
        // Whisper renders spoken numbers as digits; both sides must agree.
        let ratio = stt_match_ratio(
            "What is twenty plus thirty?",
            "hey what is 20 plus 30 exactly",
        );
        assert!(ratio >= 0.8, "digit capture must score: {ratio}");
        // Compounds expand on both sides.
        let ratio = stt_match_ratio("What is five times five?", "is 5 times 5");
        assert!((ratio - 0.8).abs() < 1e-9, "4/5 expected words: {ratio}");
        let ratio = stt_match_ratio("twenty five", "25");
        assert!((ratio - 1.0).abs() < 1e-9, "{ratio}");
    }

    #[test]
    fn turn_verdicts_flag_each_failure_class() {
        // STT mismatch: one of six prompt words missing → 5/6 ≈ 0.83, below a
        // 1.0 floor but above the default 0.8 (boundary covered separately).
        let v = turn_verdicts(
            "one two three four five six",
            "reply",
            "reply",
            "one two xxx four five six",
            1.0,
        );
        assert!(!v.stt_pass);
        assert!(v.non_empty_reply && v.no_think_leak && v.spoken_markdown_free);

        // Blank reply.
        let v = turn_verdicts("p", "   ", "", "p", 0.8);
        assert!(!v.non_empty_reply);

        // Markdown survivor in spoken text.
        let v = turn_verdicts("p", "**bold**", "**bold**", "p", 0.8);
        assert!(!v.spoken_markdown_free);

        // Everything clean passes end to end.
        let v = turn_verdicts(
            "what is your name",
            "I am Syllabix.",
            "I am Syllabix.",
            "what is your name",
            0.8,
        );
        assert!(v.passed(), "{v:?}");
    }

    #[test]
    fn think_leak_is_detected_in_raw_or_speak_text() {
        let raw = turn_verdicts("p", "<think>chain</think>Answer", "Answer", "p", 0.8);
        assert!(!raw.no_think_leak);
        let speak = turn_verdicts("p", "ok", "<think>leak", "p", 0.8);
        assert!(!speak.no_think_leak);
        let clean = turn_verdicts("p", "thinking about it openly", "thinking", "p", 0.8);
        assert!(clean.no_think_leak);
    }

    #[test]
    fn markdown_survivors_are_detected() {
        assert!(spoken_has_markdown("bold **text** stays"));
        assert!(spoken_has_markdown("code `x` stays"));
        assert!(spoken_has_markdown("# Heading\nline"));
        assert!(spoken_has_markdown("- bullet"));
        assert!(!spoken_has_markdown("plain spoken words."));
        assert!(!spoken_has_markdown("hash symbol mid sentence stays fine"));
    }

    #[test]
    fn keyword_groups_need_every_group_matched() {
        let replies = vec!["Hi there! Jupiter is huge."];
        assert!(replies_contain_any(
            &replies,
            &["hello".to_string(), "hi".to_string()]
        ));
        let both = vec![
            vec!["jupiter".to_string()],
            vec!["huge".to_string(), "big".to_string()],
        ];
        let verdicts = scenario_verdicts(
            1,
            1,
            &replies,
            &crate::eval::scenario::Expectation {
                reply_contains_any: both,
                ..Default::default()
            },
            None,
        );
        assert!(verdicts.keywords_pass);

        let miss = vec![vec!["mars".to_string()]];
        let verdicts = scenario_verdicts(
            1,
            1,
            &replies,
            &crate::eval::scenario::Expectation {
                reply_contains_any: miss,
                ..Default::default()
            },
            None,
        );
        assert!(!verdicts.keywords_pass);
    }

    #[test]
    fn forbidden_keywords_fail_the_gate() {
        let replies = vec!["I would say **bold** things"];
        let verdicts = scenario_verdicts(
            1,
            1,
            &replies,
            &crate::eval::scenario::Expectation {
                reply_not_contains: vec!["**".to_string()],
                ..Default::default()
            },
            None,
        );
        assert!(!verdicts.not_contains_pass);
        assert!(!verdicts.passed());
    }

    #[test]
    fn turn_count_mismatch_fails() {
        let verdicts = scenario_verdicts(2, 1, &["fine"], &Default::default(), None);
        assert!(!verdicts.turn_count_pass);
        assert!(!verdicts.passed());
    }

    #[test]
    fn tts_asr_gate_uses_the_shared_floor() {
        let verdicts = scenario_verdicts(1, 1, &["ok"], &Default::default(), Some(0.5));
        assert_eq!(verdicts.tts_asr_pass, Some(false));
        let verdicts = scenario_verdicts(1, 1, &["ok"], &Default::default(), Some(0.9));
        assert_eq!(verdicts.tts_asr_pass, Some(true));
        // Silence-only: no agent audio, gate passes vacuously.
        let verdicts = scenario_verdicts(0, 0, &[], &Default::default(), None);
        assert!(verdicts.passed());
    }

    #[test]
    fn boundary_thresholds_pass_with_epsilon() {
        let verdicts = turn_verdicts(
            "one two three four",
            "r",
            "r",
            "one two three four five six seven eight",
            1.0,
        );
        // 4/4 expected words found in order → ratio 1.0 passes even with extra noise.
        assert!(verdicts.stt_pass);
        let ratio_08 = turn_verdicts("a b c d e f g h i j", "r", "r", "a b c d e f g h x y", 0.8);
        assert!(
            (ratio_08.stt_ratio - 0.8).abs() < 1e-9,
            "{}",
            ratio_08.stt_ratio
        );
        assert!(ratio_08.stt_pass, "exactly-at-threshold must pass");
    }
}
