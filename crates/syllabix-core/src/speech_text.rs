//! Think-tag strip, Markdown-to-speech cleanup, and sentence chunking for TTS.

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

/// Abbreviations that should not split a sentence on the following period.
const ABBREV: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "vs", "st", "gen", "col", "sgt", "lt", "gov",
    "inc", "ltd", "eg", "ie", "am", "pm", "us", "usa",
];

/// Remove `<think>…</think>` (and leftover / unclosed think tags) so Kokoro
/// speaks the answer, not the chain of thought.
pub fn strip_think_for_speech(text: &str) -> String {
    let mut filter = ThinkFilter::default();
    collapse_ws(&filter.push(text, true))
}

/// Think-strip, then Markdown-strip. Used by TTS and the diagnostics speak-text.
pub fn speak_text_for_tts(text: &str) -> String {
    strip_markdown_for_speech(&strip_think_for_speech(text))
}

/// Streaming filter: hold tokens inside an open think block; emit the rest.
#[derive(Debug, Default)]
pub struct ThinkFilter {
    in_think: bool,
    pending: String,
}

impl ThinkFilter {
    /// True while tokens are inside an open `<think>` block.
    pub fn in_think(&self) -> bool {
        self.in_think
    }

    /// Append `chunk`. When `flush` is set, drop an unclosed think tail.
    pub fn push(&mut self, chunk: &str, flush: bool) -> String {
        self.pending.push_str(chunk);
        let mut out = String::new();
        loop {
            if self.in_think {
                if let Some(i) = self.pending.find(THINK_CLOSE) {
                    self.pending = self.pending[i + THINK_CLOSE.len()..].to_string();
                    self.in_think = false;
                    continue;
                }
                if flush {
                    self.pending.clear();
                    self.in_think = false;
                } else {
                    let held = incomplete_tag_suffix(&self.pending, THINK_CLOSE);
                    let keep_from = self.pending.len() - held.len();
                    self.pending = self.pending[keep_from..].to_string();
                }
                break;
            }
            if let Some(i) = self.pending.find(THINK_OPEN) {
                out.push_str(&self.pending[..i]);
                self.pending = self.pending[i + THINK_OPEN.len()..].to_string();
                self.in_think = true;
                continue;
            }
            if flush {
                out.push_str(&self.pending);
                self.pending.clear();
            } else {
                // Closing tags can arrive split across decoded token pieces
                // even when no opening tag was observed. Hold either tag so a
                // stray `</think>` cannot leak into the transcript or TTS.
                let open = incomplete_tag_suffix(&self.pending, THINK_OPEN);
                let close = incomplete_tag_suffix(&self.pending, THINK_CLOSE);
                let held = if close.len() > open.len() {
                    close
                } else {
                    open
                };
                let emit_end = self.pending.len() - held.len();
                out.push_str(&self.pending[..emit_end]);
                self.pending = self.pending[emit_end..].to_string();
            }
            break;
        }
        out.replace(THINK_CLOSE, "")
    }
}

fn incomplete_tag_suffix<'a>(s: &'a str, tag: &str) -> &'a str {
    let max = tag.len().saturating_sub(1);
    if max == 0 || s.is_empty() {
        return "";
    }
    let start = s
        .char_indices()
        .rev()
        .take_while(|(i, _)| s.len() - i <= max)
        .map(|(i, _)| i)
        .last()
        .unwrap_or(s.len());
    for (i, _) in s[start..].char_indices() {
        let idx = start + i;
        if tag.starts_with(&s[idx..]) {
            return &s[idx..];
        }
    }
    ""
}

/// Strip headings, list markers, emphasis, and link markup so TTS does not
/// speak punctuation from Markdown.
pub fn strip_markdown_for_speech(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let spoken = strip_line(line);
        if spoken.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&spoken);
    }
    collapse_ws(&out)
}

fn strip_line(line: &str) -> String {
    let mut s = line.trim();
    while s.starts_with('#') {
        s = s[1..].trim_start();
    }
    s = strip_list_marker(s);
    collapse_ws(&strip_inline(s))
}

fn strip_list_marker(line: &str) -> &str {
    let s = line.trim_start();
    for prefix in ["- ", "* ", "+ "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return rest;
        }
    }
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i > 0 && i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1] == b' ' {
        return s[i + 2..].trim_start();
    }
    s
}

fn strip_inline(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '`' {
            i += 1;
            while i < chars.len() && chars[i] != '`' {
                out.push(chars[i]);
                i += 1;
            }
            if i < chars.len() {
                i += 1;
            }
            continue;
        }
        if c == '[' {
            if let Some((label, next)) = take_markdown_link(&chars, i) {
                out.push_str(&label);
                i = next;
                continue;
            }
        }
        if c == '*' || c == '_' {
            let run = count_run(&chars, i, c);
            i += run;
            continue;
        }
        if c == '#' {
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn count_run(chars: &[char], start: usize, mark: char) -> usize {
    let mut n = 0;
    while start + n < chars.len() && chars[start + n] == mark {
        n += 1;
    }
    n
}

fn take_markdown_link(chars: &[char], start: usize) -> Option<(String, usize)> {
    if chars.get(start) != Some(&'[') {
        return None;
    }
    let mut i = start + 1;
    let mut label = String::new();
    while i < chars.len() && chars[i] != ']' {
        label.push(chars[i]);
        i += 1;
    }
    if i >= chars.len() || chars[i] != ']' {
        return None;
    }
    i += 1;
    if chars.get(i) != Some(&'(') {
        return None;
    }
    i += 1;
    while i < chars.len() && chars[i] != ')' {
        i += 1;
    }
    if i >= chars.len() {
        return None;
    }
    Some((label, i + 1))
}

fn collapse_ws(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_space = true;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space && !out.is_empty() {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

/// Pull finished sentences off `buffer`. When `flush` is set, the remainder is
/// a sentence even without terminal punctuation.
pub fn take_sentences(buffer: &mut String, flush: bool) -> Vec<String> {
    let mut sentences = Vec::new();
    while let Some(end) = find_sentence_end(buffer) {
        let rest = buffer.split_off(end);
        let sentence = collapse_ws(buffer);
        *buffer = rest.trim_start().to_string();
        if !sentence.is_empty() {
            sentences.push(sentence);
        }
    }
    if flush {
        let tail = collapse_ws(buffer);
        buffer.clear();
        if !tail.is_empty() {
            sentences.push(tail);
        }
    }
    sentences
}

fn find_sentence_end(text: &str) -> Option<usize> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for (i, &(byte_idx, c)) in chars.iter().enumerate() {
        if c != '.' && c != '!' && c != '?' {
            continue;
        }
        let next = chars.get(i + 1).map(|(_, ch)| *ch);
        if c == '.' && next.is_some_and(|n| n.is_ascii_digit()) {
            continue;
        }
        if c == '.' && is_abbrev_period(text, byte_idx) {
            continue;
        }
        let end = match next {
            None => byte_idx + c.len_utf8(),
            Some(n) if n.is_whitespace() || n == '"' || n == '\u{201d}' => byte_idx + c.len_utf8(),
            Some(_) => continue,
        };
        return Some(end);
    }
    None
}

fn is_abbrev_period(text: &str, period_at: usize) -> bool {
    let prefix = &text[..period_at];
    let token = prefix
        .rsplit(|c: char| !c.is_ascii_alphabetic())
        .next()
        .unwrap_or("");
    ABBREV.contains(&token.to_ascii_lowercase().as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_headings_lists_and_emphasis() {
        let raw = "# Title\n## Sub\n- **bold** item\n1. _italic_ [link](https://example.com)\nSpeak `code` now.";
        let spoken = strip_markdown_for_speech(raw);
        assert!(!spoken.contains('#'), "{spoken}");
        assert!(!spoken.contains("https://"), "{spoken}");
        assert!(!spoken.contains('*'), "{spoken}");
        assert!(!spoken.contains('_'), "{spoken}");
        assert!(spoken.to_ascii_lowercase().contains("title"), "{spoken}");
        assert!(spoken.to_ascii_lowercase().contains("bold"), "{spoken}");
        assert!(spoken.to_ascii_lowercase().contains("item"), "{spoken}");
        assert!(spoken.to_ascii_lowercase().contains("italic"), "{spoken}");
        assert!(spoken.to_ascii_lowercase().contains("link"), "{spoken}");
        assert!(spoken.to_ascii_lowercase().contains("code"), "{spoken}");
    }

    #[test]
    fn first_sentence_is_available_before_flush() {
        let mut buf = String::from("Hello world. More later");
        let ready = take_sentences(&mut buf, false);
        assert_eq!(ready, vec!["Hello world.".to_string()]);
        assert_eq!(buf, "More later");
        let rest = take_sentences(&mut buf, true);
        assert_eq!(rest, vec!["More later".to_string()]);
        assert!(buf.is_empty());
    }

    #[test]
    fn does_not_split_decimals_or_mr() {
        let mut buf = String::from("See Dr. Smith at 3.14 please.");
        let ready = take_sentences(&mut buf, false);
        assert_eq!(ready, vec!["See Dr. Smith at 3.14 please.".to_string()]);
        assert!(buf.is_empty());
    }

    #[test]
    fn splits_questions_and_keeps_plus_lists() {
        let spoken = strip_markdown_for_speech("+ item\n\nWhat now?");
        assert!(spoken.to_ascii_lowercase().contains("item"));
        assert!(spoken.contains('?'));
        let mut buf = String::from("Ready? Go!");
        assert_eq!(
            take_sentences(&mut buf, false),
            vec!["Ready?".to_string(), "Go!".to_string()]
        );
    }

    #[test]
    fn incomplete_markdown_link_is_left_as_text() {
        let spoken = strip_markdown_for_speech("[label](https://x.test and [open");
        assert!(spoken.contains("label") || spoken.contains('['));
    }

    #[test]
    fn think_blocks_and_leftover_tags_never_reach_speech() {
        assert_eq!(
            strip_think_for_speech("<think>secret plan</think> Hello there."),
            "Hello there."
        );
        assert_eq!(speak_text_for_tts("<think>**no**</think> **yes**."), "yes.");
        assert_eq!(strip_think_for_speech("<think>unclosed"), "");
        assert_eq!(strip_think_for_speech("answer</think>"), "answer");
    }

    #[test]
    fn think_filter_holds_tokens_until_the_block_closes() {
        let mut filter = ThinkFilter::default();
        assert_eq!(filter.push("<th", false), "");
        assert_eq!(filter.push("ink>hidden. ", false), "");
        assert_eq!(filter.push("still hidden</th", false), "");
        assert_eq!(filter.push("ink>Spoken now.", true), "Spoken now.");
        assert!(!filter.in_think());
    }

    #[test]
    fn think_filter_holds_a_split_stray_closing_tag() {
        let mut filter = ThinkFilter::default();
        assert_eq!(filter.push("answer</thi", false), "answer");
        assert_eq!(filter.push("nk> done", true), " done");
    }
}
