//! Markdown-to-speech cleanup and sentence chunking for TTS.

/// Abbreviations that should not split a sentence on the following period.
const ABBREV: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "jr", "sr", "vs", "st", "gen", "col", "sgt", "lt", "gov",
    "inc", "ltd", "eg", "ie", "am", "pm", "us", "usa",
];

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
}
