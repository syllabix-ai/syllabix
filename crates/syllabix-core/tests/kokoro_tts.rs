//! Markdown speech cleanup for Kokoro (no model load). Native TTS lives in
//! `native_inference` so `cargo llvm-cov` can skip inference.

use syllabix_core::strip_markdown_for_speech;

#[test]
fn markdown_fixtures_remove_headings_lists_and_emphasis() {
    let spoken = strip_markdown_for_speech(
        "## Hello\n- **world**\n1. _list_\nDo **not** read this [link](https://x.test) aloud.",
    );
    assert!(!spoken.contains('#'), "{spoken}");
    assert!(!spoken.contains('*'), "{spoken}");
    assert!(!spoken.contains('_'), "{spoken}");
    assert!(!spoken.contains("https://"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("hello"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("world"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("list"), "{spoken}");
    assert!(spoken.to_ascii_lowercase().contains("link"), "{spoken}");
}
