//! Whisper language codes for `pipeline.stt.language`.
//!
//! The table mirrors the vendored whisper.cpp language list (id order) so
//! config validation stays field-level and never needs native weights.

/// Pseudo-code: detect the language from the utterance audio.
pub const LANGUAGE_AUTO: &str = "auto";

/// `(code, English name)` pairs whisper.cpp supports, in whisper.cpp id order.
pub const WHISPER_LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("es", "Spanish"),
    ("ru", "Russian"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("ja", "Japanese"),
    ("pt", "Portuguese"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("ca", "Catalan"),
    ("nl", "Dutch"),
    ("ar", "Arabic"),
    ("sv", "Swedish"),
    ("it", "Italian"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("fi", "Finnish"),
    ("vi", "Vietnamese"),
    ("he", "Hebrew"),
    ("uk", "Ukrainian"),
    ("el", "Greek"),
    ("ms", "Malay"),
    ("cs", "Czech"),
    ("ro", "Romanian"),
    ("da", "Danish"),
    ("hu", "Hungarian"),
    ("ta", "Tamil"),
    ("no", "Norwegian"),
    ("th", "Thai"),
    ("ur", "Urdu"),
    ("hr", "Croatian"),
    ("bg", "Bulgarian"),
    ("lt", "Lithuanian"),
    ("la", "Latin"),
    ("mi", "Maori"),
    ("ml", "Malayalam"),
    ("cy", "Welsh"),
    ("sk", "Slovak"),
    ("te", "Telugu"),
    ("fa", "Persian"),
    ("lv", "Latvian"),
    ("bn", "Bengali"),
    ("sr", "Serbian"),
    ("az", "Azerbaijani"),
    ("sl", "Slovenian"),
    ("kn", "Kannada"),
    ("et", "Estonian"),
    ("mk", "Macedonian"),
    ("br", "Breton"),
    ("eu", "Basque"),
    ("is", "Icelandic"),
    ("hy", "Armenian"),
    ("ne", "Nepali"),
    ("mn", "Mongolian"),
    ("bs", "Bosnian"),
    ("kk", "Kazakh"),
    ("sq", "Albanian"),
    ("sw", "Swahili"),
    ("gl", "Galician"),
    ("mr", "Marathi"),
    ("pa", "Punjabi"),
    ("si", "Sinhala"),
    ("km", "Khmer"),
    ("sn", "Shona"),
    ("yo", "Yoruba"),
    ("so", "Somali"),
    ("af", "Afrikaans"),
    ("oc", "Occitan"),
    ("ka", "Georgian"),
    ("be", "Belarusian"),
    ("tg", "Tajik"),
    ("sd", "Sindhi"),
    ("gu", "Gujarati"),
    ("am", "Amharic"),
    ("yi", "Yiddish"),
    ("lo", "Lao"),
    ("uz", "Uzbek"),
    ("fo", "Faroese"),
    ("ht", "Haitian Creole"),
    ("ps", "Pashto"),
    ("tk", "Turkmen"),
    ("nn", "Nynorsk"),
    ("mt", "Maltese"),
    ("sa", "Sanskrit"),
    ("lb", "Luxembourgish"),
    ("my", "Myanmar"),
    ("bo", "Tibetan"),
    ("tl", "Tagalog"),
    ("mg", "Malagasy"),
    ("as", "Assamese"),
    ("tt", "Tatar"),
    ("haw", "Hawaiian"),
    ("ln", "Lingala"),
    ("ha", "Hausa"),
    ("ba", "Bashkir"),
    ("jw", "Javanese"),
    ("su", "Sundanese"),
    ("yue", "Cantonese"),
];

/// True when yaml may select `code` (a table code or [`LANGUAGE_AUTO`]).
pub fn is_supported(code: &str) -> bool {
    code == LANGUAGE_AUTO || language_name(code).is_some()
}

/// English display name for a language code (`French`). `None` when unknown.
pub fn language_name(code: &str) -> Option<&'static str> {
    WHISPER_LANGUAGES
        .iter()
        .find(|(id, _)| *id == code)
        .map(|(_, name)| *name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn table_matches_whisper_id_order_and_uniqueness() {
        // whisper.cpp assigns sequential ids; en is 0 and yue is 99.
        assert_eq!(WHISPER_LANGUAGES.len(), 100);
        let mut seen = HashSet::new();
        for (code, name) in WHISPER_LANGUAGES {
            assert!(!code.is_empty() && code.len() <= 4, "{code}");
            assert!(
                code.chars().all(|c| c.is_ascii_lowercase()),
                "codes are lowercase ISO ids: {code}"
            );
            assert!(!name.is_empty());
            assert!(seen.insert(*code), "duplicate code {code}");
        }
        assert_eq!(WHISPER_LANGUAGES[0], ("en", "English"));
        assert_eq!(language_name("yue"), Some("Cantonese"));
    }

    #[test]
    fn lookups_and_auto_are_supported() {
        assert!(is_supported(LANGUAGE_AUTO));
        assert!(is_supported("en"));
        assert!(is_supported("fr"));
        assert_eq!(language_name("auto"), None, "auto is not a real language");
        assert!(!is_supported("klingon"));
        assert!(!is_supported("xx"));
        assert!(!is_supported("EN"), "codes are lowercase");
        assert_eq!(language_name("fr"), Some("French"));
        assert_eq!(language_name("de"), Some("German"));
        assert_eq!(language_name("nope"), None);
    }
}
