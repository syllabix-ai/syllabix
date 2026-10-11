//! Guard the supported crate-root surface documented in `docs/embed.md`.
//!
//! Freeze-list names must stay crate-root re-exports. Every crate-root
//! `pub use` name must appear on that page. Feature `#[cfg]` on a re-export
//! is honored. Lane 3 loader names need `ggml` (or a coverage build). Lane 2
//! names need `ggml` and `device-audio`, which is what `full` gives you.

use std::collections::BTreeSet;

const LIB_RS: &str = include_str!("../src/lib.rs");
const EMBED_MD: &str = include_str!("../../../docs/embed.md");

/// True when `#[cfg(...)]` on a crate-root `pub use` is on in this test binary.
fn cfg_active_in_this_build(inner: &str) -> bool {
    match inner {
        r#"feature = "ggml""# => cfg!(feature = "ggml"),
        r#"feature = "device-audio""# => cfg!(feature = "device-audio"),
        r#"any(coverage, feature = "ggml")"# => cfg!(coverage) || cfg!(feature = "ggml"),
        r#"all(feature = "ggml", feature = "device-audio")"# => {
            cfg!(feature = "ggml") && cfg!(feature = "device-audio")
        }
        r#"all(target_os = "macos", not(coverage))"# => {
            cfg!(target_os = "macos") && !cfg!(coverage)
        }
        r#"all(target_os = "linux", not(coverage))"# => {
            cfg!(target_os = "linux") && !cfg!(coverage)
        }
        other => panic!("unsupported #[cfg({other})] on a crate-root pub use in lib.rs"),
    }
}

/// Whether a `### Supported types` heading applies to this test binary.
fn freeze_section_active(heading: &str) -> bool {
    if !heading.contains("needs `full`") {
        return true;
    }
    // Lane 3 loader is crate-root with `ggml` (also on in coverage builds).
    if heading.contains("Lane 3") {
        return cfg!(coverage) || cfg!(feature = "ggml");
    }
    // Lane 2 `run_live` needs ggml engines and device audio.
    cfg!(feature = "ggml") && cfg!(feature = "device-audio")
}

fn crate_root_reexports(src: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut buf = String::new();
    let mut in_use = false;
    let mut skip_item = false;
    let mut pending_cfg: Option<String> = None;
    for line in src.lines() {
        let trimmed = line.trim();
        if !in_use {
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("#[cfg(") {
                if let Some(inner) = rest.strip_suffix(")]") {
                    pending_cfg = Some(inner.to_string());
                    continue;
                }
            }
            if let Some(rest) = trimmed.strip_prefix("pub use ") {
                skip_item = pending_cfg
                    .take()
                    .is_some_and(|cfg| !cfg_active_in_this_build(&cfg));
                buf = rest.to_string();
                in_use = true;
            } else {
                pending_cfg = None;
                continue;
            }
        } else {
            buf.push(' ');
            buf.push_str(trimmed);
        }
        if in_use && buf.contains(';') {
            if !skip_item {
                parse_use_item(buf.split(';').next().expect("split"), &mut names);
            }
            buf.clear();
            in_use = false;
            skip_item = false;
        }
    }
    names
}

fn parse_use_item(item: &str, names: &mut BTreeSet<String>) {
    if let Some((_, brace)) = item.split_once('{') {
        let inner = brace.trim().trim_end_matches('}');
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let name = part
                .rsplit_once(" as ")
                .map(|(_, alias)| alias.trim())
                .unwrap_or(part);
            names.insert(name.to_string());
        }
    } else if let Some((_, name)) = item.rsplit_once("::") {
        let name = name
            .rsplit_once(" as ")
            .map(|(_, alias)| alias.trim())
            .unwrap_or(name.trim());
        names.insert(name.to_string());
    }
}

fn backtick_idents(text: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut rest = text;
    while let Some(start) = rest.find('`') {
        rest = &rest[start + 1..];
        let Some(end) = rest.find('`') else {
            break;
        };
        let inner = &rest[..end];
        rest = &rest[end + 1..];
        if is_ident(inner) {
            names.insert(inner.to_string());
        }
    }
    names
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn documented_freeze_names(embed: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_supported = false;
    let mut section_active = false;
    for line in embed.lines() {
        if line.starts_with("### Supported types") {
            in_supported = true;
            section_active = freeze_section_active(line);
            continue;
        }
        if in_supported && line.starts_with("## ") {
            in_supported = false;
            continue;
        }
        if in_supported && section_active && line.starts_with('|') {
            let first_cell = line.trim_start_matches('|').split('|').next().unwrap_or("");
            names.extend(backtick_idents(first_cell));
        }
    }
    names
}

#[test]
fn freeze_list_names_are_crate_root_reexports() {
    let reexports = crate_root_reexports(LIB_RS);
    let missing: Vec<_> = documented_freeze_names(EMBED_MD)
        .into_iter()
        .filter(|name| !reexports.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "docs/embed.md freeze list names missing from this build's crate-root pub use: {missing:?}"
    );
}

#[test]
fn crate_root_reexports_are_on_the_embed_md_freeze_list() {
    let documented = documented_freeze_names(EMBED_MD);
    let extra: Vec<_> = crate_root_reexports(LIB_RS)
        .into_iter()
        .filter(|name| !documented.contains(name))
        .collect();
    assert!(
        extra.is_empty(),
        "crate-root pub use names missing from the docs/embed.md freeze list: {extra:?}"
    );
}

#[test]
fn loop_traits_and_fakes_compile_without_optional_features() {
    use syllabix_core::testing::{CollectingSink, FakeLlm, FakeStt, FakeTts, FakeVad};
    use syllabix_core::{
        run_loop, AudioCapture, AudioSink, Llm, LoopConfig, Session, Stt, Tts, Vad,
    };
    let _ = std::any::type_name::<FakeLlm>();
    let _ = std::any::type_name::<FakeStt>();
    let _ = std::any::type_name::<FakeTts>();
    let _ = std::any::type_name::<FakeVad>();
    let _ = std::any::type_name::<Session>();
    let _ = std::any::type_name::<LoopConfig>();
    let _ = std::any::type_name::<CollectingSink>();
    let _ = run_loop::<
        FakeVad,
        FakeStt,
        FakeLlm,
        FakeTts,
        CollectingSink,
        std::iter::Empty<syllabix_core::AudioFrame>,
    >;
    fn _traits(
        _: &dyn Vad,
        _: &dyn Stt,
        _: &dyn Llm,
        _: &dyn Tts,
        _: &dyn AudioCapture,
        _: &dyn AudioSink,
    ) {
    }
    let _ = _traits;
}

#[cfg(all(feature = "ggml", feature = "device-audio"))]
#[test]
fn full_loader_compiles_with_ggml_and_device_audio() {
    use syllabix_core::{load_real_providers, run_live};
    let _ = run_live;
    let _ = load_real_providers;
}

#[test]
fn lib_rs_states_the_supported_api_rule() {
    assert!(
        LIB_RS.contains("docs/embed.md"),
        "crates/syllabix-core/src/lib.rs must point at docs/embed.md for the supported-API rule"
    );
    assert!(
        LIB_RS.contains("Do not add a new crate-root `pub use`"),
        "crates/syllabix-core/src/lib.rs must state the no-new-pub-use rule"
    );
}
