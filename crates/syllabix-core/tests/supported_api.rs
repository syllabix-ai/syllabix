//! Guard the supported crate-root surface documented in `docs/embed.md`.
//!
//! Freeze-list names must stay crate-root re-exports. A strict check that
//! every crate-root `pub use` is named on that page comes back in Phase 2,
//! after fakes, `*_ASSET` constants, and the sandbox/G2P helpers leave the
//! crate root.

use std::collections::BTreeSet;

const LIB_RS: &str = include_str!("../src/lib.rs");
const EMBED_MD: &str = include_str!("../../../docs/embed.md");

fn crate_root_reexports(src: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut buf = String::new();
    let mut in_use = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if !in_use {
            if trimmed.starts_with("//") {
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("pub use ") {
                buf = rest.to_string();
                in_use = true;
            } else {
                continue;
            }
        } else {
            buf.push(' ');
            buf.push_str(trimmed);
        }
        if in_use && buf.contains(';') {
            parse_use_item(buf.split(';').next().expect("split"), &mut names);
            buf.clear();
            in_use = false;
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
        names.insert(name.trim().to_string());
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

fn freeze_list_names(embed: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_supported = false;
    for line in embed.lines() {
        if line.starts_with("### Supported types") {
            in_supported = true;
            continue;
        }
        if in_supported && line.starts_with("## ") {
            in_supported = false;
            continue;
        }
        if in_supported && line.starts_with('|') {
            let first_cell = line.trim_start_matches('|').split('|').next().unwrap_or("");
            names.extend(backtick_idents(first_cell));
        }
    }
    names
}

#[test]
fn freeze_list_names_are_crate_root_reexports() {
    let reexports = crate_root_reexports(LIB_RS);
    let missing: Vec<_> = freeze_list_names(EMBED_MD)
        .into_iter()
        .filter(|name| !reexports.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "docs/embed.md freeze list names missing from crate-root pub use: {missing:?}"
    );
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
