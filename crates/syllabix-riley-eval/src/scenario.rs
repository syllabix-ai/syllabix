//! Scenario pack for scored Riley calls (#329 phase A3).
//!
//! A scenario is a starting database (the seed plus a `db_overlay`), a
//! caller with a private goal, traits, and facts, and binary checks: the
//! deterministic [`Expectation`]s on the tool trace and final database, and
//! natural-language `assertions` for the judge. A call passes only when
//! every check and every assertion passes. The pack lives in
//! `crates/syllabix-riley-eval/scenarios/` and is compiled in.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::checks::Expectation;

/// Scenario files, in run order.
const PACK: [(&str, &str); 3] = [
    (
        "simple-freeze-lost-card.json",
        include_str!("../scenarios/simple-freeze-lost-card.json"),
    ),
    (
        "complex-stolen-card-replacement.json",
        include_str!("../scenarios/complex-stolen-card-replacement.json"),
    ),
    (
        "adversarial-impostor-cancel.json",
        include_str!("../scenarios/adversarial-impostor-cancel.json"),
    ),
];

/// Difficulty bucket reported in the summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    /// One clear request from a cooperative caller.
    Simple,
    /// Several steps or a changed starting database.
    Complex,
    /// A caller who tries to break policy.
    Adversarial,
}

impl Category {
    /// Name used in JSON output.
    pub fn as_str(self) -> &'static str {
        match self {
            Category::Simple => "simple",
            Category::Complex => "complex",
            Category::Adversarial => "adversarial",
        }
    }
}

/// What the simulated caller knows. The agent never sees this.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Persona {
    /// Private goal for the call.
    pub goal: String,
    /// How the caller behaves.
    pub traits: Vec<String>,
    /// Details the caller can give when asked.
    pub facts: BTreeMap<String, String>,
}

/// One scored call.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    /// Unique id (also the file name without `.json`).
    pub id: String,
    /// Difficulty bucket.
    pub category: Category,
    /// SQL run after the seed, in one transaction.
    pub db_overlay: Vec<String>,
    /// The simulated caller.
    pub caller: Persona,
    /// Caller turns before the caller hangs up.
    pub max_caller_turns: usize,
    /// Deterministic checks on the tool trace and final database.
    pub expect: Vec<Expectation>,
    /// Yes/no statements the judge decides from the call record.
    pub assertions: Vec<String>,
}

impl Scenario {
    /// Parse and check one scenario file.
    pub fn parse(file: &str, text: &str) -> Result<Self, String> {
        let scenario: Scenario =
            serde_json::from_str(text).map_err(|err| format!("{file}: {err}"))?;
        let problem = if format!("{}.json", scenario.id) != file {
            Some("id must match the file name")
        } else if scenario.caller.goal.trim().is_empty() {
            Some("caller.goal is empty")
        } else if scenario.max_caller_turns == 0 {
            Some("max_caller_turns must be at least 1")
        } else if scenario.expect.is_empty() {
            Some("expect is empty")
        } else if scenario.assertions.is_empty()
            || scenario.assertions.iter().any(|a| a.trim().is_empty())
        {
            Some("assertions must be non-empty statements")
        } else {
            None
        };
        match problem {
            Some(problem) => Err(format!("{file}: {problem}")),
            None => Ok(scenario),
        }
    }
}

/// The compiled-in pack, in run order.
pub fn pack() -> Vec<Scenario> {
    PACK.iter()
        .map(|(file, text)| Scenario::parse(file, text).expect("scenario pack is valid"))
        .collect()
}

/// Scenarios whose id is in `ids` (all when `ids` is empty). Unknown ids are
/// an error.
pub fn select(ids: &[String]) -> Result<Vec<Scenario>, String> {
    let all = pack();
    if let Some(unknown) = ids.iter().find(|id| !all.iter().any(|s| &s.id == *id)) {
        let known: Vec<&str> = all.iter().map(|s| s.id.as_str()).collect();
        return Err(format!(
            "unknown scenario {unknown:?}; known: {}",
            known.join(", ")
        ));
    }
    Ok(all
        .into_iter()
        .filter(|s| ids.is_empty() || ids.contains(&s.id))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_has_simple_complex_and_adversarial() {
        let pack = pack();
        let categories: Vec<Category> = pack.iter().map(|s| s.category).collect();
        assert_eq!(
            categories,
            [Category::Simple, Category::Complex, Category::Adversarial]
        );
        assert_eq!(
            categories.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
            ["simple", "complex", "adversarial"]
        );
        assert!(pack
            .iter()
            .any(|s| s.expect.contains(&Expectation::NoMutation)));
        assert!(pack.iter().any(|s| !s.db_overlay.is_empty()));
    }

    #[test]
    fn select_filters_and_rejects_unknown_ids() {
        assert_eq!(select(&[]).unwrap().len(), 3);
        let one = select(&["adversarial-impostor-cancel".into()]).unwrap();
        assert_eq!(one.len(), 1);
        let err = select(&["nope".into()]).unwrap_err();
        assert!(err.contains("unknown scenario \"nope\""), "{err}");
        assert!(err.contains("simple-freeze-lost-card"), "{err}");
    }

    #[test]
    fn parse_rejects_bad_files() {
        let good = PACK[0].1;
        let mut value: serde_json::Value = serde_json::from_str(good).unwrap();
        let file = PACK[0].0;
        let cases = [
            ("id", serde_json::json!("other"), "id must match"),
            ("max_caller_turns", serde_json::json!(0), "at least 1"),
            ("expect", serde_json::json!([]), "expect is empty"),
            (
                "assertions",
                serde_json::json!([" "]),
                "non-empty statements",
            ),
        ];
        for (field, bad, message) in cases {
            let original = value[field].clone();
            value[field] = bad;
            let err = Scenario::parse(file, &value.to_string()).unwrap_err();
            assert!(err.contains(message), "{field}: {err}");
            value[field] = original;
        }
        value["caller"]["goal"] = serde_json::json!("");
        let err = Scenario::parse(file, &value.to_string()).unwrap_err();
        assert!(err.contains("caller.goal is empty"), "{err}");
        let err = Scenario::parse(file, "{\"id\": 1}").unwrap_err();
        assert!(err.starts_with(file), "{err}");
    }
}
