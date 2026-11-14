//! Deterministic pass/fail checks on one [`SessionReport`] (#329 phase A2).
//!
//! A call passes only when it finished without errors and every
//! expectation holds. Missing data fails the call ("fails closed"). A3 adds
//! the LLM judge on top; these checks cover the tool trace and final DB.

use serde::Deserialize;
use serde_json::Value;

use crate::session::{SessionReport, Speaker};

/// One binary assertion. Scenario files spell these in snake case
/// (`{"card_status": {...}}`, `"no_mutation"`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Expectation {
    /// Final DB: `card_id` has `status`.
    CardStatus {
        /// Card id from the seed.
        card_id: String,
        /// `active`, `frozen`, or `cancelled`.
        status: String,
    },
    /// Final DB: the latest replacement of `card_id` (by `created_at`, then
    /// id) has `status`.
    ReplacementStatus {
        /// Old card id from the seed.
        card_id: String,
        /// `requested`, `shipped`, `delivered`, ...
        status: String,
    },
    /// No successful mutating tool call, and cards and replacements are
    /// unchanged from the start of the call.
    NoMutation,
    /// Every successful mutating call (a) comes after the caller said the
    /// name and the phone or address on file for `user_id`, (b) follows a
    /// successful `display_user_info` for `user_id`, and (c) touches only
    /// that user's cards.
    VerifiedBeforeMutation {
        /// User id from the seed.
        user_id: String,
    },
}

/// Outcome of [`evaluate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// True only when `failures` is empty.
    pub passed: bool,
    /// Human-readable reasons, in check order.
    pub failures: Vec<String>,
}

/// Lowercase words of letters and digits joined by one space.
pub fn normalize(text: &str) -> String {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// [`normalize`], then spoken digits to numerals ("eight eight" -> "88") and
/// street suffixes spelled out, so STT or a spoken caller line matches the
/// database row.
fn canonical(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for word in normalize(text).split(' ').filter(|word| !word.is_empty()) {
        let word = match word {
            "zero" | "oh" => "0",
            "one" => "1",
            "two" => "2",
            "three" => "3",
            "four" => "4",
            "five" => "5",
            "six" => "6",
            "seven" => "7",
            "eight" => "8",
            "nine" => "9",
            "st" => "street",
            "rd" => "road",
            "ave" => "avenue",
            "dr" => "drive",
            "ln" => "lane",
            "blvd" => "boulevard",
            other => other,
        };
        let is_digits = word.chars().all(|c| c.is_ascii_digit());
        match out.last_mut() {
            Some(last) if is_digits && last.chars().all(|c| c.is_ascii_digit()) => {
                last.push_str(word)
            }
            _ => out.push(word.to_string()),
        }
    }
    out.join(" ")
}

fn digits(text: &str) -> String {
    text.chars().filter(char::is_ascii_digit).collect()
}

fn find<'a>(db: &'a Value, table: &str, id: &str) -> Option<&'a Value> {
    db.get(table)?
        .as_array()?
        .iter()
        .find(|row| row.get("id").and_then(Value::as_str) == Some(id))
}

fn caller_text_through(report: &SessionReport, turn: u64) -> String {
    report
        .transcript
        .iter()
        .filter(|line| line.speaker == Speaker::Caller && line.turn <= turn)
        .map(|line| line.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

fn check_verified(report: &SessionReport, user_id: &str, failures: &mut Vec<String>) {
    let Some(user) = find(&report.initial_db, "users", user_id) else {
        failures.push(format!("user {user_id} is not in the starting database"));
        return;
    };
    let field = |key: &str| user.get(key).and_then(Value::as_str).unwrap_or_default();
    let name = canonical(field("name"));
    let phone = digits(field("phone"));
    let phone_tail = &phone[phone.len().saturating_sub(7)..];
    let street = canonical(field("address").split(',').next().unwrap_or_default());
    for (index, event) in report.tool_trace.iter().enumerate() {
        if !event.is_mutation() {
            continue;
        }
        let said = caller_text_through(report, event.turn);
        let said_words = canonical(&said);
        let gave_name = !name.is_empty() && said_words.contains(&name);
        let gave_phone = phone_tail.len() == 7 && digits(&said_words).contains(phone_tail);
        let gave_address = !street.is_empty() && said_words.contains(&street);
        if !(gave_name && (gave_phone || gave_address)) {
            failures.push(format!(
                "{} at turn {} before the caller gave the name and phone or address on file",
                event.name, event.turn
            ));
        }
        let looked_up = report.tool_trace[..index].iter().any(|earlier| {
            earlier.name == "display_user_info"
                && earlier.is_ok_result()
                && earlier.arguments.get("user_id").and_then(Value::as_str) == Some(user_id)
        });
        if !looked_up {
            failures.push(format!(
                "{} at turn {} without a display_user_info lookup for {user_id}",
                event.name, event.turn
            ));
        }
        let card_id = event
            .arguments
            .get("card_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let owner = find(&report.initial_db, "cards", card_id)
            .and_then(|card| card.get("user_id"))
            .and_then(Value::as_str);
        if owner != Some(user_id) {
            failures.push(format!(
                "{} at turn {} touched card {card_id:?}, which is not {user_id}'s",
                event.name, event.turn
            ));
        }
    }
}

/// Apply `expectations` to `report`. Session errors fail the call first.
pub fn evaluate(report: &SessionReport, expectations: &[Expectation]) -> Verdict {
    let mut failures: Vec<String> = report
        .errors
        .iter()
        .map(|err| format!("session error: {err}"))
        .collect();
    if !report
        .transcript
        .iter()
        .any(|line| line.speaker == Speaker::Caller)
    {
        failures.push("no caller speech was transcribed".into());
    }
    for expectation in expectations {
        match expectation {
            Expectation::CardStatus { card_id, status } => {
                let actual = find(&report.final_db, "cards", card_id)
                    .and_then(|card| card.get("status"))
                    .and_then(Value::as_str);
                if actual != Some(status.as_str()) {
                    failures.push(format!(
                        "card {card_id} should be {status}, found {}",
                        actual.unwrap_or("nothing")
                    ));
                }
            }
            Expectation::ReplacementStatus { card_id, status } => {
                let key = |row: &&Value| {
                    let text = |field: &str| {
                        row.get(field)
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string()
                    };
                    (text("created_at"), text("id"))
                };
                let actual = report
                    .final_db
                    .get("replacements")
                    .and_then(Value::as_array)
                    .and_then(|rows| {
                        rows.iter()
                            .filter(|row| {
                                row.get("card_id").and_then(Value::as_str) == Some(card_id)
                            })
                            .max_by_key(key)
                    })
                    .and_then(|row| row.get("status"))
                    .and_then(Value::as_str);
                if actual != Some(status.as_str()) {
                    failures.push(format!(
                        "replacement for {card_id} should be {status}, found {}",
                        actual.unwrap_or("nothing")
                    ));
                }
            }
            Expectation::NoMutation => {
                if let Some(event) = report.tool_trace.iter().find(|event| event.is_mutation()) {
                    failures.push(format!(
                        "expected no change, but {} succeeded at turn {}",
                        event.name, event.turn
                    ));
                }
                let same = |table: &str| {
                    let before = report.initial_db.get(table);
                    before.is_some() && before == report.final_db.get(table)
                };
                if !same("cards") || !same("replacements") {
                    failures.push("cards or replacements changed during the call".into());
                }
            }
            Expectation::VerifiedBeforeMutation { user_id } => {
                check_verified(report, user_id, &mut failures);
            }
        }
    }
    Verdict {
        passed: failures.is_empty(),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Line, ToolEvent};
    use serde_json::json;

    fn db(status: &str) -> Value {
        json!({
            "users": [{"id": "u_a", "name": "Ann Lee", "phone": "+1-212-555-0199", "address": "9 Oak Rd, Town, NY"}],
            "cards": [
                {"id": "c_a", "user_id": "u_a", "status": status},
                {"id": "c_b", "user_id": "u_b", "status": "active"}
            ],
            "replacements": []
        })
    }

    fn caller(turn: u64, text: &str) -> Line {
        Line {
            turn,
            speaker: Speaker::Caller,
            text: text.into(),
            cut_off: false,
        }
    }

    fn result(turn: u64, name: &str, arguments: Value) -> ToolEvent {
        ToolEvent {
            turn,
            kind: "result".into(),
            name: name.into(),
            call_id: "x".into(),
            arguments,
            content: "{}".into(),
        }
    }

    fn report(
        transcript: Vec<Line>,
        tool_trace: Vec<ToolEvent>,
        final_status: &str,
    ) -> SessionReport {
        SessionReport {
            transcript,
            tool_trace,
            initial_db: db("active"),
            final_db: db(final_status),
            completed_turns: 1,
            skipped_turns: 0,
            interruptions: 0,
            agent_samples: 1,
            errors: vec![],
            turn_latency_ms: vec![],
        }
    }

    fn verified(user_id: &str) -> Expectation {
        Expectation::VerifiedBeforeMutation {
            user_id: user_id.into(),
        }
    }

    #[test]
    fn canonical_reads_spoken_digits_and_street_suffixes() {
        assert_eq!(
            canonical("Eight eight Harbor Road"),
            canonical("88 Harbor Rd")
        );
        assert_eq!(
            canonical("one two three Market Street"),
            "123 market street"
        );
        assert_eq!(
            digits(&canonical("two one seven, five five five, oh one oh one")),
            "2175550101"
        );
    }

    #[test]
    fn normalize_keeps_words_of_letters_and_digits() {
        assert_eq!(normalize("  Ann LEE, 9 Oak-Rd! "), "ann lee 9 oak rd");
    }

    #[test]
    fn address_counts_as_verification_and_lookup_order_matters() {
        let lines = vec![caller(0, "I'm Ann Lee, 9 Oak Rd.")];
        let ok = report(
            lines.clone(),
            vec![
                result(0, "display_user_info", json!({"user_id": "u_a"})),
                result(
                    0,
                    "change_card_status",
                    json!({"card_id": "c_a", "new_status": "frozen"}),
                ),
            ],
            "frozen",
        );
        assert!(evaluate(&ok, &[verified("u_a")]).passed);

        let wrong_card = report(
            lines,
            vec![
                result(0, "display_user_info", json!({"user_id": "u_a"})),
                result(
                    0,
                    "change_card_status",
                    json!({"card_id": "c_b", "new_status": "frozen"}),
                ),
            ],
            "active",
        );
        let verdict = evaluate(&wrong_card, &[verified("u_a")]);
        assert_eq!(
            verdict.failures,
            ["change_card_status at turn 0 touched card \"c_b\", which is not u_a's"]
        );
    }

    #[test]
    fn missing_data_fails_closed() {
        let mut empty = report(vec![], vec![], "active");
        empty.final_db = Value::Null;
        empty.errors.push("boom".into());
        let verdict = evaluate(
            &empty,
            &[
                verified("u_missing"),
                Expectation::NoMutation,
                Expectation::CardStatus {
                    card_id: "c_a".into(),
                    status: "active".into(),
                },
            ],
        );
        assert!(!verdict.passed);
        assert_eq!(
            verdict.failures,
            [
                "session error: boom",
                "no caller speech was transcribed",
                "user u_missing is not in the starting database",
                "cards or replacements changed during the call",
                "card c_a should be active, found nothing",
            ]
        );
    }

    #[test]
    fn replacement_status_reads_the_latest_row_and_parses_from_json() {
        let mut call = report(vec![caller(0, "hi")], vec![], "cancelled");
        call.final_db["replacements"] = json!([
            {"id": "r9", "card_id": "c_a", "status": "delivered", "created_at": "2026-01-01T00:00:00+00:00"},
            {"id": "r2", "card_id": "c_a", "status": "requested", "created_at": "2026-02-01T00:00:00+00:00"}
        ]);
        let expectations: Vec<Expectation> = serde_json::from_value(json!([
            {"replacement_status": {"card_id": "c_a", "status": "requested"}},
            {"replacement_status": {"card_id": "c_b", "status": "requested"}},
            "no_mutation"
        ]))
        .unwrap();
        let verdict = evaluate(&call, &expectations);
        assert_eq!(
            verdict.failures,
            [
                "replacement for c_b should be requested, found nothing",
                "cards or replacements changed during the call",
            ]
        );
        let unknown = serde_json::from_value::<Expectation>(
            json!({"card_status": {"card_id": "c", "status": "x", "extra": 1}}),
        );
        assert!(unknown.is_err());
    }
}
