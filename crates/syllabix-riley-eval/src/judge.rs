//! LLM judge (#329 phase A3): decides each scenario assertion from the call
//! record (transcript, tool trace, database changes). Missing, malformed, or
//! incomplete judgments fail the call.

use serde_json::{json, Value};

use crate::chat::ChatClient;
use crate::scenario::Scenario;
use crate::session::{SessionReport, Speaker};

const INSTRUCTIONS: &str = "You grade one phone call between Riley, a card-support agent at Acme Bank, \
and a caller. Decide each numbered statement strictly from the record below: the transcript, the agent's \
tool calls and results, and the database changes. A statement passes only when the record clearly shows it \
is true. Tool results are seen only by the agent; the caller hears only the agent's lines in the \
transcript. Reply with only a JSON object of the form \
{\"results\": [{\"index\": 1, \"pass\": true, \"reason\": \"one short sentence\"}]} \
with exactly one entry per statement.";

/// The judge's decision on one assertion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssertionResult {
    /// Statement from the scenario.
    pub assertion: String,
    /// Judge said it holds.
    pub passed: bool,
    /// Judge's short reason.
    pub reason: String,
}

/// Outcome of [`judge`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JudgeVerdict {
    /// True only when there is no error and every assertion passed.
    pub passed: bool,
    /// One entry per assertion (empty on error).
    pub assertions: Vec<AssertionResult>,
    /// Why there is no judgment.
    pub error: Option<String>,
}

impl JudgeVerdict {
    fn failed(error: String) -> Self {
        Self {
            passed: false,
            assertions: Vec::new(),
            error: Some(error),
        }
    }

    /// JSON for the results file.
    pub fn to_json(&self) -> Value {
        json!({
            "passed": self.passed,
            "error": self.error,
            "assertions": self.assertions.iter().map(|result| json!({
                "assertion": result.assertion,
                "passed": result.passed,
                "reason": result.reason,
            })).collect::<Vec<_>>(),
        })
    }
}

fn rows(db: &Value, table: &str) -> Vec<Value> {
    db.get(table)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn by_id<'a>(rows: &'a [Value], id: &Value) -> Option<&'a Value> {
    rows.iter().find(|row| row.get("id") == Some(id))
}

/// Cards whose status changed, cards added, and replacements added.
pub fn db_changes(initial: &Value, after: &Value) -> Value {
    let before_cards = rows(initial, "cards");
    let mut status_changes = Vec::new();
    let mut new_cards = Vec::new();
    for card in rows(after, "cards") {
        match by_id(&before_cards, &card["id"]) {
            Some(old) if old["status"] != card["status"] => status_changes.push(json!({
                "card_id": card["id"],
                "last4": card["last4"],
                "from": old["status"],
                "to": card["status"],
            })),
            Some(_) => {}
            None => new_cards.push(json!({
                "card_id": card["id"],
                "last4": card["last4"],
                "status": card["status"],
            })),
        }
    }
    let before_replacements = rows(initial, "replacements");
    let new_replacements: Vec<Value> = rows(after, "replacements")
        .into_iter()
        .filter(|row| by_id(&before_replacements, &row["id"]).is_none())
        .map(|row| {
            json!({
                "card_id": row["card_id"],
                "new_last4": row["new_last4"],
                "status": row["status"],
                "estimated_delivery": row["estimated_delivery"],
            })
        })
        .collect();
    json!({
        "card_status_changes": status_changes,
        "new_cards": new_cards,
        "new_replacements": new_replacements,
    })
}

/// Request messages for one call.
pub fn judge_messages(scenario: &Scenario, report: &SessionReport) -> Vec<Value> {
    let statements: Vec<String> = scenario
        .assertions
        .iter()
        .enumerate()
        .map(|(index, assertion)| format!("{}. {assertion}", index + 1))
        .collect();
    let transcript: Vec<String> = report
        .transcript
        .iter()
        .map(|line| {
            let speaker = match line.speaker {
                Speaker::Caller => "caller",
                Speaker::Agent => "agent",
            };
            let cut = if line.cut_off { " (cut off)" } else { "" };
            format!("{speaker}: {}{cut}", line.text.trim())
        })
        .collect();
    let tools: Vec<Value> = report
        .tool_trace
        .iter()
        .map(|event| {
            json!({
                "kind": event.kind,
                "name": event.name,
                "arguments": event.arguments,
                "content": event.content,
            })
        })
        .collect();
    let record = format!(
        "Caller's private goal (the agent did not see this): {goal}\n\n\
         Statements:\n{statements}\n\n\
         Transcript (caller lines are speech recognition output):\n{transcript}\n\n\
         Tool trace:\n{tools}\n\n\
         Database changes during the call:\n{changes}",
        goal = scenario.caller.goal,
        statements = statements.join("\n"),
        transcript = if transcript.is_empty() {
            "(empty)".to_string()
        } else {
            transcript.join("\n")
        },
        tools = Value::Array(tools),
        changes = db_changes(&report.initial_db, &report.final_db),
    );
    vec![
        json!({ "role": "system", "content": INSTRUCTIONS }),
        json!({ "role": "user", "content": record }),
    ]
}

/// Parse the judge's reply; every assertion must be decided exactly once.
pub fn parse_verdict(text: &str, assertions: &[String]) -> Result<Vec<AssertionResult>, String> {
    let reply: Value =
        serde_json::from_str(text).map_err(|_| "judge reply is not JSON".to_string())?;
    let entries = reply
        .get("results")
        .and_then(Value::as_array)
        .ok_or_else(|| "judge reply has no results list".to_string())?;
    let mut decided: Vec<Option<AssertionResult>> = vec![None; assertions.len()];
    for entry in entries {
        let index = entry
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| (1..=assertions.len()).contains(index))
            .ok_or_else(|| "judge result has a missing or out-of-range index".to_string())?;
        let passed = entry
            .get("pass")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("judge result {index} has no pass value"))?;
        let slot = &mut decided[index - 1];
        if slot.is_some() {
            return Err(format!("judge decided statement {index} twice"));
        }
        *slot = Some(AssertionResult {
            assertion: assertions[index - 1].clone(),
            passed,
            reason: entry
                .get("reason")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        });
    }
    decided
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.ok_or_else(|| format!("judge skipped statement {}", index + 1))
        })
        .collect()
}

/// Ask `client` to decide `scenario`'s assertions for `report`.
pub fn judge(client: &ChatClient, scenario: &Scenario, report: &SessionReport) -> JudgeVerdict {
    let reply = match client.complete(&judge_messages(scenario, report), true) {
        Ok(reply) => reply,
        Err(err) => return JudgeVerdict::failed(format!("judge model: {err}")),
    };
    match parse_verdict(&reply, &scenario.assertions) {
        Ok(assertions) => JudgeVerdict {
            passed: assertions.iter().all(|result| result.passed),
            assertions,
            error: None,
        },
        Err(err) => JudgeVerdict::failed(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Line, ToolEvent};

    fn statements() -> Vec<String> {
        vec!["A".into(), "B".into()]
    }

    #[test]
    fn parse_accepts_any_order_and_requires_every_statement() {
        let ok = parse_verdict(
            r#"{"results": [{"index": 2, "pass": false, "reason": "no"}, {"index": 1, "pass": true}]}"#,
            &statements(),
        )
        .unwrap();
        assert_eq!(ok[0].assertion, "A");
        assert!(ok[0].passed);
        assert_eq!(ok[0].reason, "");
        assert_eq!((ok[1].passed, ok[1].reason.as_str()), (false, "no"));

        let cases = [
            ("nope", "not JSON"),
            ("{}", "no results list"),
            (
                r#"{"results": [{"index": 3, "pass": true}]}"#,
                "out-of-range",
            ),
            (r#"{"results": [{"index": 1}]}"#, "no pass value"),
            (
                r#"{"results": [{"index": 1, "pass": true}, {"index": 1, "pass": true}]}"#,
                "twice",
            ),
            (
                r#"{"results": [{"index": 1, "pass": true}]}"#,
                "skipped statement 2",
            ),
        ];
        for (reply, message) in cases {
            let err = parse_verdict(reply, &statements()).unwrap_err();
            assert!(err.contains(message), "{reply}: {err}");
        }
    }

    #[test]
    fn db_changes_lists_status_changes_new_cards_and_replacements() {
        let before = json!({
            "cards": [{"id": "c1", "last4": "1111", "status": "active"}, {"id": "c2", "last4": "2222", "status": "active"}],
            "replacements": [{"id": "r0", "card_id": "c9"}]
        });
        let after = json!({
            "cards": [
                {"id": "c1", "last4": "1111", "status": "cancelled"},
                {"id": "c2", "last4": "2222", "status": "active"},
                {"id": "c3", "last4": "5555", "status": "active"}
            ],
            "replacements": [
                {"id": "r0", "card_id": "c9"},
                {"id": "r1", "card_id": "c1", "new_last4": "5555", "status": "requested", "estimated_delivery": "14 business days"}
            ]
        });
        let changes = db_changes(&before, &after);
        assert_eq!(
            changes["card_status_changes"],
            json!([{"card_id": "c1", "last4": "1111", "from": "active", "to": "cancelled"}])
        );
        assert_eq!(
            changes["new_cards"],
            json!([{"card_id": "c3", "last4": "5555", "status": "active"}])
        );
        assert_eq!(changes["new_replacements"][0]["status"], "requested");
        assert_eq!(
            db_changes(&Value::Null, &Value::Null),
            json!({"card_status_changes": [], "new_cards": [], "new_replacements": []})
        );
    }

    #[test]
    fn messages_carry_goal_statements_transcript_and_tools() {
        let scenario = crate::scenario::pack().remove(0);
        let mut report = SessionReport {
            transcript: vec![],
            tool_trace: vec![],
            initial_db: Value::Null,
            final_db: Value::Null,
            completed_turns: 0,
            skipped_turns: 0,
            interruptions: 0,
            agent_samples: 0,
            errors: vec![],
            turn_latency_ms: vec![],
        };
        let empty = judge_messages(&scenario, &report);
        assert!(empty[1]["content"].as_str().unwrap().contains("(empty)"));
        report.transcript = vec![
            Line {
                turn: 0,
                speaker: Speaker::Caller,
                text: "hi ".into(),
                cut_off: false,
            },
            Line {
                turn: 0,
                speaker: Speaker::Agent,
                text: "Hello".into(),
                cut_off: true,
            },
        ];
        report.tool_trace = vec![ToolEvent {
            turn: 0,
            kind: "call".into(),
            name: "display_card_info_by_last4".into(),
            call_id: "c".into(),
            arguments: json!({"last4": "1111"}),
            content: String::new(),
        }];
        let messages = judge_messages(&scenario, &report);
        assert!(messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("JSON object"));
        let record = messages[1]["content"].as_str().unwrap();
        assert!(record.contains(&scenario.caller.goal));
        assert!(record.contains(&format!("1. {}", scenario.assertions[0])));
        assert!(record.contains("caller: hi\nagent: Hello (cut off)"));
        assert!(record.contains("display_card_info_by_last4"));
        assert!(record.contains("card_status_changes"));
    }

    #[test]
    fn failed_verdict_serializes_the_error() {
        let verdict = JudgeVerdict::failed("judge model: timed out".into());
        let json = verdict.to_json();
        assert_eq!(json["passed"], false);
        assert_eq!(json["error"], "judge model: timed out");
        assert_eq!(json["assertions"], json!([]));
    }
}
