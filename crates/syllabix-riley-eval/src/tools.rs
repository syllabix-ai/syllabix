//! The five Riley card tools: schemas, argument checks, and dispatch.
//! Names, descriptions, and parameter schemas are copied from
//! `gradbot/app/tools.py` in <https://github.com/veris-ai/riley-agent> (MIT,
//! Copyright (c) 2026 Veris Technologies Inc.).

use serde_json::{json, Value};
use syllabix_core::types::{ToolCall, ToolResult};

use crate::db::CardDb;

/// Tool names in upstream order.
pub const TOOL_NAMES: [&str; 5] = [
    "display_user_info",
    "display_card_info_by_last4",
    "change_card_status",
    "request_card_replacement",
    "update_card_replacement_status",
];

const CARD_STATUSES: [&str; 3] = ["active", "frozen", "cancelled"];
const REPLACEMENT_STATUSES: [&str; 3] = ["requested", "mailed", "delivered"];

/// `[{name, description, parameters}]`, verbatim from upstream `_SCHEMAS`.
pub fn tool_schemas() -> Value {
    json!([
        {
            "name": "display_user_info",
            "description": "Retrieve user account information including name, email, phone, address, and list of card IDs. Returns {} if not found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user's unique identifier (e.g. u_alice_johnson)."
                    }
                },
                "required": ["user_id"],
                "additionalProperties": false
            }
        },
        {
            "name": "display_card_info_by_last4",
            "description": "Find a card by the last 4 digits of the card number and return its details. Returns {} if not found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last4": {
                        "type": "string",
                        "description": "The last 4 digits of the card number (e.g. '1234')."
                    }
                },
                "required": ["last4"],
                "additionalProperties": false
            }
        },
        {
            "name": "change_card_status",
            "description": "Update a card's status. A cancelled card cannot be changed to any other status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "string",
                        "description": "The card's unique identifier (e.g. c_alice_debit)."
                    },
                    "new_status": {
                        "type": "string",
                        "enum": ["active", "frozen", "cancelled"],
                        "description": "The new card status."
                    }
                },
                "required": ["card_id", "new_status"],
                "additionalProperties": false
            }
        },
        {
            "name": "request_card_replacement",
            "description": "Cancel the given card and issue a replacement. Returns the new card. Cannot replace an already cancelled card.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "string",
                        "description": "The card's unique identifier to replace."
                    }
                },
                "required": ["card_id"],
                "additionalProperties": false
            }
        },
        {
            "name": "update_card_replacement_status",
            "description": "Update a replacement's delivery status (requested/mailed/delivered).",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_id": {
                        "type": "string",
                        "description": "The card's unique identifier."
                    },
                    "new_status": {
                        "type": "string",
                        "enum": ["requested", "mailed", "delivered"],
                        "description": "The new replacement status."
                    }
                },
                "required": ["card_id", "new_status"],
                "additionalProperties": false
            }
        }
    ])
}

/// OpenAI chat-completions `tools` array (same shape as
/// `syllabix_core::llm::local_tool_definitions_json_for`).
pub fn openai_tool_definitions() -> Value {
    let tools = tool_schemas()
        .as_array()
        .expect("schemas array")
        .iter()
        .map(|schema| json!({ "type": "function", "function": schema }))
        .collect();
    Value::Array(tools)
}

/// MCP `tools/list` entries (`inputSchema` instead of `parameters`).
pub fn mcp_tool_list() -> Value {
    let tools = tool_schemas()
        .as_array()
        .expect("schemas array")
        .iter()
        .map(|schema| {
            json!({
                "name": schema["name"],
                "description": schema["description"],
                "inputSchema": schema["parameters"],
            })
        })
        .collect();
    Value::Array(tools)
}

/// One validated tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardOp {
    UserInfo { user_id: String },
    CardByLast4 { last4: String },
    ChangeStatus { card_id: String, new_status: String },
    Replace { card_id: String },
    UpdateReplacement { card_id: String, new_status: String },
}

fn required_str(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("missing string argument: {key}"))
}

fn required_enum(args: &Value, key: &str, allowed: &[&str]) -> Result<String, String> {
    let value = required_str(args, key)?;
    if allowed.contains(&value.as_str()) {
        Ok(value)
    } else {
        Err(format!("{key} must be one of: {}", allowed.join(", ")))
    }
}

/// Check the name and arguments without touching the database.
pub fn parse_call(name: &str, args: &Value) -> Result<CardOp, String> {
    if !TOOL_NAMES.contains(&name) {
        return Err(format!("unknown tool: {name}"));
    }
    if !args.is_object() {
        return Err("tool arguments must be a JSON object".into());
    }
    Ok(match name {
        "display_user_info" => CardOp::UserInfo {
            user_id: required_str(args, "user_id")?,
        },
        "display_card_info_by_last4" => CardOp::CardByLast4 {
            last4: required_str(args, "last4")?,
        },
        "change_card_status" => CardOp::ChangeStatus {
            card_id: required_str(args, "card_id")?,
            new_status: required_enum(args, "new_status", &CARD_STATUSES)?,
        },
        "request_card_replacement" => CardOp::Replace {
            card_id: required_str(args, "card_id")?,
        },
        _ => CardOp::UpdateReplacement {
            card_id: required_str(args, "card_id")?,
            new_status: required_enum(args, "new_status", &REPLACEMENT_STATUSES)?,
        },
    })
}

/// Run one tool. Lookups that find nothing return `{}` like upstream.
pub fn dispatch(db: &mut CardDb, name: &str, args: &Value) -> Result<Value, String> {
    match parse_call(name, args)? {
        CardOp::UserInfo { user_id } => Ok(db.user_info(&user_id)?.unwrap_or_else(|| json!({}))),
        CardOp::CardByLast4 { last4 } => Ok(db.card_by_last4(&last4)?.unwrap_or_else(|| json!({}))),
        CardOp::ChangeStatus {
            card_id,
            new_status,
        } => db.change_card_status(&card_id, &new_status),
        CardOp::Replace { card_id } => db.request_card_replacement(&card_id),
        CardOp::UpdateReplacement {
            card_id,
            new_status,
        } => db.update_card_replacement_status(&card_id, &new_status),
    }
}

/// [`dispatch`] wrapped as a harness [`ToolResult`]. Errors become
/// `{"error": "..."}` with `ok: false`, as upstream surfaces them.
pub fn execute(db: &mut CardDb, call: &ToolCall) -> ToolResult {
    match dispatch(db, &call.name, &call.arguments) {
        Ok(value) => ToolResult {
            tool_call_id: call.id.clone(),
            content: value.to_string(),
            ok: true,
        },
        Err(message) => ToolResult {
            tool_call_id: call.id.clone(),
            content: json!({ "error": message }).to_string(),
            ok: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemas_list_the_five_upstream_tools_in_order() {
        let names: Vec<String> = tool_schemas()
            .as_array()
            .unwrap()
            .iter()
            .map(|schema| schema["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(names, TOOL_NAMES);
        for schema in tool_schemas().as_array().unwrap() {
            assert_eq!(schema["parameters"]["additionalProperties"], false);
        }
    }

    #[test]
    fn openai_definitions_wrap_each_schema_as_a_function() {
        let tools = openai_tool_definitions();
        let tools = tools.as_array().unwrap();
        assert_eq!(tools.len(), 5);
        assert!(tools.iter().all(|tool| tool["type"] == "function"));
        assert_eq!(
            tools[2]["function"]["parameters"]["properties"]["new_status"]["enum"],
            json!(["active", "frozen", "cancelled"])
        );
        assert_eq!(
            tools[4]["function"]["parameters"]["properties"]["new_status"]["enum"],
            json!(["requested", "mailed", "delivered"])
        );
    }

    #[test]
    fn mcp_list_uses_input_schema() {
        let list = mcp_tool_list();
        let first = &list.as_array().unwrap()[0];
        assert_eq!(first["name"], "display_user_info");
        assert_eq!(first["inputSchema"]["required"], json!(["user_id"]));
        assert!(first.get("parameters").is_none());
    }

    #[test]
    fn parse_call_accepts_every_tool() {
        let cases = [
            ("display_user_info", json!({"user_id": "u_bob_smith"})),
            ("display_card_info_by_last4", json!({"last4": "3333"})),
            (
                "change_card_status",
                json!({"card_id": "c_bob_debit", "new_status": "frozen"}),
            ),
            (
                "request_card_replacement",
                json!({"card_id": "c_bob_debit"}),
            ),
            (
                "update_card_replacement_status",
                json!({"card_id": "c_bob_debit", "new_status": "mailed"}),
            ),
        ];
        for (name, args) in cases {
            assert!(parse_call(name, &args).is_ok(), "{name}");
        }
    }

    #[test]
    fn parse_call_rejects_bad_names_and_arguments() {
        let err = |name: &str, args: Value| parse_call(name, &args).unwrap_err();
        assert_eq!(err("shell", json!({})), "unknown tool: shell");
        assert_eq!(
            err("display_user_info", json!("u_bob_smith")),
            "tool arguments must be a JSON object"
        );
        assert_eq!(
            err("display_user_info", json!({})),
            "missing string argument: user_id"
        );
        assert_eq!(
            err("display_card_info_by_last4", json!({"last4": 3333})),
            "missing string argument: last4"
        );
        assert_eq!(
            err(
                "change_card_status",
                json!({"card_id": "c_bob_debit", "new_status": "deleted"})
            ),
            "new_status must be one of: active, frozen, cancelled"
        );
        assert_eq!(
            err(
                "update_card_replacement_status",
                json!({"card_id": "c_bob_debit", "new_status": "lost"})
            ),
            "new_status must be one of: requested, mailed, delivered"
        );
    }
}
