//! Postgres-backed tests for the Riley card tools. They skip (pass with a
//! note) when SYLLABIX_RILEY_DATABASE_URL is unset; CI's coverage job sets it.

use serde_json::{json, Value};
use syllabix_core::types::ToolCall;
use syllabix_riley_eval::db::CardDb;
use syllabix_riley_eval::tools::{dispatch, execute};
use syllabix_riley_eval::DATABASE_URL_ENV;

fn seeded(test: &str) -> Option<CardDb> {
    let Ok(url) = std::env::var(DATABASE_URL_ENV) else {
        eprintln!("skip {test}: {DATABASE_URL_ENV} is not set");
        return None;
    };
    let mut db = CardDb::connect(&url, &format!("riley_t_{test}")).expect("connect");
    db.reset().expect("reset");
    Some(db)
}

fn call(db: &mut CardDb, name: &str, args: Value) -> Result<Value, String> {
    dispatch(db, name, &args)
}

fn card<'a>(snapshot: &'a Value, id: &str) -> &'a Value {
    snapshot["cards"]
        .as_array()
        .unwrap()
        .iter()
        .find(|card| card["id"] == id)
        .unwrap_or_else(|| panic!("card {id}"))
}

fn count(snapshot: &Value, table: &str) -> usize {
    snapshot[table].as_array().unwrap().len()
}

#[test]
fn seed_loads_20_users_35_cards_and_no_replacements() {
    let Some(mut db) = seeded("seed") else { return };
    let snap = db.snapshot().unwrap();
    assert_eq!(count(&snap, "users"), 20);
    assert_eq!(count(&snap, "cards"), 35);
    assert_eq!(count(&snap, "replacements"), 0);
    let statuses = |status: &str| {
        snap["cards"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|card| card["status"] == status)
            .count()
    };
    assert_eq!(
        (
            statuses("active"),
            statuses("frozen"),
            statuses("cancelled")
        ),
        (28, 5, 2)
    );
    db.drop_schema().unwrap();
}

#[test]
fn display_user_info_returns_profile_and_card_ids() {
    let Some(mut db) = seeded("user_info") else {
        return;
    };
    let user = call(
        &mut db,
        "display_user_info",
        json!({"user_id": "u_alice_johnson"}),
    )
    .unwrap();
    assert_eq!(user["name"], "Alice Johnson");
    assert_eq!(user["phone"], "+1-217-555-0101");
    assert_eq!(user["address"], "123 Market St, Springfield, IL");
    assert_eq!(
        user["cards"],
        json!(["c_alice_credit", "c_alice_debit", "c_alice_virtual"])
    );
    let missing = call(&mut db, "display_user_info", json!({"user_id": "u_nobody"})).unwrap();
    assert_eq!(missing, json!({}));
    db.drop_schema().unwrap();
}

#[test]
fn display_card_info_by_last4_includes_a_null_replacement() {
    let Some(mut db) = seeded("by_last4") else {
        return;
    };
    let card = call(
        &mut db,
        "display_card_info_by_last4",
        json!({"last4": "2222"}),
    )
    .unwrap();
    assert_eq!(card["id"], "c_alice_credit");
    assert_eq!(card["status"], "frozen");
    assert_eq!(card["type"], "CREDIT");
    assert_eq!(card["replacement"], Value::Null);
    assert!(card["created_at"]
        .as_str()
        .unwrap()
        .starts_with("2025-01-15T09:00:00"));
    let missing = call(
        &mut db,
        "display_card_info_by_last4",
        json!({"last4": "0000"}),
    )
    .unwrap();
    assert_eq!(missing, json!({}));
    db.drop_schema().unwrap();
}

#[test]
fn change_card_status_freezes_and_unfreezes() {
    let Some(mut db) = seeded("freeze") else {
        return;
    };
    let frozen = call(
        &mut db,
        "change_card_status",
        json!({"card_id": "c_bob_debit", "new_status": "frozen"}),
    )
    .unwrap();
    assert_eq!(frozen["status"], "frozen");
    assert_eq!(frozen["replacement"], Value::Null);
    assert_eq!(
        card(&db.snapshot().unwrap(), "c_bob_debit")["status"],
        "frozen"
    );
    call(
        &mut db,
        "change_card_status",
        json!({"card_id": "c_bob_debit", "new_status": "active"}),
    )
    .unwrap();
    assert_eq!(
        card(&db.snapshot().unwrap(), "c_bob_debit")["status"],
        "active"
    );
    db.drop_schema().unwrap();
}

#[test]
fn change_card_status_never_revives_a_cancelled_card() {
    let Some(mut db) = seeded("cancelled") else {
        return;
    };
    let err = call(
        &mut db,
        "change_card_status",
        json!({"card_id": "c_elaine_debit", "new_status": "active"}),
    )
    .unwrap_err();
    assert_eq!(err, "cannot change status of a cancelled card");
    assert_eq!(
        card(&db.snapshot().unwrap(), "c_elaine_debit")["status"],
        "cancelled"
    );
    let err = call(
        &mut db,
        "change_card_status",
        json!({"card_id": "c_nobody", "new_status": "frozen"}),
    )
    .unwrap_err();
    assert_eq!(err, "card not found");
    db.drop_schema().unwrap();
}

#[test]
fn request_card_replacement_cancels_old_and_issues_successor() {
    let Some(mut db) = seeded("replace") else {
        return;
    };
    let new = call(
        &mut db,
        "request_card_replacement",
        json!({"card_id": "c_bob_travel"}),
    )
    .unwrap();
    let new_last4 = new["last4"].as_str().unwrap().to_string();
    assert!(new["id"].as_str().unwrap().starts_with("c_"));
    assert_eq!(new_last4.len(), 4);
    assert_eq!(new["name"], "Bob Smith - Travel Credit (replacement)");
    assert_eq!(new["user_id"], "u_bob_smith");
    assert_eq!(new["type"], "CREDIT");
    assert_eq!(new["status"], "active");
    let rep = &new["replacement"];
    assert_eq!(rep["card_id"], "c_bob_travel");
    assert_eq!(rep["new_last4"], new_last4.as_str());
    assert_eq!(rep["status"], "requested");
    assert_eq!(rep["reason"], Value::Null);
    assert_eq!(rep["estimated_delivery"], "14 business days");

    let snap = db.snapshot().unwrap();
    assert_eq!(card(&snap, "c_bob_travel")["status"], "cancelled");
    assert_eq!(count(&snap, "cards"), 36);
    assert_eq!(count(&snap, "replacements"), 1);
    let same_last4 = snap["cards"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|card| card["last4"] == new_last4.as_str())
        .count();
    assert_eq!(same_last4, 1, "successor last4 is unused");

    let old = call(
        &mut db,
        "display_card_info_by_last4",
        json!({"last4": "4444"}),
    )
    .unwrap();
    assert_eq!(old["replacement"]["status"], "requested");
    let succ = call(
        &mut db,
        "display_card_info_by_last4",
        json!({"last4": new_last4}),
    )
    .unwrap();
    assert_eq!(succ["replacement"]["card_id"], "c_bob_travel");
    db.drop_schema().unwrap();
}

#[test]
fn request_card_replacement_rejects_cancelled_and_unknown_cards() {
    let Some(mut db) = seeded("replace_err") else {
        return;
    };
    let err = call(
        &mut db,
        "request_card_replacement",
        json!({"card_id": "c_kevin_student"}),
    )
    .unwrap_err();
    assert_eq!(err, "cannot replace an already cancelled card");
    let err = call(
        &mut db,
        "request_card_replacement",
        json!({"card_id": "c_nobody"}),
    )
    .unwrap_err();
    assert_eq!(err, "card not found");
    let snap = db.snapshot().unwrap();
    assert_eq!(count(&snap, "cards"), 35);
    assert_eq!(count(&snap, "replacements"), 0);
    db.drop_schema().unwrap();
}

#[test]
fn update_card_replacement_status_accepts_old_or_new_card() {
    let Some(mut db) = seeded("rep_status") else {
        return;
    };
    let new = call(
        &mut db,
        "request_card_replacement",
        json!({"card_id": "c_dana_credit"}),
    )
    .unwrap();
    let mailed = call(
        &mut db,
        "update_card_replacement_status",
        json!({"card_id": "c_dana_credit", "new_status": "mailed"}),
    )
    .unwrap();
    assert_eq!(mailed["status"], "mailed");
    let delivered = call(
        &mut db,
        "update_card_replacement_status",
        json!({"card_id": new["id"], "new_status": "delivered"}),
    )
    .unwrap();
    assert_eq!(delivered["status"], "delivered");
    assert_eq!(
        db.snapshot().unwrap()["replacements"][0]["status"],
        "delivered"
    );
    let err = call(
        &mut db,
        "update_card_replacement_status",
        json!({"card_id": "c_bob_debit", "new_status": "mailed"}),
    )
    .unwrap_err();
    assert_eq!(err, "no replacement found for this card");
    db.drop_schema().unwrap();
}

#[test]
fn execute_wraps_results_and_errors_as_tool_results() {
    let Some(mut db) = seeded("execute") else {
        return;
    };
    let ok = execute(
        &mut db,
        &ToolCall {
            id: "call-1".into(),
            name: "display_card_info_by_last4".into(),
            arguments: json!({"last4": "1111"}),
        },
    );
    assert!(ok.ok);
    assert_eq!(ok.tool_call_id, "call-1");
    let body: Value = serde_json::from_str(&ok.content).unwrap();
    assert_eq!(body["id"], "c_alice_debit");
    let bad = execute(
        &mut db,
        &ToolCall {
            id: "call-2".into(),
            name: "shell".into(),
            arguments: json!({"command": "ls"}),
        },
    );
    assert!(!bad.ok);
    assert_eq!(bad.content, r#"{"error":"unknown tool: shell"}"#);
    db.drop_schema().unwrap();
}

#[test]
fn reset_restores_the_seed_and_schemas_are_isolated() {
    let Some(mut a) = seeded("iso_a") else { return };
    let mut b = seeded("iso_b").unwrap();
    call(
        &mut a,
        "request_card_replacement",
        json!({"card_id": "c_bob_debit"}),
    )
    .unwrap();
    assert_eq!(count(&a.snapshot().unwrap(), "cards"), 36);
    assert_eq!(count(&b.snapshot().unwrap(), "cards"), 35);
    a.reset().unwrap();
    let snap = a.snapshot().unwrap();
    assert_eq!(count(&snap, "cards"), 35);
    assert_eq!(card(&snap, "c_bob_debit")["status"], "active");
    a.drop_schema().unwrap();
    b.drop_schema().unwrap();
}
