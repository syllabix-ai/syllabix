//! Postgres access for the Riley card tools. Ported from
//! `gradbot/app/db.py` in <https://github.com/veris-ai/riley-agent> (MIT,
//! Copyright (c) 2026 Veris Technologies Inc.). Rows come back as JSON
//! built by Postgres (`row_to_json`), so no date/time crate is needed.

use postgres::{Client, NoTls};
use serde_json::Value;

use crate::INIT_SQL;

/// One connection bound to one Postgres schema. Each schema is an
/// independent copy of the seed, so parallel sessions do not collide.
pub struct CardDb {
    client: Client,
    schema: String,
}

/// Lowercase ASCII identifier, at most 63 bytes, so it can be spliced into
/// `CREATE SCHEMA` without quoting.
pub fn validate_schema(schema: &str) -> Result<(), String> {
    let mut chars = schema.chars();
    let first_ok = matches!(chars.next(), Some('a'..='z' | '_'));
    let rest_ok = chars.all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_'));
    if first_ok && rest_ok && schema.len() <= 63 {
        Ok(())
    } else {
        Err(format!("invalid schema name: {schema:?}"))
    }
}

fn db_err(err: postgres::Error) -> String {
    format!("database error: {err}")
}

fn parse_json(text: String) -> Result<Value, String> {
    serde_json::from_str(&text).map_err(|err| format!("database returned bad JSON: {err}"))
}

impl CardDb {
    /// Connect and select `schema`. Call [`CardDb::reset`] before first use.
    /// Errors never include `url`.
    pub fn connect(url: &str, schema: &str) -> Result<Self, String> {
        validate_schema(schema)?;
        let mut client = Client::connect(url, NoTls).map_err(db_err)?;
        client
            .batch_execute(&format!("SET TIME ZONE 'UTC'; SET search_path TO {schema}"))
            .map_err(db_err)?;
        Ok(Self {
            client,
            schema: schema.to_string(),
        })
    }

    /// Drop the schema and reload [`INIT_SQL`]: 20 users, 35 cards, no replacements.
    pub fn reset(&mut self) -> Result<(), String> {
        let schema = &self.schema;
        self.client
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema}; SET search_path TO {schema}"
            ))
            .map_err(db_err)?;
        self.client.batch_execute(INIT_SQL).map_err(db_err)
    }

    /// Remove the schema (test and scenario cleanup).
    pub fn drop_schema(mut self) -> Result<(), String> {
        let schema = &self.schema;
        self.client
            .batch_execute(&format!("DROP SCHEMA IF EXISTS {schema} CASCADE"))
            .map_err(db_err)
    }

    fn json_opt(
        &mut self,
        sql: &str,
        params: &[&(dyn postgres::types::ToSql + Sync)],
    ) -> Result<Option<Value>, String> {
        match self.client.query_opt(sql, params).map_err(db_err)? {
            Some(row) => parse_json(row.get(0)).map(Some),
            None => Ok(None),
        }
    }

    /// Whole database as `{users, cards, replacements}`, each ordered by id,
    /// for final-state checks.
    pub fn snapshot(&mut self) -> Result<Value, String> {
        let sql = "SELECT json_build_object(\
            'users', COALESCE((SELECT json_agg(u ORDER BY u.id) FROM users u), '[]'::json), \
            'cards', COALESCE((SELECT json_agg(c ORDER BY c.id) FROM cards c), '[]'::json), \
            'replacements', COALESCE((SELECT json_agg(r ORDER BY r.id) FROM replacements r), '[]'::json))::text";
        let row = self.client.query_one(sql, &[]).map_err(db_err)?;
        parse_json(row.get(0))
    }

    /// User row plus sorted `cards` ids, or `None`.
    pub fn user_info(&mut self, user_id: &str) -> Result<Option<Value>, String> {
        self.json_opt(
            "SELECT (to_jsonb(u) || jsonb_build_object('cards', COALESCE(\
                (SELECT jsonb_agg(c.id ORDER BY c.id) FROM cards c WHERE c.user_id = u.id), \
                '[]'::jsonb)))::text FROM users u WHERE u.id = $1",
            &[&user_id],
        )
    }

    fn card_by_id(&mut self, card_id: &str) -> Result<Option<Value>, String> {
        self.json_opt(
            "SELECT row_to_json(c)::text FROM cards c WHERE c.id = $1",
            &[&card_id],
        )
    }

    fn replacement_for_card(&mut self, card_id: &str) -> Result<Option<Value>, String> {
        self.json_opt(
            "SELECT row_to_json(r)::text FROM replacements r WHERE r.card_id = $1 \
             ORDER BY r.created_at DESC, r.id DESC LIMIT 1",
            &[&card_id],
        )
    }

    fn replacement_by_new_last4(&mut self, last4: &str) -> Result<Option<Value>, String> {
        self.json_opt(
            "SELECT row_to_json(r)::text FROM replacements r WHERE r.new_last4 = $1 \
             ORDER BY r.created_at DESC, r.id DESC LIMIT 1",
            &[&last4],
        )
    }

    /// Card by last four digits with its latest `replacement` (by this card's
    /// id, else by `new_last4`), or `None`.
    pub fn card_by_last4(&mut self, last4: &str) -> Result<Option<Value>, String> {
        let Some(mut card) = self.json_opt(
            "SELECT row_to_json(c)::text FROM cards c WHERE c.last4 = $1 ORDER BY c.id LIMIT 1",
            &[&last4],
        )?
        else {
            return Ok(None);
        };
        let card_id = card["id"].as_str().unwrap_or_default().to_string();
        let replacement = match self.replacement_for_card(&card_id)? {
            Some(found) => Some(found),
            None => self.replacement_by_new_last4(last4)?,
        };
        card["replacement"] = replacement.unwrap_or(Value::Null);
        Ok(Some(card))
    }

    /// Set a card's status. A cancelled card can only stay cancelled.
    pub fn change_card_status(&mut self, card_id: &str, new_status: &str) -> Result<Value, String> {
        let current = self
            .card_by_id(card_id)?
            .ok_or_else(|| "card not found".to_string())?;
        if current["status"] == "cancelled" && new_status != "cancelled" {
            return Err("cannot change status of a cancelled card".into());
        }
        let mut card = self
            .json_opt(
                "UPDATE cards SET status = $2, updated_at = now() WHERE id = $1 \
                 RETURNING row_to_json(cards)::text",
                &[&card_id, &new_status],
            )?
            .ok_or_else(|| "card not found".to_string())?;
        card["replacement"] = Value::Null;
        Ok(card)
    }

    /// Cancel `card_id`, create an active successor with an unused last4, and
    /// record a `requested` replacement (14 business days). One transaction.
    pub fn request_card_replacement(&mut self, card_id: &str) -> Result<Value, String> {
        let mut tx = self.client.transaction().map_err(db_err)?;
        let old = tx
            .query_opt(
                "SELECT status FROM cards WHERE id = $1 FOR UPDATE",
                &[&card_id],
            )
            .map_err(db_err)?
            .ok_or_else(|| "card not found".to_string())?;
        let status: String = old.get(0);
        if status == "cancelled" {
            return Err("cannot replace an already cancelled card".into());
        }
        tx.execute(
            "UPDATE cards SET status = 'cancelled', updated_at = now() WHERE id = $1",
            &[&card_id],
        )
        .map_err(db_err)?;
        let row = tx
            .query_one(
                "INSERT INTO cards (id, user_id, name, last4, type, status) \
                 SELECT 'c_' || substr(md5(random()::text || clock_timestamp()::text), 1, 8), \
                        old.user_id, old.name || ' (replacement)', \
                        (SELECT d FROM (SELECT lpad(g::text, 4, '0') AS d FROM generate_series(0, 9999) g) free \
                         WHERE d NOT IN (SELECT last4 FROM cards) ORDER BY random() LIMIT 1), \
                        old.type, 'active' \
                 FROM cards old WHERE old.id = $1 \
                 RETURNING row_to_json(cards)::text",
                &[&card_id],
            )
            .map_err(db_err)?;
        let mut successor = parse_json(row.get(0))?;
        let new_last4 = successor["last4"].as_str().unwrap_or_default().to_string();
        let row = tx
            .query_one(
                "INSERT INTO replacements (id, card_id, new_last4, reason, status, estimated_delivery) \
                 VALUES ('rep_' || substr(md5(random()::text || clock_timestamp()::text), 1, 8), \
                         $1, $2, NULL, 'requested', '14 business days') \
                 RETURNING row_to_json(replacements)::text",
                &[&card_id, &new_last4],
            )
            .map_err(db_err)?;
        successor["replacement"] = parse_json(row.get(0))?;
        tx.commit().map_err(db_err)?;
        Ok(successor)
    }

    /// Advance the latest replacement found by `card_id` (the old card) or by
    /// the successor card's last4.
    pub fn update_card_replacement_status(
        &mut self,
        card_id: &str,
        new_status: &str,
    ) -> Result<Value, String> {
        let mut replacement = self.replacement_for_card(card_id)?;
        if replacement.is_none() {
            if let Some(card) = self.card_by_id(card_id)? {
                let last4 = card["last4"].as_str().unwrap_or_default().to_string();
                replacement = self.replacement_by_new_last4(&last4)?;
            }
        }
        let replacement =
            replacement.ok_or_else(|| "no replacement found for this card".to_string())?;
        let id = replacement["id"].as_str().unwrap_or_default().to_string();
        self.json_opt(
            "UPDATE replacements SET status = $2, updated_at = now() WHERE id = $1 \
             RETURNING row_to_json(replacements)::text",
            &[&id, &new_status],
        )?
        .ok_or_else(|| "no replacement found for this card".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_names_are_plain_lowercase_identifiers() {
        for good in ["riley", "_r", "riley_t_01"] {
            assert!(validate_schema(good).is_ok(), "{good}");
        }
        let long = "a".repeat(64);
        for bad in [
            "",
            "Riley",
            "1riley",
            "riley-x",
            "riley;drop",
            long.as_str(),
        ] {
            assert!(validate_schema(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn connect_rejects_a_bad_schema_before_opening_a_socket() {
        let err = CardDb::connect("postgres://u:hunter2@127.0.0.1:9/x", "Bad")
            .err()
            .expect("bad schema");
        assert!(err.contains("invalid schema name"), "{err}");
        assert!(!err.contains("hunter2"), "{err}");
    }
}
