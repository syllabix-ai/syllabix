//! Convert HuggingFace `tokenizer.json` into Moonshine's BinTokenizer format.
//!
//! Matches `moonshine-src/scripts/convert_tokenizer.py` so the C API can load
//! official HF streaming assets that ship `tokenizer.json` instead of
//! `tokenizer.bin`.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

fn provider(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: "moonshine",
        message: message.into(),
    }
}

fn write_token(out: &mut Vec<u8>, token: &[u8]) {
    let length = token.len();
    if length == 0 {
        out.push(0);
    } else if length < 128 {
        out.push(length as u8);
        out.extend_from_slice(token);
    } else {
        let first = (length % 128) as u8 + 128;
        let second = (length / 128) as u8;
        out.push(first);
        out.push(second);
        out.extend_from_slice(token);
    }
}

/// Convert a HuggingFace tokenizer.json document into BinTokenizer bytes.
pub fn convert_huggingface_json(text: &str) -> Result<Vec<u8>> {
    let data: Value =
        serde_json::from_str(text).map_err(|err| provider(format!("bad tokenizer.json: {err}")))?;
    let vocab = data
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| provider("tokenizer.json missing model.vocab"))?;

    let mut max_id = 0usize;
    for id in vocab.values() {
        let id = id
            .as_u64()
            .ok_or_else(|| provider("tokenizer vocab id is not an integer"))?
            as usize;
        max_id = max_id.max(id);
    }
    if let Some(added) = data.get("added_tokens").and_then(Value::as_array) {
        for entry in added {
            if let Some(id) = entry.get("id").and_then(Value::as_u64) {
                max_id = max_id.max(id as usize);
            }
        }
    }

    let mut tokens = vec![Vec::<u8>::new(); max_id + 1];
    for (token, id) in vocab {
        let id = id.as_u64().unwrap() as usize;
        tokens[id] = token.as_bytes().to_vec();
    }
    if let Some(added) = data.get("added_tokens").and_then(Value::as_array) {
        for entry in added {
            let Some(id) = entry.get("id").and_then(Value::as_u64) else {
                continue;
            };
            let content = entry
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default();
            tokens[id as usize] = content.as_bytes().to_vec();
        }
    }

    let mut out = Vec::new();
    for token in &tokens {
        write_token(&mut out, token);
    }
    Ok(out)
}

/// Read `tokenizer.json` and write `tokenizer.bin` beside it (or to `dest`).
pub fn write_bin_beside_json(json_path: &Path, dest: &Path) -> Result<()> {
    let text = fs::read_to_string(json_path).map_err(|err| {
        provider(format!(
            "could not read tokenizer.json {}: {err}",
            json_path.display()
        ))
    })?;
    let bytes = convert_huggingface_json(&text)?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            provider(format!(
                "could not create tokenizer.bin directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    fs::write(dest, bytes).map_err(|err| {
        provider(format!(
            "could not write tokenizer.bin {}: {err}",
            dest.display()
        ))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_minimal_vocab() {
        let json = r#"{
            "model": {"type":"BPE","vocab":{"<unk>":0,"<s>":1,"</s>":2,"▁hi":3}},
            "added_tokens":[{"id":0,"content":"<unk>"},{"id":1,"content":"<s>"},{"id":2,"content":"</s>"}]
        }"#;
        let bytes = convert_huggingface_json(json).expect("convert");
        // 4 tokens: lengths 5,3,4,4 plus payloads
        assert!(bytes.len() > 4);
        assert_eq!(bytes[0], 5); // <unk>
    }
}
