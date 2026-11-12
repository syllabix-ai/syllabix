//! Blocking chat-completions client for the simulated caller and the judge
//! (#329 phase A3). Uses only `SYLLABIX_LLM_API_KEY`, resolved by the caller
//! of [`ChatClient::new`]. Errors never contain the key, the request, or the
//! response body.

use std::time::Duration;

use serde_json::{json, Value};
use zeroize::Zeroizing;

/// One model behind one OpenAI-compatible endpoint.
pub struct ChatClient {
    endpoint: String,
    model: String,
    api_key: Zeroizing<String>,
    agent: ureq::Agent,
}

impl ChatClient {
    /// `{base_url}/chat/completions` with `model`; each request is limited to
    /// `timeout`.
    pub fn new(base_url: &str, model: &str, api_key: Zeroizing<String>, timeout: Duration) -> Self {
        Self {
            endpoint: syllabix_core::openai::join_endpoint(base_url),
            model: model.to_string(),
            api_key,
            agent: ureq::AgentBuilder::new().timeout(timeout).build(),
        }
    }

    /// Model id sent with every request.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Text of the first choice. `json_object` asks for a JSON object reply.
    pub fn complete(&self, messages: &[Value], json_object: bool) -> Result<String, String> {
        let mut body = json!({ "model": self.model, "messages": messages });
        if json_object {
            body["response_format"] = json!({ "type": "json_object" });
        }
        let response = self
            .agent
            .post(&self.endpoint)
            .set(
                "Authorization",
                &format!("Bearer {}", self.api_key.as_str()),
            )
            .set("Content-Type", "application/json")
            .send_string(&body.to_string());
        let text = match response {
            Ok(response) => response
                .into_string()
                .map_err(|err| format!("could not read the response: {}", err.kind()))?,
            Err(ureq::Error::Status(code, _)) => {
                return Err(format!("chat completions returned HTTP {code}"))
            }
            Err(ureq::Error::Transport(err)) => {
                return Err(format!("chat completions request failed: {}", err.kind()))
            }
        };
        let reply: Value = serde_json::from_str(&text)
            .map_err(|_| "chat completions returned invalid JSON".to_string())?;
        reply["choices"][0]["message"]["content"]
            .as_str()
            .map(str::trim)
            .filter(|content| !content.is_empty())
            .map(str::to_string)
            .ok_or_else(|| "chat completions reply had no text".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// Serve `replies` (status, body) in order on loopback; send back each
    /// request body.
    fn serve(replies: Vec<(u16, &'static str)>) -> (String, mpsc::Receiver<(String, String)>) {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for (status, body) in replies {
                let mut request = server.recv().unwrap();
                let mut sent = String::new();
                request.as_reader().read_to_string(&mut sent).unwrap();
                let auth = request
                    .headers()
                    .iter()
                    .find(|h| h.field.equiv("Authorization"))
                    .map(|h| h.value.to_string())
                    .unwrap_or_default();
                tx.send((sent, auth)).unwrap();
                let response = tiny_http::Response::from_string(body).with_status_code(status);
                request.respond(response).unwrap();
            }
        });
        (format!("http://127.0.0.1:{port}/v1"), rx)
    }

    #[test]
    fn complete_returns_text_and_reports_failures_without_details() {
        let (base, requests) = serve(vec![
            (200, r#"{"choices":[{"message":{"content":"  hello  "}}]}"#),
            (401, r#"{"error":"bad key secret-detail"}"#),
            (200, "not json"),
            (200, r#"{"choices":[{"message":{"content":"   "}}]}"#),
        ]);
        let client = ChatClient::new(
            &base,
            "judge-m",
            Zeroizing::new("chat-key".into()),
            Duration::from_secs(5),
        );
        assert_eq!(client.model(), "judge-m");
        let messages = [json!({"role": "user", "content": "hi"})];
        assert_eq!(client.complete(&messages, true).unwrap(), "hello");
        let (body, auth) = requests.recv().unwrap();
        let body: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(body["model"], "judge-m");
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(auth, "Bearer chat-key");
        assert_eq!(
            client.complete(&messages, false).unwrap_err(),
            "chat completions returned HTTP 401"
        );
        let (body, _) = requests.recv().unwrap();
        assert!(!body.contains("response_format"));
        assert_eq!(
            client.complete(&messages, false).unwrap_err(),
            "chat completions returned invalid JSON"
        );
        assert_eq!(
            client.complete(&messages, false).unwrap_err(),
            "chat completions reply had no text"
        );
    }
}
