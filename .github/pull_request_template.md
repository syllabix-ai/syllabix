# Pull request

## What changed

## How verified

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] Per-PR GitHub Checks green

## Harness-quality evidence (only if this PR touches the tool-harness boundary)

If this PR touches any of:

- `crates/syllabix-core/src/executor.rs`
- `crates/syllabix-core/src/openai.rs`
- `crates/syllabix-core/src/types.rs`
- `crates/syllabix-core/src/providers.rs`
- `crates/syllabix-core/tests/harness-quality.rs`

then run the keyed live admission test with your own key (CI never calls the model) and paste the verdict report into the PR body inside an HTML comment starting with `<!-- syllabix-harness-quality -->`, including the five fixture lines (`[disk-space]`, `[repo-search]`, `[known-fetch]`, `[discovery-primary-source]`, `[hostile-prompt]`) and the `summary: valid=X/Y ratio=… escapes=…` line:

```bash
SYLLABIX_LLM_API_KEY=… \
SYLLABIX_HARNESS_BASE_URL=https://api.openai.com/v1 \
SYLLABIX_HARNESS_MODEL=<model-id> \
cargo test -p syllabix-core --test harness-quality -- --ignored --nocapture
```

- [ ] Harness boundary not touched, or report pasted below.
