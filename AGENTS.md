# AGENTS.md

## AI Attribution — MANDATORY

No in-file tags. Git trailers are the audit trail.

### Commits
Keep titles clean: no prefixes in commit subjects or PR titles.

Every commit must be authored by human credentials: Git `Author` name and
email must be the human operator (`user.name` / `user.email`), never an AI
identity, harness, or bot. Do not use `GIT_AUTHOR_*` or `--author` to put
the model or harness in the Author field.

AI is recorded only as a co-author, using the trailers below. For every
commit you create, append:

```
AI-Agent: <harness>
AI-Model: <exact model identifier>
Co-authored-by: {model}-{harness} <{model}-{harness}@syllabix.local>
```

- `<harness>` is one of: `opencode`, `codex`, `cursor`, `agy`.
- `<exact model identifier>` is the full provider/model id
  (e.g. `anthropic/claude-sonnet-4-6`, not just `sonnet`).
- `{model}` is the last `/`-separated segment of `AI-Model`
  (`anthropic/claude-sonnet-4-6` → `claude-sonnet-4-6`), so a co-author
  looks like
  `Co-authored-by: claude-sonnet-4-6-cursor <claude-sonnet-4-6-cursor@syllabix.local>`.
- Each contributing harness appends its own set and never removes others.
- `.local` emails are grep-able in `git log` but map to no GitHub account,
  so no co-author chip — expected. If the harness owns a GitHub bot
  account, use its `noreply` email instead.

### PRs
For repository changes, agents must work in a dedicated Git worktree and raise
a PR for human review. Agents must never merge a PR, even when checks pass or
the PR has been approved. A request to create, fix, verify, or push a PR does
not authorize merging it; only the human operator performs merges.

Follow `.github/pull_request_template.md` exactly: same headings, no added
or removed sections. Check the `cargo fmt` / `cargo clippy` boxes (checked
even when N/A — the template check requires it). Fill `## AI attribution`
with your harness and model. Follow `docs/contributing.md`.
Keep PR titles clean. Squash to one commit per PR (re-squash after fixes).
Merge requires zero unresolved review threads.

### PR comments and reviews
Keep PR titles clean.

If you are writing a review/comment, begin the comment with:

```
[<model> via <harness>]
```

Example: `[claude-sonnet-4-6 via cursor]`. Never post as an unattributed human.

When working on PR comments: reply to every review thread and resolve it
once addressed. No thread left unreplied. Resolving fires no workflow
event, so afterwards edit the PR body to re-run the threads check.
