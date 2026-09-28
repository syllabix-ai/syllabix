# Issues

## Bugs and features

Open a public issue: https://github.com/syllabix-ai/syllabix/issues/new/choose

Pick the matching template:

- Audio / conversation — mic, speakers, echo, barge-in, STT, or spoken reply.
- Install / first open — binary will not verify, open, or fetch models.
- Config / yaml — `syllabix.yaml` keys, models, online LLM, diagnostics.

Check [troubleshooting](troubleshooting.md) and [FAQ](faq.md) first — mic, echo, keys, and first-run fetch are covered there.

Include in every report:

- Syllabix version or git SHA, and OS (macOS Apple Silicon / Intel, Linux x64, Windows x64).
- Artifact or how you installed (Release file name, or `cargo run`).
- Exact command and full error text.
- `syllabix.yaml` when used. Never paste API keys — `SYLLABIX_LLM_API_KEY` is environment-only.
- For audio reports: mic and speaker names from the startup line, barge-in on/off, and diagnostics (`turn.json` plus `utterance.wav` / `clean.wav` / `tts.wav` when not private) — see [diagnostics](diagnostics.md).

## Security

Syllabix is a local voice agent. The default `run` path keeps microphone audio on the machine after the model cache is filled. An optional online LLM (`pipeline.llm.provider: online`) sends **transcript text only** to the endpoint you set; put the key in `SYLLABIX_LLM_API_KEY`, never in yaml or a `.env` file.

Do **not** open a public GitHub issue for a security bug, a leaked key, or a sandbox escape.

Use GitHub's private advisory form:

https://github.com/syllabix-ai/syllabix/security/advisories/new

Include the Syllabix version or git SHA, OS, and the shortest steps that reproduce the problem.

### What is in scope

- Unexpected network use on the default local stack after the cache is full
- Secrets appearing in yaml, logs, diagnostics sidecars, or spoken output
- Sandbox / developer-harness escapes (write, network, or secret access beyond the configured ceiling)

### What is not a vulnerability

- First `run` downloading SHA-pinned model weights over HTTPS
- Unsigned macOS / Windows binaries until release signing exists — see [install](install.md)
- Model quality, latency, or echo on a given laptop
