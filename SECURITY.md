# Security

Syllabix is a local voice agent. The default `run` path keeps microphone audio on the machine after the model cache is filled. An optional online LLM (`pipeline.llm.provider: online`) sends **transcript text only** to the endpoint you set; put the key in `SYLLABIX_LLM_API_KEY`, never in yaml or a `.env` file.

## Report a vulnerability

Do **not** open a public GitHub issue for a security bug, a leaked key, or a sandbox escape.

Use GitHub's private advisory form:

https://github.com/syllabix-ai/syllabix/security/advisories/new

Include the Syllabix version or git SHA, OS, and the shortest steps that reproduce the problem.

## What is in scope

- Unexpected network use on the default local stack after the cache is full
- Secrets appearing in yaml, logs, diagnostics sidecars, or spoken output
- Sandbox / developer-harness escapes (write, network, or secret access beyond the configured ceiling)

## What is not a vulnerability

- First `run` downloading SHA-pinned model weights over HTTPS
- Unsigned macOS / Windows binaries until release signing exists — see [docs/install.md](docs/install.md)
- Model quality, latency, or echo on a given laptop
