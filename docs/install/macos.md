# Install on macOS

Artifacts: `syllabix-Darwin-arm64` (Apple Silicon) or `syllabix-Darwin-x86_64` (Intel).

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing
chmod +x syllabix
./syllabix run
```

`uname -s` / `uname -m` must be `Darwin` / `arm64` or `Darwin` / `x86_64`.

## Gatekeeper

A curl-downloaded unsigned binary is quarantined. Until release artifacts are signed and notarized, a clean Mac typically shows “cannot be opened because Apple cannot check it for malicious software.”

1. Finder: right-click the binary → **Open** → confirm.
2. Or, after checksum verify:

```bash
xattr -d com.apple.quarantine syllabix
./syllabix run
```

## Microphone

System Settings → Privacy & Security → Microphone. Grant access when `run` prompts.

## Models

First `run` fetches the default stack (~2.2 GB) into `~/.cache/syllabix/models/v1` (or `$SYLLABIX_CACHE_DIR/models/v1`). Apple Silicon uses Metal for Whisper and the local LLM. Intel uses CPU.

Cache, Windows/Linux, and build-from-source: [../install.md](../install.md).
