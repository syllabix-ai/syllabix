# Install

Users download one executable per OS from GitHub Releases. They never install Python, pip, rustc, CUDA, or Ollama.

## macOS and Linux

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS   # macOS: shasum -a 256 -c SHA256SUMS --ignore-missing
chmod +x syllabix
./syllabix run
```

`uname -s` / `uname -m` map to the artifact names in the table below (`Linux`/`x86_64`, `Darwin`/`arm64`, `Darwin`/`x86_64`).

## Windows

```powershell
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Windows-x86_64.exe -o syllabix.exe
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
certutil -hashfile syllabix.exe SHA256    # compare the hash to the line for syllabix-Windows-x86_64.exe
.\syllabix.exe run
```

## Artifacts

| Target | Artifact |
| --- | --- |
| Linux x64 | `syllabix-Linux-x86_64` |
| macOS Apple Silicon | `syllabix-Darwin-arm64` |
| macOS Intel | `syllabix-Darwin-x86_64` |
| Windows x64 | `syllabix-Windows-x86_64.exe` |

Each GitHub Release also attaches `SHA256SUMS`. Verify before you run.

## First-run model cache

Weights are **not** packed into the executable. The first `run` fetches only the selected stack (defaults: Silero, Whisper `small`, Llama 3.2 1B, Kokoro — about **1.5 GB**) into:

- `$SYLLABIX_CACHE_DIR/models/v1` if `SYLLABIX_CACHE_DIR` is set (that variable is the cache root)
- otherwise `~/.cache/syllabix/models/v1`
- Windows: `%LOCALAPPDATA%\syllabix\cache\models\v1`

Each file is SHA-256 verified. A later `run` with a full cache does not touch the network. `--help` and `init` do not download weights.

Yaml-selected models (larger Whisper, Qwen3.5 GGUFs, Qwen3-TTS) fetch on first use of that id only. Default `run` must not download extra GGUFs.

Cold-start time (binary on disk → first spoken reply on a normal connection) is not yet a published measurement. Treat the first fetch as a bulk download, not a three-minute install.

## macOS Gatekeeper

A curl-downloaded unsigned binary is quarantined. Until Release artifacts are signed and notarized, a clean Mac typically shows “cannot be opened because Apple cannot check it for malicious software.”

Verified escape (pick one):

1. Finder: right-click the binary → **Open** → confirm.
2. Terminal, after checksum verify:

```bash
xattr -d com.apple.quarantine syllabix
./syllabix run
```

If that still fails, the OS refused the file; that is an install bug, not a user error. File it with the macOS version and the exact dialog text.

## Windows SmartScreen

On first open, SmartScreen may warn that the app is unrecognized. Choose **More info** → **Run anyway** after the checksum matches `SHA256SUMS`.

## Microphone permission

macOS: System Settings → Privacy & Security → Microphone. Grant access once; the binary requests it through CoreAudio.

Windows: Settings → Privacy → Microphone.

## After a Release is published

Validate the documented download against that tag (per OS you can touch):

```bash
SMOKE_RELEASE_URL=https://github.com/syllabix-ai/syllabix/releases/download/<tag> \
  scripts/smoke-setup.sh syllabix-Linux-x86_64
```

How to *build* the artifacts lives in [contributing.md](contributing.md).
