# Install

## Binary

Download a release for your OS, verify it, and run:

```bash
# macOS / Linux
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS   # macOS: shasum -a 256 -c SHA256SUMS --ignore-missing
chmod +x syllabix
./syllabix run
```

```powershell
# Windows PowerShell
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Windows-x86_64.exe -o syllabix.exe
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
certutil -hashfile syllabix.exe SHA256    # compare the hash to the line for syllabix-Windows-x86_64.exe
.\syllabix.exe run
```

`uname -s` / `uname -m` map to the names below (`Linux`/`x86_64`, `Darwin`/`arm64`, `Darwin`/`x86_64`).

| Target | Artifact |
| --- | --- |
| Linux x64 | `syllabix-Linux-x86_64` |
| macOS Apple Silicon | `syllabix-Darwin-arm64` |
| macOS Intel | `syllabix-Darwin-x86_64` |
| Windows x64 | `syllabix-Windows-x86_64.exe` |

Each release also attaches `SHA256SUMS`. Verify before you run.

## First run

The first `run` downloads the selected models over HTTPS (defaults: Silero, Whisper `small`, Llama 3.2 1B, Kokoro — about **1.5 GB**), checks SHA-256, and stores them in:

- `$SYLLABIX_CACHE_DIR/models/v1` if `SYLLABIX_CACHE_DIR` is set
- otherwise `~/.cache/syllabix/models/v1`
- Windows: `%LOCALAPPDATA%\syllabix\cache\models\v1`

Later runs reuse the cache offline. `--help` and `init` do not download weights. Other yaml model ids fetch on first use of that id. Sources and sizes: [contributing.md](contributing.md#models).

## macOS Gatekeeper

A curl-downloaded unsigned binary is quarantined. Until release artifacts are signed and notarized, a clean Mac typically shows “cannot be opened because Apple cannot check it for malicious software.”

1. Finder: right-click the binary → **Open** → confirm.
2. Or, after checksum verify:

```bash
xattr -d com.apple.quarantine syllabix
./syllabix run
```

## Windows SmartScreen

On first open, SmartScreen may warn that the app is unrecognized. Choose **More info** → **Run anyway** after the checksum matches `SHA256SUMS`.

## Microphone

macOS: System Settings → Privacy & Security → Microphone.

Windows: Settings → Privacy → Microphone.

## Build from source

Requires Rust 1.91+, CMake, and a C++ compiler. Linux also needs ALSA headers (`libasound2-dev`).

```bash
cargo run -p syllabix -- run
```

Tests, packaging, and CI: [contributing.md](contributing.md).
