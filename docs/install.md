# Install

## Download

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

Verify the checksum before running.

## Install from git

The usual install is still the GitHub Release binary above. Lane 1 hosts that spawn `syllabix` do not need to compile anything when they use that artifact.

From a machine that already has the compile toolchain, you can instead install the CLI crate from a git tag:

```bash
cargo install --git https://github.com/syllabix-ai/syllabix.git --tag v0.1.0 syllabix
```

`syllabix` at the end is the CLI package (`crates/syllabix`). Pin `--tag vX.Y.Z` to match the SDK contract. Do not track `main` or a branch. The binary lands on your Cargo bin path (`~/.cargo/bin/syllabix` unless `CARGO_HOME` is set).

This compiles ggml. Same packages as [Compile](#compile); do not repeat them here — embed hosts also list them under [embed — Host compile](embed.md#host-compile). First `run` still downloads models as under [First run](#first-run).

## First run

The first `run` downloads the selected models over HTTPS (defaults: Silero, Whisper `small`, LFM2.5-2.6B, Pocket TTS — about **2.2 GB**; pinned sizes sum to 2,234,270,682 bytes), checks SHA-256, and stores them under the cache root:

| When | Asset directory |
| --- | --- |
| `SYLLABIX_CACHE_DIR` set | `$SYLLABIX_CACHE_DIR/models/v1` |
| else `$XDG_CACHE_HOME` set | `$XDG_CACHE_HOME/syllabix/models/v1` |
| else Windows | `%LOCALAPPDATA%\syllabix\cache\models\v1` |
| else `$HOME` set | `~/.cache/syllabix/models/v1` |
| else | `.syllabix-cache/models/v1` (cwd) |

`SYLLABIX_CACHE_DIR` is the cache root, not the `models/v1` folder. Leave it unset to share the per-user default across the CLI and any host that embeds `syllabix-core`, or set the same root in every process. Hosts: [embed — Cache](embed.md#cache).

Later runs reuse the cache offline. `--help` and `init` do not download weights. Other yaml model ids fetch on first use of that id. Catalogue: [engines.md](engines.md).

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

## Compile

Requires Rust 1.91+ (pinned in `rust-toolchain.toml`), CMake, and a C++ compiler. whisper.cpp and llama.cpp share one `ggml` compiled into the binary (Darwin Metal + Accelerate; Linux/Windows portable CPU by default). Vendored trees ship in the repo — no git submodules. Linux also needs ALSA headers (`libasound2-dev`) and `pkg-config`. Other repos that embed `syllabix-core` (Lane 2 / Lane 3) need this same compile toolchain — [embed — Host compile](embed.md#host-compile).

### Compile dependencies

#### Linux (Debian / Ubuntu)

```bash
./scripts/setup-linux.sh
```

Installs `build-essential`, `cmake`, `pkg-config`, `libasound2-dev`, `git`, and `bubblewrap`, then rustup if missing. CI Linux jobs (including the out-of-tree git-consumer check) run this same script. Optional Vulkan toolchain packages: `./scripts/setup-linux.sh --with-vulkan`, then `SYLLABIX_GGML_VULKAN=1 cargo build -p syllabix --release`. Vulkan details: [vendor/README.md](../vendor/README.md).

A host crate that only depends on `syllabix-core` needs `build-essential`, `cmake`, `pkg-config`, and `libasound2-dev`. `bubblewrap` is for this repo’s Linux sandbox tests, not for compiling the SDK.

On Ubuntu 24.04+, if `bwrap --unshare-net` fails during sandbox tests, see [contributing.md](contributing.md) / CI (`kernel.apparmor_restrict_unprivileged_userns=0`).

#### macOS

```bash
./scripts/setup-macos.sh
```

Ensures Xcode Command Line Tools, installs CMake via Homebrew when needed, and rustup if missing. CI macOS jobs run this same script. Prefer rustup’s `cargo`/`rustc` over Homebrew’s (`"${CARGO_HOME:-$HOME/.cargo}/bin"` first on `PATH`). Metal and Accelerate link automatically.

#### Windows

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (or full VS) with the **Desktop development with C++** workload (MSVC + Windows SDK).
2. Install [CMake](https://cmake.org/download/) and ensure `cmake` is on `PATH`.
3. Install [rustup](https://rustup.rs/) for the `x86_64-pc-windows-msvc` toolchain.
4. Open **x64 Native Tools Command Prompt for VS** (or a Developer PowerShell) so `cl.exe` is visible, then build from that environment.

```powershell
rustc --version
cmake --version
where.exe cl
```

### Build and run

```bash
git clone https://github.com/syllabix-ai/syllabix.git
cd syllabix
# Linux: ./scripts/setup-linux.sh
# macOS: ./scripts/setup-macos.sh
cargo run -p syllabix -- run
```

The first compile builds ggml and the Rust workspace (several minutes cold). The first `run` still downloads models as under [First run](#first-run).

Tests, packaging, and CI: [contributing.md](contributing.md). Embedding `syllabix-core` from another repo: [embed.md](embed.md).
