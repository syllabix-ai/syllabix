#!/usr/bin/env bash
# Install compile dependencies for Syllabix on macOS (CMake + rustup).
# Apple Clang comes from Xcode Command Line Tools; Metal/Accelerate link
# automatically. Matches .github/workflows macOS steps (brew install cmake).
set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
usage: scripts/setup-macos.sh

Install Xcode CLT (if missing), CMake via Homebrew, and rustup (if missing).
EOF
      exit 0
      ;;
    *)
      echo "unknown option: $1 (try --help)" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "setup-macos.sh is for macOS only; on Linux use scripts/setup-linux.sh" >&2
  exit 1
fi

echo "== Xcode Command Line Tools =="
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Installing Xcode Command Line Tools (GUI prompt)…"
  xcode-select --install
  echo "Finish the installer, then re-run scripts/setup-macos.sh" >&2
  exit 1
fi
echo "CLT: $(xcode-select -p)"

echo "== CMake =="
if command -v cmake >/dev/null 2>&1; then
  cmake --version | head -n 1
else
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found; install from https://brew.sh then re-run" >&2
    exit 1
  fi
  brew install cmake
  cmake --version | head -n 1
fi

rustup_fresh=0
if ! command -v rustup >/dev/null 2>&1; then
  echo "== rustup =="
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "${CARGO_HOME:-$HOME/.cargo}/env"
  rustup_fresh=1
else
  echo "== rustup already installed =="
fi

# Prefer rustup over Homebrew rustc/cargo when both are on PATH.
cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
export PATH="${cargo_bin}:${PATH}"

root="$(cd "$(dirname "$0")/.." && pwd)"
echo "== toolchain (from ${root}/rust-toolchain.toml) =="
(cd "${root}" && rustup show active-toolchain && rustc --version && cargo --version)

echo
echo "macOS compile dependencies ready."
if [[ "${rustup_fresh}" -eq 1 ]]; then
  echo "New rustup install: run  source \"${CARGO_HOME:-$HOME/.cargo}/env\"  (or restart the shell) so cargo is on PATH."
fi
echo "Ensure ${cargo_bin} is first on PATH if Homebrew also ships rustc."
echo "Build: cargo run -p syllabix -- run"
