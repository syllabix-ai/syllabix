#!/usr/bin/env bash
# Install compile dependencies for Syllabix on Debian/Ubuntu.
# Packages match .github/workflows (libasound2-dev, pkg-config, cmake,
# bubblewrap) plus build-essential for hosts without a C++ toolchain.
set -euo pipefail

with_vulkan=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-vulkan) with_vulkan=1 ;;
    -h|--help)
      cat <<'EOF'
usage: scripts/setup-linux.sh [--with-vulkan]

Install apt packages and rustup (if missing) needed to build Syllabix.
  --with-vulkan  also install libvulkan-dev, glslc, spirv-headers
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

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "setup-linux.sh is for Linux only; on macOS use scripts/setup-macos.sh" >&2
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found; this script supports Debian/Ubuntu" >&2
  exit 1
fi

packages=(
  build-essential
  cmake
  pkg-config
  libasound2-dev
  git
  bubblewrap
)
if [[ "${with_vulkan}" -eq 1 ]]; then
  packages+=(libvulkan-dev glslc spirv-headers)
fi

echo "== apt packages =="
sudo apt-get update
sudo apt-get install -y "${packages[@]}"

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

# Entering the repo selects rust-toolchain.toml (1.91.0 + rustfmt/clippy).
root="$(cd "$(dirname "$0")/.." && pwd)"
echo "== toolchain (from ${root}/rust-toolchain.toml) =="
(cd "${root}" && rustup show active-toolchain && rustc --version && cargo --version)
cmake --version | head -n 1
if ! pkg-config --exists alsa; then
  echo "alsa pkg-config check failed" >&2
  exit 1
fi
echo "alsa pkg-config: ok"

echo
echo "Linux compile dependencies ready."
if [[ "${rustup_fresh}" -eq 1 ]]; then
  echo "New rustup install: run  source \"${CARGO_HOME:-$HOME/.cargo}/env\"  (or restart the shell) so cargo is on PATH."
fi
echo "Build: cargo run -p syllabix -- run"
if [[ "${with_vulkan}" -eq 1 ]]; then
  echo "Vulkan opt-in: SYLLABIX_GGML_VULKAN=1 cargo build -p syllabix --release"
fi
