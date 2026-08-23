#!/usr/bin/env bash
# Linux stand-in for .github/workflows/ci.yml (GitHub Actions is unavailable).
# Order matches DEVELOPMENT.md: cheap/fail-fast first; native inference last.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

echo "== rustfmt =="
cargo fmt --all -- --check

echo "== clippy =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== release build =="
cargo build --workspace --release

echo "== dynamic libs =="
chmod +x scripts/check-dynamic-libs.sh
scripts/check-dynamic-libs.sh target/release/syllabix

echo "== one ggml =="
chmod +x scripts/check-one-ggml.sh
scripts/check-one-ggml.sh target/release/syllabix

if command -v cargo-llvm-cov >/dev/null 2>&1 || cargo llvm-cov --version >/dev/null 2>&1; then
  echo "== coverage (no whisper.cpp / llama.cpp / Kokoro weights) =="
  cargo llvm-cov --workspace --fail-under-lines 85 --cobertura --output-path coverage.xml
else
  echo "cargo-llvm-cov not installed; skip coverage (CI used cargo-llvm-cov@0.6.21)"
fi

echo "== test =="
# Launch-stack native inference (whisper small / llama 1B / kokoro / six-turn) once.
# Optional yaml models are exclusive: SYLLABIX_NATIVE_MODELS=<ids> (not this script).
cargo test --workspace

echo "== dist package =="
chmod +x scripts/package-release.sh scripts/check-clean-artifact.sh
scripts/package-release.sh
sidecar="$(ls "${root}/dist"/*.repro.json)"
artifact="${sidecar%.repro.json}"
echo "== clean artifact =="
scripts/check-clean-artifact.sh "${artifact}"

echo "== smoke-offline-setup =="
chmod +x scripts/smoke-offline-setup.sh scripts/write-sha256sums.sh
scripts/smoke-offline-setup.sh "${artifact}"

echo "== smoke-setup (sequence 27) =="
chmod +x scripts/smoke-setup.sh
scripts/smoke-setup.sh "${artifact}"

echo "linux CI stand-in passed"
