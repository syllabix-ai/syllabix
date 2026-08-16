#!/usr/bin/env bash
# Linux stand-in for .github/workflows/ci.yml (GitHub Actions is unavailable).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

echo "== rustfmt =="
cargo fmt --all -- --check

echo "== clippy =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== test =="
cargo test --workspace

echo "== release build =="
cargo build --workspace --release

echo "== dynamic libs =="
chmod +x scripts/check-dynamic-libs.sh
scripts/check-dynamic-libs.sh target/release/syllabix

echo "== one ggml =="
chmod +x scripts/check-one-ggml.sh
scripts/check-one-ggml.sh target/release/syllabix

if command -v cargo-llvm-cov >/dev/null 2>&1 || cargo llvm-cov --version >/dev/null 2>&1; then
  echo "== coverage =="
  cargo llvm-cov --workspace --fail-under-lines 85 --cobertura --output-path coverage.xml
else
  echo "cargo-llvm-cov not installed; skip coverage (CI used cargo-llvm-cov@0.6.21)"
fi

echo "linux CI stand-in passed"
