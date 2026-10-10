#!/usr/bin/env bash
# Out-of-tree cargo check of the Lane 2 / Lane 3 embed examples.
#
# Other repos pin syllabix-core with a git tag (see docs/embed.md). This
# script builds a dummy crate that is not a workspace member and depends on
# this repo the same way. On v* tags it uses git + tag. On PRs and local
# runs a new tag does not exist, so it substitutes git + rev (CI) or a path
# override (local default).
#
# Usage:
#   ./scripts/ci-git-consumer.sh
#   SYLLABIX_GIT_TAG=v0.1.0 ./scripts/ci-git-consumer.sh
#   SYLLABIX_GIT_URL=https://github.com/syllabix-ai/syllabix.git \
#     SYLLABIX_GIT_REV=<sha> ./scripts/ci-git-consumer.sh
#   ./scripts/ci-git-consumer.sh --prepare   # write crate only (for rust-cache)
#   ./scripts/ci-git-consumer.sh --check     # cargo check an already-written crate
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
work="${root}/target/git-consumer"
embed_examples="${root}/crates/syllabix-core/examples"
mode="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare) mode="prepare" ;;
    --check) mode="check" ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $1 (try --help)" >&2
      exit 2
      ;;
  esac
  shift
done

toml_escape() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  printf '%s' "$s"
}

prepare() {
  local embed custom dep
  embed="${embed_examples}/embed-loop.rs"
  custom="${embed_examples}/custom-llm.rs"
  if [[ ! -f "${embed}" || ! -f "${custom}" ]]; then
    echo "missing embed examples under ${embed_examples}" >&2
    exit 1
  fi

  if [[ -n "${SYLLABIX_GIT_TAG:-}" ]]; then
    local url="${SYLLABIX_GIT_URL:-https://github.com/syllabix-ai/syllabix.git}"
    dep="{ git = \"$(toml_escape "${url}")\", tag = \"$(toml_escape "${SYLLABIX_GIT_TAG}")\" }"
  elif [[ -n "${SYLLABIX_GIT_REV:-}" ]]; then
    local url="${SYLLABIX_GIT_URL:-https://github.com/syllabix-ai/syllabix.git}"
    dep="{ git = \"$(toml_escape "${url}")\", rev = \"$(toml_escape "${SYLLABIX_GIT_REV}")\" }"
  else
    # Local default: path override. Unpushed commits cannot be fetched as git+rev.
    dep="{ path = \"$(toml_escape "${root}/crates/syllabix-core")\" }"
  fi

  rm -rf "${work}"
  mkdir -p "${work}/src/bin"
  cp "${embed}" "${work}/src/bin/embed-loop.rs"
  cp "${custom}" "${work}/src/bin/custom-llm.rs"
  cp "${root}/rust-toolchain.toml" "${work}/rust-toolchain.toml"

  cat > "${work}/Cargo.toml" <<EOF
# Empty [workspace] keeps this crate out of the Syllabix workspace when the
# dummy lives under target/ (Cargo still walks parent directories).
[workspace]

[package]
name = "syllabix-git-consumer"
version = "0.0.0"
edition = "2021"
publish = false
description = "CI stand-in for an out-of-tree host of syllabix-core. Not a workspace member."

[dependencies]
syllabix-core = ${dep}
EOF

  echo "git-consumer crate at ${work}"
  echo "syllabix-core = ${dep}"
}

check() {
  if [[ ! -f "${work}/Cargo.toml" ]]; then
    echo "missing ${work}/Cargo.toml; run $0 --prepare first" >&2
    exit 1
  fi
  (
    cd "${work}"
    cargo check --bins
  )
}

case "${mode}" in
  prepare) prepare ;;
  check) check ;;
  all)
    prepare
    check
    ;;
esac
