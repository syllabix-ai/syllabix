#!/usr/bin/env bash
# Setup smoke test (V0_LAUNCH sequence 27 merge gate): run the
# documented download -> verify -> ./syllabix path without a contributor
# toolchain, then prove the warm-cache offline second run.
#
# Usage:
#   scripts/smoke-setup.sh dist/syllabix-Darwin-arm64
#   SMOKE_RELEASE_URL=https://github.com/syllabix-ai/syllabix/releases/download/v0.1.0 \
#     scripts/smoke-setup.sh syllabix-Darwin-arm64
#
# The spoken 3-minute gate needs a laptop mic/speakers and stays a human
# check (docs/reference-profiles.md). This script proves the machine parts:
# README-exact checksum verify, `--help` and `init` with no rustc/cargo on
# PATH, and an offline second run that must not fetch weights.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
chmod +x scripts/write-sha256sums.sh scripts/smoke-offline-setup.sh

release_url="${SMOKE_RELEASE_URL:-}"
artifact="${1:-}"
if [[ -z "${artifact}" ]]; then
  echo "usage: $0 <path-to-dist-artifact>" >&2
  echo "       SMOKE_RELEASE_URL=<base-url> $0 <asset-name>" >&2
  exit 2
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/syllabix-smoke.XXXXXX")"
cleanup() { rm -rf "${tmp}"; }
trap cleanup EXIT
mkdir -p "${tmp}/home"

fetch() { # fetch <url> <dest>
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$2" "$1"
  else
    echo "need curl or wget to download from a release" >&2
    exit 2
  fi
}

if [[ -n "${release_url}" ]]; then
  base="${artifact}"
  case "${base}" in
    syllabix-Linux-x86_64|syllabix-Darwin-arm64|syllabix-Darwin-x86_64|syllabix-Windows-x86_64.exe) ;;
    *)
      echo "download mode needs a launch asset name, got: ${base}" >&2
      exit 2
      ;;
  esac
  echo "== README download from ${release_url} =="
  fetch "${release_url}/${base}" "${tmp}/${base}"
  fetch "${release_url}/SHA256SUMS" "${tmp}/SHA256SUMS"
else
  if [[ ! -f "${artifact}" ]]; then
    echo "artifact missing: ${artifact}" >&2
    exit 1
  fi
  base="$(basename "${artifact}")"
  case "${base}" in
    syllabix-Linux-x86_64|syllabix-Darwin-arm64|syllabix-Darwin-x86_64|syllabix-Windows-x86_64.exe) ;;
    *)
      os="$(uname -s)"
      arch="$(uname -m)"
      case "${os}/${arch}" in
        Linux/x86_64) base="syllabix-Linux-x86_64" ;;
        Darwin/arm64) base="syllabix-Darwin-arm64" ;;
        Darwin/x86_64) base="syllabix-Darwin-x86_64" ;;
        MINGW*/*|MSYS*/*|CYGWIN*/*) base="syllabix-Windows-x86_64.exe" ;;
        *)
          echo "cannot map ${os}/${arch} to a launch artifact; rename it first" >&2
          exit 2
          ;;
      esac
      echo "staging $(basename "${artifact}") as ${base} (README uname mapping)"
      ;;
  esac
  cp -f "${artifact}" "${tmp}/${base}"
  "${root}/scripts/write-sha256sums.sh" "${tmp}" >/dev/null
fi

echo "== README checksum verify (${base}) =="
(
  cd "${tmp}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --ignore-missing -c SHA256SUMS
  else
    shasum -a 256 -c SHA256SUMS --ignore-missing
  fi
)

chmod +x "${tmp}/${base}"

echo "== clean toolchain --help (no rustc/cargo on PATH) =="
help_out="$(env -i \
  PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
  HOME="${tmp}/home" \
  TERM="${TERM:-xterm}" \
  LC_ALL=C \
  "${tmp}/${base}" --help 2>&1)"
printf '%s\n' "${help_out}"
printf '%s\n' "${help_out}" | grep -q "Local voice agent"

echo "== init needs no cache =="
env -i \
  PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
  HOME="${tmp}/home" \
  TERM="${TERM:-xterm}" \
  LC_ALL=C \
  "${tmp}/${base}" init "${tmp}/home/project" >/dev/null
test -f "${tmp}/home/project/syllabix.yaml"
grep -q "^pipeline:" "${tmp}/home/project/syllabix.yaml"

echo "== offline second run (warm cache) =="
"${root}/scripts/smoke-offline-setup.sh" "${tmp}/${base}"

echo "smoke-setup passed for ${base}"
