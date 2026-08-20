#!/usr/bin/env bash
# Prove a dist artifact starts without a contributor toolchain and without
# packed model weights. Hardware is not required.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
bin="${1:-}"
if [[ -z "${bin}" || ! -f "${bin}" ]]; then
  echo "usage: $0 <path-to-dist-syllabix>" >&2
  exit 2
fi

# Packed Whisper small is 487_601_967 bytes; keep the ceiling in sync with
# syllabix_core::MAX_DIST_BINARY_BYTES.
max_bytes=400000000
size="$(wc -c <"${bin}" | tr -d ' ')"
if (( size > max_bytes )); then
  echo "artifact is ${size} bytes; looks like packed weights (max ${max_bytes})" >&2
  exit 1
fi

tmp="$(mktemp -d "${TMPDIR:-/tmp}/syllabix-clean-artifact.XXXXXX")"
cleanup() { rm -rf "${tmp}"; }
trap cleanup EXIT

copied="${tmp}/$(basename "${bin}")"
cp -f "${bin}" "${copied}"
chmod +x "${copied}"

sums_src="$(dirname "${bin}")/SHA256SUMS"
if [[ -f "${sums_src}" ]]; then
  want="$(awk -v n="$(basename "${bin}")" '$2==n {print $1; exit}' "${sums_src}")"
  got="$(
    if command -v sha256sum >/dev/null 2>&1; then
      sha256sum "${copied}" | awk '{print $1}'
    else
      shasum -a 256 "${copied}" | awk '{print $1}'
    fi
  )"
  if [[ -z "${want}" || "${want}" != "${got}" ]]; then
    echo "SHA256SUMS does not match $(basename "${bin}")" >&2
    exit 1
  fi
  echo "SHA256SUMS matches $(basename "${bin}")"
fi

if [[ "$(uname -s)" == "Linux" ]]; then
  "${root}/scripts/check-dynamic-libs.sh" "${copied}"
  "${root}/scripts/check-one-ggml.sh" "${copied}"
fi

os="$(uname -s)"
arch="$(uname -m)"
base="$(basename "${bin}")"
skip_exec=0
case "${base}" in
  syllabix-Linux-x86_64)
    [[ "${os}" == "Linux" && "${arch}" == "x86_64" ]] || skip_exec=1
    ;;
  syllabix-Darwin-arm64)
    [[ "${os}" == "Darwin" && "${arch}" == "arm64" ]] || skip_exec=1
    ;;
  syllabix-Darwin-x86_64)
    [[ "${os}" == "Darwin" && "${arch}" == "x86_64" ]] || skip_exec=1
    ;;
  syllabix-Windows-x86_64.exe)
    [[ "${os}" == MINGW* || "${os}" == MSYS* || "${os}" == CYGWIN* || "${os}" == "Windows_NT" ]] || skip_exec=1
    ;;
  *)
    echo "unknown artifact name ${base}" >&2
    exit 1
    ;;
esac

if [[ "${skip_exec}" -eq 1 ]]; then
  echo "skipping --help exec: ${base} is not native on ${os}/${arch}"
  echo "clean artifact check passed (${size} bytes, no exec)"
  exit 0
fi

# No cargo, rustc, rustup, or repo path on PATH. HOME is empty of models.
# Windows PE needs SYSTEMROOT; Unix needs a minimal PATH.
clean_path="/usr/bin:/bin:/usr/sbin:/sbin"
if [[ -n "${SYSTEMROOT:-}" ]]; then
  clean_path="${SYSTEMROOT}/System32:${SYSTEMROOT}:${clean_path}"
fi

run_help() {
  env -i \
    PATH="${clean_path}" \
    HOME="${tmp}/home" \
    TERM="${TERM:-xterm}" \
    LC_ALL=C \
    SYSTEMROOT="${SYSTEMROOT:-}" \
    WINDIR="${WINDIR:-}" \
    "$@"
}

mkdir -p "${tmp}/home"
echo "== --help without contributor toolchain =="
run_help "${copied}" --help | grep -q "Local voice agent"
run_help "${copied}" --help | grep -q "run"
run_help "${copied}" --help | grep -q "init"
echo "== run --help =="
run_help "${copied}" run --help | grep -q -- "--turn-debug"
run_help "${copied}" run --help | grep -q -- "--barge-in"
echo "== --version =="
run_help "${copied}" --version | grep -q "syllabix"

if command -v docker >/dev/null 2>&1 && [[ "${SYLLABIX_DOCKER_CLEAN:-}" == "1" ]]; then
  echo "== docker ubuntu clean image =="
  docker run --rm \
    -v "${copied}:/syllabix:ro" \
    ubuntu:24.04 \
    bash -lc 'apt-get update -qq && apt-get install -y -qq libasound2 >/dev/null && /syllabix --help' \
    | grep -q "Local voice agent"
fi

echo "clean artifact check passed (${size} bytes)"
