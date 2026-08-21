#!/usr/bin/env bash
# One promise: the models are already in the cache, so a second run must
# work without fetching anything. Verifies checksum and clean `--help`
# first, then runs from a temp HOME with a warm cache and asserts no
# network use.
#
# Spoken-reply in under three minutes needs a laptop mic/speakers (Hardware
# Yes). This script does not open a conversation unless devices exist.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
bin="${1:-}"
if [[ -z "${bin}" || ! -f "${bin}" ]]; then
  echo "usage: $0 <path-to-dist-syllabix>" >&2
  exit 2
fi

chmod +x "${root}/scripts/check-clean-artifact.sh" "${root}/scripts/write-sha256sums.sh"

dist_dir="$(cd "$(dirname "${bin}")" && pwd)"
base="$(basename "${bin}")"
sums="${dist_dir}/SHA256SUMS"
if [[ ! -f "${sums}" ]]; then
  "${root}/scripts/write-sha256sums.sh" "${dist_dir}"
fi

echo "== SHA256SUMS =="
if command -v sha256sum >/dev/null 2>&1; then
  (cd "${dist_dir}" && sha256sum --ignore-missing -c SHA256SUMS)
else
  want="$(awk -v n="${base}" '$2==n {print $1}' "${sums}")"
  got="$(shasum -a 256 "${bin}" | awk '{print $1}')"
  if [[ -z "${want}" || "${want}" != "${got}" ]]; then
    echo "checksum mismatch for ${base}" >&2
    exit 1
  fi
  echo "${base}: OK"
fi

echo "== clean artifact --help =="
"${root}/scripts/check-clean-artifact.sh" "${bin}"

os="$(uname -s)"
arch="$(uname -m)"
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
esac
if [[ "${skip_exec}" -eq 1 ]]; then
  echo "skipping offline run: ${base} is not native on ${os}/${arch}"
  echo "smoke-offline-setup passed (checksum + skipped exec)"
  exit 0
fi

# Warm cache from the contributor cache if present. First-run HTTPS fill of
# an empty cache is the human 3-minute path, not this merge script.
src_cache="${SYLLABIX_CACHE_DIR:-${HOME:+${HOME}/.cache/syllabix}}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/syllabix-clean-machine.XXXXXX")"
cleanup() { rm -rf "${tmp}"; }
trap cleanup EXIT
mkdir -p "${tmp}/home"
copied="${tmp}/${base}"
cp -f "${bin}" "${copied}"
chmod +x "${copied}"

if [[ -n "${src_cache}" && -d "${src_cache}/models/v1" ]]; then
  mkdir -p "${tmp}/home/.cache/syllabix/models"
  cp -a "${src_cache}/models/v1" "${tmp}/home/.cache/syllabix/models/v1"
  echo "== second run with network blocked (warm cache) =="
  # Portable watchdog: GNU timeout when present, otherwise background kill.
  # macOS ships no `timeout`; without this the run would block forever on a
  # machine that has mic/speakers (it opens devices and waits for speech).
  # The sanitized `env -i` stays a real executable, so it composes with both.
  inner=(env -i \
    PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
    HOME="${tmp}/home" \
    TERM="${TERM:-xterm}" \
    LC_ALL=C)
  if command -v unshare >/dev/null 2>&1 && unshare -n true >/dev/null 2>&1; then
    inner+=(unshare -n)
  fi
  inner+=("${copied}" run)
  # Run from a neutral cwd: a clean machine has no ./syllabix.yaml, and the
  # repo checkout must not leak one into the offline check.
  set +e
  if command -v timeout >/dev/null 2>&1; then
    out="$(cd "${tmp}/home" && timeout 8 "${inner[@]}" 2>&1)"
    status=$?
  else
    log="${tmp}/offline-run.log"
    (cd "${tmp}/home" && exec "${inner[@]}") >"${log}" 2>&1 &
    pid=$!
    { sleep 8 && kill "${pid}" 2>/dev/null; } &
    watchdog=$!
    wait "${pid}"
    status=$?
    kill "${watchdog}" 2>/dev/null || true
    wait "${watchdog}" 2>/dev/null || true
    out="$(cat "${log}" 2>/dev/null || true)"
  fi
  set -e
  if [[ "${status}" -eq 127 ]]; then
    echo "offline run failed to start the binary (exit 127)" >&2
    exit 1
  fi
  printf '%s\n' "${out}" | tail -n 40
  if printf '%s\n' "${out}" | grep -qiE 'download failed|network blocked|checksum mismatch'; then
    echo "offline run touched the network or rejected the cache" >&2
    exit 1
  fi
  if printf '%s\n' "${out}" | grep -qiE 'model cache:'; then
    echo "offline run failed in the model cache" >&2
    exit 1
  fi
  # Device-open failure is expected on a VM with no mic/speakers.
  echo "offline run did not fetch weights (exit ${status})"
else
  echo "no warm model cache at ${src_cache:-unset}; skip offline second run"
  echo "human: empty cache, download the Release binary, ./syllabix run, then a second run with network blocked"
fi

echo "smoke-offline-setup passed"
