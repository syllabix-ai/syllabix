#!/usr/bin/env bash
# Two isolated `dist` builds of the same revision must write matching
# reproducibility sidecars (including artifact SHA-256).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"
target="${1:-}"
host="$(rustc -vV | awk '/^host:/{print $2}')"
target="${target:-${host}}"

a="$(mktemp -d "${TMPDIR:-/tmp}/syllabix-repro-a.XXXXXX")"
b="$(mktemp -d "${TMPDIR:-/tmp}/syllabix-repro-b.XXXXXX")"
cleanup() { rm -rf "${a}" "${b}"; }
trap cleanup EXIT

run_one() {
  local dest="$1"
  mkdir -p "${dest}/dist" "${dest}/target"
  CARGO_TARGET_DIR="${dest}/target" \
    DIST_DIR="${dest}/dist" \
    SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "${root}" log -1 --pretty=%ct)}" \
    "${root}/scripts/package-release.sh" "${target}"
}

echo "== first dist build =="
run_one "${a}"
echo "== second dist build =="
run_one "${b}"

sidecar_a="$(ls "${a}"/dist/*.repro.json)"
sidecar_b="$(ls "${b}"/dist/*.repro.json)"

python3 - "${sidecar_a}" "${sidecar_b}" <<'PY'
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
keys = [
    "git_sha",
    "rustc",
    "cargo",
    "target",
    "profile",
    "artifact",
    "size_bytes",
    "sha256",
    "cargo_lock_sha256",
    "source_date_epoch",
]
mismatch = []
for key in keys:
    if a.get(key) != b.get(key):
        mismatch.append(f"{key}: {a.get(key)!r} vs {b.get(key)!r}")
if mismatch:
    print("reproducibility outputs differ:", file=sys.stderr)
    for line in mismatch:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)
print("two dist builds matched:")
for key in keys:
    print(f"  {key}={a[key]}")
PY

# Leave the second build in repo dist/ so later clean-artifact checks can reuse it.
mkdir -p "${root}/dist"
cp -f "${b}/dist/"* "${root}/dist/"
echo "copied second build to ${root}/dist"
