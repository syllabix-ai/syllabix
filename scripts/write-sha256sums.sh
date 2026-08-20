#!/usr/bin/env bash
# Write GNU SHA256SUMS for launch artifacts present in a dist directory.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dist_dir="${1:-${DIST_DIR:-${root}/dist}}"
if [[ ! -d "${dist_dir}" ]]; then
  echo "dist directory missing: ${dist_dir}" >&2
  exit 1
fi

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

names=(
  syllabix-Linux-x86_64
  syllabix-Darwin-arm64
  syllabix-Darwin-x86_64
  syllabix-Windows-x86_64.exe
)

out="${dist_dir}/SHA256SUMS"
tmp="$(mktemp "${TMPDIR:-/tmp}/syllabix-sha256sums.XXXXXX")"
cleanup() { rm -f "${tmp}"; }
trap cleanup EXIT

found=0
for name in "${names[@]}"; do
  path="${dist_dir}/${name}"
  if [[ -f "${path}" ]]; then
    echo "$(file_sha256 "${path}")  ${name}" >>"${tmp}"
    found=1
  fi
done

if [[ "${found}" -eq 0 ]]; then
  echo "no launch artifacts in ${dist_dir}" >&2
  exit 1
fi

mv -f "${tmp}" "${out}"
trap - EXIT
echo "wrote ${out}"
cat "${out}"
