#!/usr/bin/env bash
# Attach dist/ binaries to a GitHub Release when Actions cannot run.
# Usage: scripts/publish-release.sh v0.1.0 [dist_dir]
#
# Requires gh with permission to create releases. This script does not
# build the Darwin/Windows matrices; copy those files into dist/ first.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"

tag="${1:-}"
dist_dir="${2:-${root}/dist}"
if [[ -z "${tag}" || ! "${tag}" =~ ^v[0-9] ]]; then
  echo "usage: $0 v0.1.0 [dist_dir]" >&2
  exit 2
fi
if [[ ! -d "${dist_dir}" ]]; then
  echo "dist directory missing: ${dist_dir}" >&2
  exit 1
fi

chmod +x "${root}/scripts/write-sha256sums.sh"
"${root}/scripts/write-sha256sums.sh" "${dist_dir}"

required=(
  syllabix-Linux-x86_64
  syllabix-Darwin-arm64
  syllabix-Darwin-x86_64
  syllabix-Windows-x86_64.exe
  SHA256SUMS
)
files=()
for name in "${required[@]}"; do
  path="${dist_dir}/${name}"
  if [[ ! -f "${path}" ]]; then
    echo "missing ${path} (build each launch target into dist/ first)" >&2
    exit 1
  fi
  files+=("${path}")
done

shopt -s nullglob
sidecars=("${dist_dir}"/*.repro.json)
if [[ "${#sidecars[@]}" -eq 0 ]]; then
  echo "missing ${dist_dir}/*.repro.json" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh is not on PATH. Create the release in the GitHub UI and attach:" >&2
  printf '  %s\n' "${files[@]}" "${sidecars[@]}" >&2
  exit 1
fi

echo "gh release create ${tag} (cwd files below)"
gh release create "${tag}" \
  "${files[@]}" \
  "${sidecars[@]}" \
  --title "${tag}" \
  --notes "Single-file Syllabix binaries. Download for your OS, chmod +x, then ./syllabix run. Weights fetch on first run into the local cache. Verify with SHA256SUMS. No Python, pip, or API key."
