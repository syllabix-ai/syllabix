#!/usr/bin/env bash
# Attach dist/ binaries to a GitHub Release when Actions cannot run.
# Usage: scripts/publish-release.sh v0.1.0 [dist_dir]
#
# Requires gh with permission to create releases. This script does not
# build the Darwin/Windows matrices; copy those files into dist/ first.
#
# Same rules as .github/workflows/release.yml: never delete a release.
# Immutable releases lock a published name forever.
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
  echo "gh is not on PATH. Install GitHub CLI, then re-run this script." >&2
  printf '  would attach: %s\n' "${files[@]}" "${sidecars[@]}" >&2
  exit 1
fi

notes="Single-file Syllabix binaries. Download for your OS, chmod +x, then ./syllabix run. Weights fetch on first run into the local cache. Verify with SHA256SUMS. No Python, pip, or API key."

# A brand-new draft can 404 for a second or two when we try to publish it.
publish_now() {
  local n=0
  while (( n < 5 )); do
    if gh release edit "${tag}" --draft=false; then
      return 0
    fi
    n=$((n + 1))
    echo "could not publish ${tag} yet (try ${n}/5); waiting for the draft to show up"
    sleep 2
  done
  echo "gave up publishing ${tag}" >&2
  return 1
}

if gh release view "${tag}" >/dev/null 2>&1; then
  is_draft="$(gh release view "${tag}" --json isDraft --jq .isDraft)"
  if [[ "${is_draft}" == true ]]; then
    gh release upload "${tag}" "${files[@]}" "${sidecars[@]}" --clobber
    publish_now
    exit 0
  fi
  echo "A published GitHub Release already exists for ${tag}." >&2
  echo "Someone created it by hand in the GitHub Releases page. This script has to create the release itself so the binaries are attached before GitHub locks the name." >&2
  echo "Once a name is used for a published release, GitHub never lets that name come back." >&2
  echo "Next time, push only the tag and let the Release workflow create the release." >&2
  exit 1
fi

echo "gh release create --draft ${tag} (cwd files below)"
gh release create "${tag}" \
  "${files[@]}" \
  "${sidecars[@]}" \
  --draft \
  --title "${tag}" \
  --notes "${notes}"
publish_now
