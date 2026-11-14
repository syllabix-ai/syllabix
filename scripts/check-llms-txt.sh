#!/usr/bin/env bash
# Fail if llms.txt omits a top-level docs/*.md page, if an in-repo
# Syllabix URL in llms.txt points at a missing path, or if a relative
# link in README.md's For AI agents section is broken. Offline.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"

llms="${root}/llms.txt"
readme="${root}/README.md"
fail=0

if [[ ! -f "${llms}" ]]; then
  echo "llms.txt is missing" >&2
  exit 1
fi

shopt -s nullglob
for page in docs/*.md; do
  if ! grep -F -q "${page}" "${llms}"; then
    echo "llms.txt does not mention ${page}" >&2
    fail=1
  fi
done

while IFS= read -r url; do
  [[ -z "${url}" ]] && continue
  path="${url}"
  path="${path#https://raw.githubusercontent.com/syllabix-ai/syllabix/main/}"
  path="${path#https://github.com/syllabix-ai/syllabix/tree/main/}"
  path="${path#https://github.com/syllabix-ai/syllabix/tree/main}"
  path="${path%%#*}"
  path="${path%%\?*}"
  path="${path%/}"
  if [[ -z "${path}" ]]; then
    continue
  fi
  if [[ ! -e "${root}/${path}" ]]; then
    echo "llms.txt URL is missing from the checkout: ${url}" >&2
    fail=1
  fi
done < <(grep -oE 'https://(raw\.githubusercontent\.com/syllabix-ai/syllabix/main/|github\.com/syllabix-ai/syllabix/tree/main/?)[^)[:space:]"]+' "${llms}" || true)

if [[ ! -f "${readme}" ]]; then
  echo "README.md is missing" >&2
  exit 1
fi

section="$(awk '
  $0 == "## For AI agents" {on=1; next}
  on && /^## / {exit}
  on {print}
' "${readme}")"

if [[ -z "${section}" ]]; then
  echo "README.md is missing a ## For AI agents section" >&2
  exit 1
fi

while IFS= read -r dest; do
  [[ -z "${dest}" ]] && continue
  dest="${dest%%\"*}"
  dest="${dest%%\'*}"
  dest="${dest%% *}"
  dest="${dest%%#*}"
  dest="${dest%%\?*}"
  [[ -z "${dest}" ]] && continue
  case "${dest}" in
    http://*|https://*|mailto:*|\#*) continue ;;
  esac
  if [[ ! -e "${root}/${dest}" ]]; then
    echo "README For AI agents link is missing: ${dest}" >&2
    fail=1
  fi
done < <(printf '%s\n' "${section}" | grep -oE '\[[^]]*\]\([^)]+\)' | sed -E 's/^\[[^]]*\]\(//; s/\)$//' || true)

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo "llms.txt and README For AI agents links are complete"
