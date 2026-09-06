#!/usr/bin/env bash
# Submit one immutable component-benchmark run without requiring a compiler.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
from_source=false
if [[ "${1:-}" == "--from-source" ]]; then
  from_source=true
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--from-source]" >&2
  exit 2
fi

cd "$root"
validator="$root/scripts/validate-performance-ledger.py"
if [[ ! -x "$validator" ]]; then
  echo "missing executable validator: $validator" >&2
  exit 1
fi

artifact_name() {
  case "$(uname -s)/$(uname -m)" in
    Linux/x86_64) echo syllabix-Linux-x86_64 ;;
    Darwin/arm64) echo syllabix-Darwin-arm64 ;;
    Darwin/x86_64) echo syllabix-Darwin-x86_64 ;;
    *) echo "unsupported release target: $(uname -s)/$(uname -m)" >&2; return 1 ;;
  esac
}

verify_sha256() {
  local asset="$1" sums="$2"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "$asset")" && sha256sum --ignore-missing -c "$sums")
  elif command -v shasum >/dev/null 2>&1; then
    (cd "$(dirname "$asset")" && shasum -a 256 -c "$sums" --ignore-missing)
  else
    echo "need sha256sum or shasum to verify the release binary" >&2
    return 1
  fi
}

download_release() {
  local destination="$1" asset
  asset="$(artifact_name)"
  local base="https://github.com/syllabix-ai/syllabix/releases/latest/download"
  if command -v curl >/dev/null 2>&1; then
    if ! curl -fsSL "$base/$asset" -o "$destination/$asset" \
      || ! curl -fsSL "$base/SHA256SUMS" -o "$destination/SHA256SUMS"; then
      echo "no matching published Release is available for ${asset}" >&2
      return 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -qO "$destination/$asset" "$base/$asset" \
      || ! wget -qO "$destination/SHA256SUMS" "$base/SHA256SUMS"; then
      echo "no matching published Release is available for ${asset}" >&2
      return 1
    fi
  else
    echo "need curl or wget to download a release binary" >&2
    return 1
  fi
  verify_sha256 "$destination/$asset" "$destination/SHA256SUMS" >&2
  chmod +x "$destination/$asset"
  printf '%s\n' "$destination/$asset"
}

tmp="$(mktemp -d "${TMPDIR:-/tmp}/syllabix-performance.XXXXXX")"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

if "$from_source"; then
  command -v cargo >/dev/null 2>&1 || { echo "--from-source needs cargo" >&2; exit 2; }
  echo "== developer fallback: verify then build the local source =="
  cargo test -p syllabix --test cli
  cargo build -p syllabix --release
  binary="$root/target/release/syllabix"
elif [[ -x "$root/target/release/syllabix" ]]; then
  binary="$root/target/release/syllabix"
elif [[ -x "$root/target/debug/syllabix" ]]; then
  binary="$root/target/debug/syllabix"
elif command -v syllabix >/dev/null 2>&1; then
  binary="$(command -v syllabix)"
else
  echo "== downloading the matching verified Syllabix Release =="
  if ! binary="$(download_release "$tmp")"; then
    cat >&2 <<EOF
Cannot continue without a compatible binary. Install syllabix on PATH, wait
for a matching GitHub Release, or use the developer fallback:
  ./scripts/contribute-performance.sh --from-source
EOF
    exit 1
  fi
fi

echo "== component benchmarks for every supported local model (no microphone or speakers) =="
raw_run="$tmp/run.jsonl"
"$binary" bench --out "$raw_run"
fingerprint="$($validator --fingerprint "$raw_run")"
run_path="docs/eval/runs/$fingerprint.jsonl"
mkdir -p docs/eval/runs
if [[ -e "$run_path" ]]; then
  if git ls-files --error-unmatch "$run_path" >/dev/null 2>&1; then
    echo "a published run for this machine/build fingerprint already exists: $run_path" >&2
    exit 1
  fi
  existing_fingerprint="$($validator --fingerprint "$run_path")"
  if [[ "$existing_fingerprint" != "$fingerprint" ]]; then
    echo "existing contribution run has a mismatched fingerprint: $run_path" >&2
    exit 1
  fi
  echo "reusing uncommitted contribution run at $run_path"
else
  mv "$raw_run" "$run_path"
  echo "wrote $run_path"
fi

if ! command -v gh >/dev/null 2>&1 || ! gh auth status --hostname github.com >/dev/null 2>&1; then
  cat <<EOF
GitHub CLI is unavailable or not authenticated; the JSONL was preserved.
Finish with:
  git switch -c perf/$fingerprint
  git add $run_path && git commit -m 'perf(bench): add contribution run'
  gh pr create --base main --title 'perf(bench): add contribution run' --body 'Closes #45'
EOF
  exit 0
fi

branch="perf/$fingerprint"
git fetch origin main
if ! git merge-base --is-ancestor HEAD origin/main; then
  cat >&2 <<EOF
The contribution workflow is not merged into origin/main yet. The JSONL is
preserved at $run_path; do not submit it until B5 has merged.
EOF
  exit 0
fi
if git show-ref --verify --quiet "refs/heads/$branch"; then
  echo "benchmark JSONL is preserved at $run_path; branch $branch already exists." >&2
  exit 0
fi
git switch -c "$branch" origin/main
git add "$run_path"
git commit --only "$run_path" -m "perf(bench): add contribution run"
git push -u origin "$branch"
gh pr create --base main --head "$branch" --title "perf(bench): add contribution run" --body "Closes #45"
