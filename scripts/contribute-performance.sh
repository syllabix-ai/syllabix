#!/usr/bin/env bash
# Run the default Syllabix fixture benchmark and contribute its JSONL.
#
# This is intentionally an opt-in evidence path, not a test or merge gate.
# It installs no toolchain or system dependency. A benchmark may contain
# measured, gate-miss, or unavailable rows; all are diagnostic evidence.
set -euo pipefail

repo="syllabix-ai/syllabix"
release_base="https://github.com/${repo}/releases/latest/download"

usage() {
  cat <<'EOF'
Usage: ./scripts/contribute-performance.sh [--from-source] [--no-submit]

Resolve a Syllabix binary, run the default fixture benchmark, and contribute
the generated docs/eval/runs/*.jsonl file.

  --from-source  Build target/release/syllabix with the installed Rust toolchain.
  --no-submit    Write and validate the JSONL, but do not commit, push, or open a PR.
  -h, --help     Show this help.

The default path installs nothing. It uses syllabix on PATH, an existing
in-tree release binary, or the matching verified GitHub Release artifact.
EOF
}

die() {
  echo "contribute-performance: $*" >&2
  exit 1
}

artifact_name() {
  local os="${1:-$(uname -s)}"
  local arch="${2:-$(uname -m)}"
  case "${os}/${arch}" in
    Darwin/arm64) echo "syllabix-Darwin-arm64" ;;
    Darwin/x86_64) echo "syllabix-Darwin-x86_64" ;;
    Linux/x86_64) echo "syllabix-Linux-x86_64" ;;
    *)
      echo "unsupported contributor platform: ${os}/${arch}" >&2
      echo "supported: Darwin/arm64, Darwin/x86_64, Linux/x86_64" >&2
      return 2
      ;;
  esac
}

fetch() {
  local url="$1"
  local destination="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${destination}"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "${destination}" "${url}"
  else
    die "need curl or wget to download the release binary"
  fi
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "need sha256sum or shasum to verify the release binary"
  fi
}

download_release_binary() {
  local destination_dir="$1"
  local asset expected actual
  asset="$(artifact_name)"

  echo "contribute-performance: downloading ${asset}" >&2
  fetch "${release_base}/${asset}" "${destination_dir}/${asset}"
  fetch "${release_base}/SHA256SUMS" "${destination_dir}/SHA256SUMS"

  expected="$(awk -v name="${asset}" '$2 == name || $2 == "*" name { print $1; exit }' "${destination_dir}/SHA256SUMS")"
  [[ -n "${expected}" ]] || die "SHA256SUMS has no entry for ${asset}"
  actual="$(file_sha256 "${destination_dir}/${asset}")"
  [[ "${actual}" == "${expected}" ]] || die "SHA-256 mismatch for ${asset}"

  chmod +x "${destination_dir}/${asset}"
  echo "${destination_dir}/${asset}"
}

manual_finish() {
  local branch="$1"
  local output="$2"
  local reason="$3"
  echo >&2
  echo "contribute-performance: ${reason}" >&2
  echo "JSONL preserved at ${output}" >&2
  echo "Finish manually from the repository root:" >&2
  printf '  git switch -c %q\n' "${branch}" >&2
  printf '  git add -- %q && git commit -m %q\n' \
    "${output}" "perf: contribute default benchmark" >&2
  printf '  git push -u origin %q\n' "${branch}" >&2
  local origin_url owner head
  origin_url="$(git remote get-url origin 2>/dev/null || true)"
  owner="$(printf '%s\n' "${origin_url}" | sed -nE 's#^(git@github.com:|https://github.com/)([^/]+)/.*#\2#p')"
  head="${branch}"
  if [[ -n "${owner}" && "${owner}" != "syllabix-ai" ]]; then
    head="${owner}:${branch}"
  fi
  echo "Then open https://github.com/${repo}/compare/main...${head}?expand=1" >&2
}

manual_push_retry() {
  local branch="$1"
  local output="$2"
  echo >&2
  echo "contribute-performance: push failed; commit and JSONL are preserved" >&2
  echo "JSONL: ${output}" >&2
  printf 'Retry with: git push -u origin %q\n' "${branch}" >&2
}

manual_pr_retry() {
  local login="$1"
  local branch="$2"
  local output="$3"
  echo >&2
  echo "contribute-performance: push succeeded, but automatic PR creation failed" >&2
  echo "JSONL: ${output}" >&2
  echo "Open https://github.com/${repo}/compare/main...${login}:${branch}?expand=1" >&2
}

main() {
  local from_source=0
  local no_submit=0
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --from-source) from_source=1 ;;
      --no-submit) no_submit=1 ;;
      -h|--help) usage; return 0 ;;
      *) usage >&2; die "unknown argument: $1" ;;
    esac
    shift
  done

  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
  local git_root
  git_root="$(git -C "${root}" rev-parse --show-toplevel 2>/dev/null)" || \
    die "run this script from a Syllabix Git checkout"
  [[ "${git_root}" == "${root}" ]] || die "script root is not the Git worktree root"
  cd "${root}"

  local temp_dir=""
  cleanup() {
    if [[ -n "${temp_dir:-}" && -d "${temp_dir}" ]]; then
      rm -rf "${temp_dir}"
    fi
  }
  trap cleanup EXIT

  local syllabix_bin=""
  if [[ "${from_source}" -eq 1 ]]; then
    command -v cargo >/dev/null 2>&1 || die "--from-source requires an existing Cargo installation"
    echo "contribute-performance: building Syllabix from source" >&2
    cargo build -p syllabix --release
    syllabix_bin="${root}/target/release/syllabix"
  elif command -v syllabix >/dev/null 2>&1; then
    syllabix_bin="$(command -v syllabix)"
    echo "contribute-performance: using syllabix from PATH (${syllabix_bin})" >&2
  elif [[ -x "${root}/target/release/syllabix" ]]; then
    syllabix_bin="${root}/target/release/syllabix"
    echo "contribute-performance: using in-tree release binary" >&2
  else
    temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/syllabix-contribute.XXXXXX")"
    syllabix_bin="$(download_release_binary "${temp_dir}")"
    echo "contribute-performance: verified downloaded release binary" >&2
  fi
  [[ -x "${syllabix_bin}" ]] || die "resolved binary is not executable: ${syllabix_bin}"

  echo "contribute-performance: running the default fixture profile" >&2
  local bench_stdout
  bench_stdout="$("${syllabix_bin}" bench)"
  printf '%s\n' "${bench_stdout}"

  local output
  output="$(printf '%s\n' "${bench_stdout}" | sed -n 's/^wrote \(docs\/eval\/runs\/.*\.jsonl\) ([0-9][0-9]* rows)$/\1/p' | tail -n 1)"
  [[ -n "${output}" ]] || die "bench completed without reporting a docs/eval/runs/*.jsonl output"
  [[ -s "${output}" ]] || die "benchmark JSONL is missing or empty: ${output}"
  case "${output}" in
    docs/eval/runs/*.jsonl) ;;
    *) die "benchmark wrote outside docs/eval/runs: ${output}" ;;
  esac

  local rows
  rows="$(wc -l <"${output}" | tr -d ' ')"
  [[ "${rows}" == "6" ]] || die "expected six scenario rows in ${output}, got ${rows}"
  echo "contribute-performance: validated ${output} (${rows} rows)" >&2

  local branch
  branch="performance/$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)-$(date -u +%Y%m%d%H%M%S)"
  if [[ "${no_submit}" -eq 1 ]]; then
    echo "contribute-performance: --no-submit; JSONL preserved at ${output}" >&2
    return 0
  fi

  if ! git diff --cached --quiet; then
    manual_finish "${branch}" "${output}" "existing staged changes prevent safe automatic submission"
    return 0
  fi
  if ! command -v gh >/dev/null 2>&1; then
    manual_finish "${branch}" "${output}" "gh is not installed"
    return 0
  fi
  if ! gh auth status >/dev/null 2>&1; then
    manual_finish "${branch}" "${output}" "gh is not logged in; run: gh auth login"
    return 0
  fi

  git switch -c "${branch}"
  git add -- "${output}"
  local staged
  staged="$(git diff --cached --name-only)"
  if [[ "${staged}" != "${output}" ]]; then
    git restore --staged -- "${output}" >/dev/null 2>&1 || true
    manual_finish "${branch}" "${output}" "refusing to commit files other than the generated JSONL"
    return 0
  fi
  git commit -m "perf: contribute default benchmark"

  if ! git push -u origin "${branch}"; then
    manual_push_retry "${branch}" "${output}"
    return 0
  fi

  local login
  login="$(gh api user --jq .login)"
  if ! gh pr create \
    --repo "${repo}" \
    --base main \
    --head "${login}:${branch}" \
    --title "Performance: default fixture run ($(uname -s) $(uname -m))" \
    --body "Generated by \`./scripts/contribute-performance.sh\`. This JSONL is diagnostic benchmark evidence, not a required test result."; then
    manual_pr_retry "${login}" "${branch}" "${output}"
    return 0
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
