#!/usr/bin/env bash
# Build one installable `dist` profile executable and write dist/<name> plus
# a reproducibility sidecar. Weights are not packed into the binary.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "${root}"

host="$(rustc -vV | awk '/^host:/{print $2}')"
target="${1:-${host}}"

artifact_name() {
  case "$1" in
    x86_64-unknown-linux-gnu) echo "syllabix-Linux-x86_64" ;;
    aarch64-apple-darwin) echo "syllabix-Darwin-arm64" ;;
    x86_64-apple-darwin) echo "syllabix-Darwin-x86_64" ;;
    x86_64-pc-windows-msvc) echo "syllabix-Windows-x86_64.exe" ;;
    *)
      echo "unsupported launch target: $1" >&2
      echo "expected one of: x86_64-unknown-linux-gnu aarch64-apple-darwin x86_64-apple-darwin x86_64-pc-windows-msvc" >&2
      return 1
      ;;
  esac
}

name="$(artifact_name "${target}")"
dist_dir="${DIST_DIR:-${root}/dist}"
mkdir -p "${dist_dir}"

export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
if [[ -z "${SOURCE_DATE_EPOCH:-}" ]]; then
  SOURCE_DATE_EPOCH="$(git -C "${root}" log -1 --pretty=%ct)"
  export SOURCE_DATE_EPOCH
fi
export RUSTFLAGS="${RUSTFLAGS:--D warnings} --remap-path-prefix=${root}=."

echo "== cargo build -p syllabix --profile dist --target ${target} =="
cargo build -p syllabix --profile dist --target "${target}"

src_dir="${CARGO_TARGET_DIR:-${root}/target}/${target}/dist"
src="${src_dir}/syllabix"
if [[ ! -f "${src}" && -f "${src}.exe" ]]; then
  src="${src}.exe"
fi
if [[ ! -f "${src}" ]]; then
  echo "dist binary missing at ${src_dir}/syllabix(.exe)" >&2
  exit 1
fi

dest="${dist_dir}/${name}"
cp -f "${src}" "${dest}"
chmod +x "${dest}" || true

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

size="$(wc -c <"${dest}" | tr -d ' ')"
sha="$(file_sha256 "${dest}")"
lock_sha="$(file_sha256 "${root}/Cargo.lock")"
git_sha="$(git -C "${root}" rev-parse HEAD)"
rustc_ver="$(rustc --version)"
cargo_ver="$(cargo --version)"

sidecar="${dest}.repro.json"
cat >"${sidecar}" <<EOF
{
  "git_sha": $(json_escape "${git_sha}"),
  "rustc": $(json_escape "${rustc_ver}"),
  "cargo": $(json_escape "${cargo_ver}"),
  "target": $(json_escape "${target}"),
  "profile": "dist",
  "artifact": $(json_escape "${name}"),
  "size_bytes": ${size},
  "sha256": $(json_escape "${sha}"),
  "cargo_lock_sha256": $(json_escape "${lock_sha}"),
  "source_date_epoch": $(json_escape "${SOURCE_DATE_EPOCH}")
}
EOF

echo "wrote ${dest} (${size} bytes, sha256 ${sha})"
echo "wrote ${sidecar}"

chmod +x "${root}/scripts/write-sha256sums.sh"
"${root}/scripts/write-sha256sums.sh" "${dist_dir}"
