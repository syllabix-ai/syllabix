#!/usr/bin/env bash
# Fail if the syllabix executable links unexpected shared libraries.
# Expected on glibc Linux: libc family + libgcc_s + the dynamic loader.
set -euo pipefail

bin="${1:-}"
if [[ -z "${bin}" || ! -f "${bin}" ]]; then
  echo "usage: $0 <path-to-syllabix>" >&2
  exit 2
fi

if ! command -v ldd >/dev/null 2>&1; then
  echo "ldd not found; skipping dynamic library check" >&2
  exit 0
fi

output="$(ldd "${bin}")"
echo "${output}"

if grep -q "statically linked" <<<"${output}"; then
  echo "binary is statically linked"
  exit 0
fi

allowed='^(linux-vdso\.so|ld-linux|ld-linux-x86-64\.so|ld-linux-aarch64\.so|libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|libgcc_s\.so)'

while read -r line; do
  [[ -z "${line}" ]] && continue
  soname="${line%% *}"
  soname="${soname##*/}"
  if [[ "${soname}" == "statically" ]]; then
    continue
  fi
  if [[ ! "${soname}" =~ ${allowed} ]]; then
    echo "undeclared dynamic dependency: ${soname}" >&2
    echo "full ldd output:" >&2
    echo "${output}" >&2
    exit 1
  fi
done < <(echo "${output}" | awk '{print $1}')
