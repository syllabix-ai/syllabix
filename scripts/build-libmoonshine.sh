#!/usr/bin/env bash
# Build libmoonshine.so for Syllabix official medium STT (C API).
#
# Does NOT patch moonshine sources. Neutralizes -Werror via CMAKE_CXX_FLAGS /
# target compile options, and disables TTS ONNX to avoid git-LFS voice assets.
#
# Usage:
#   ./scripts/build-libmoonshine.sh [/path/to/moonshine-src] [/path/to/build-dir]
# Env:
#   MOONSHINE_SRC   default: vendor/moonshine or ../../moonshine-src
#   MOONSHINE_BUILD default: /tmp/moonshine-build-syllabix

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-${MOONSHINE_SRC:-}}"
if [[ -z "${SRC}" ]]; then
  if [[ -d "${ROOT}/vendor/moonshine/core" ]]; then
    SRC="${ROOT}/vendor/moonshine"
  elif [[ -d /workspace/moonshine-src/core ]]; then
    SRC=/workspace/moonshine-src
  else
    echo "Set MOONSHINE_SRC to a moonshine checkout (needs core/)." >&2
    exit 1
  fi
fi
BUILD="${2:-${MOONSHINE_BUILD:-/tmp/moonshine-build-syllabix}}"
mkdir -p "${BUILD}"

# Pin documented in PR notes. Prefer that commit when SRC is a git checkout.
cmake -S "${SRC}/core" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMOONSHINE_TTS_BUILD_ONNX=OFF \
  -DCMAKE_CXX_FLAGS="-Wno-error" \
  -DCMAKE_C_FLAGS="-Wno-error"

cmake --build "${BUILD}" -j"$(nproc)" --target moonshine

SO="${BUILD}/libmoonshine.so"
if [[ ! -f "${SO}" ]]; then
  SO="$(find "${BUILD}" -name 'libmoonshine.so' | head -n1)"
fi
echo "Built: ${SO}"
echo "Export for Syllabix:"
echo "  export SYLLABIX_LIBMOONSHINE=${SO}"
ORT="$(find "${SRC}/core/third-party/onnxruntime/lib" -name 'libonnxruntime.so*' | head -n1 || true)"
if [[ -n "${ORT}" ]]; then
  echo "  export LD_LIBRARY_PATH=$(dirname "${ORT}"):\${LD_LIBRARY_PATH}"
fi
