#!/usr/bin/env bash
# Fail unless the syllabix binary contains exactly one ggml (unprefixed
# or a single compile-time llama_ggml_* / llama_gguf_* prefix).
set -euo pipefail

bin="${1:-}"
if [[ -z "${bin}" || ! -f "${bin}" ]]; then
  echo "usage: $0 <path-to-syllabix>" >&2
  exit 2
fi

if ! command -v nm >/dev/null 2>&1; then
  echo "nm not found; cannot prove a single ggml" >&2
  exit 1
fi

# Defined symbols. Prefer portable nm -P; fall back to BSD/default.
if ! symbols="$(nm -P --defined-only "${bin}" 2>/dev/null)"; then
  symbols="$(nm -P "${bin}")"
fi

if grep -E '^_?llama_iso_' <<<"${symbols}" >/dev/null; then
  echo "forbidden post-build llama_iso_* symbols present" >&2
  exit 1
fi

count_global() {
  local name="$1"
  grep -E "^_?${name} [Tt] " <<<"${symbols}" | wc -l | tr -d ' ' || true
}

ggml_new="$(count_global ggml_new_tensor)"
pref_new="$(count_global llama_ggml_new_tensor)"
llama_info="$(count_global llama_print_system_info)"
whisper_init="$(count_global whisper_init_from_file_with_params)"
whisper_full="$(count_global whisper_full)"
shim_decode="$(count_global syllabix_whisper_decode)"

echo "ggml_new_tensor: ${ggml_new}"
echo "llama_ggml_new_tensor: ${pref_new}"
echo "llama_print_system_info: ${llama_info}"
echo "whisper_init_from_file_with_params: ${whisper_init}"
echo "whisper_full: ${whisper_full}"
echo "syllabix_whisper_decode: ${shim_decode}"

if [[ "${ggml_new}" -gt 0 && "${pref_new}" -gt 0 ]]; then
  echo "two ggml engines: ggml_new_tensor and llama_ggml_new_tensor" >&2
  exit 1
fi

if [[ "${ggml_new}" -eq 0 && "${pref_new}" -eq 0 ]]; then
  echo "no ggml engine in ${bin}" >&2
  exit 1
fi

if [[ "${ggml_new}" -gt 1 || "${pref_new}" -gt 1 ]]; then
  echo "ggml_new_tensor defined more than once (two copies linked)" >&2
  exit 1
fi

if [[ "${llama_info}" -lt 1 ]]; then
  echo "llama.cpp frontend is not linked" >&2
  exit 1
fi

if [[ "${whisper_init}" -lt 1 && "${whisper_full}" -lt 1 && "${shim_decode}" -lt 1 ]]; then
  echo "whisper.cpp frontend is not linked" >&2
  exit 1
fi

echo "one ggml (or one compile-time-prefixed copy); both frontends linked"
