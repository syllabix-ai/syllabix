#!/usr/bin/env bash
# Type-check a throwaway crate that depends on syllabix-stt / syllabix-tts the
# way an external host would (path or git), outside the Syllabix workspace.
#
# Usage:
#   ./scripts/ci-speech-consumer.sh
#   ./scripts/ci-speech-consumer.sh --prepare
#   ./scripts/ci-speech-consumer.sh --check
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
work="${root}/target/speech-consumer"
mode="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare) mode="prepare" ;;
    --check) mode="check" ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $1 (try --help)" >&2
      exit 2
      ;;
  esac
  shift
done

toml_escape() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  printf '%s' "$s"
}

prepare() {
  rm -rf "${work}"
  mkdir -p "${work}/src"
  cp "${root}/rust-toolchain.toml" "${work}/rust-toolchain.toml"

  local stt_path tts_path
  stt_path="$(toml_escape "${root}/crates/syllabix-stt")"
  tts_path="$(toml_escape "${root}/crates/syllabix-tts")"

  cat > "${work}/Cargo.toml" <<EOF
# Empty [workspace] keeps this crate out of the Syllabix workspace.
[workspace]

[package]
name = "syllabix-speech-consumer"
version = "0.0.0"
edition = "2021"
publish = false
description = "CI stand-in for an out-of-tree host of syllabix-stt/tts. Not a workspace member."

[dependencies]
syllabix-stt = { path = "${stt_path}" }
syllabix-tts = { path = "${tts_path}" }
EOF

  cat > "${work}/src/main.rs" <<'EOF'
//! Type-check the public load-and-run API without downloading models.
fn main() {
    speech_api_type_check();
}

fn speech_api_type_check() {
    use std::path::Path;

    use syllabix_stt::{
        fluid_gguf::FluidGgufStt,
        moonshine::MoonshineStt,
        remote::{RemoteStt, RemoteSttSettings},
        transcribe_wav,
        HttpFetcher, Manifest, ModelAsset, ModelCache, SttModel, DEFAULT_SAMPLE_RATE_HZ,
    };
    use syllabix_tts::{
        pocket::PocketTts, remote::RemoteTtsSettings, speak_to_wav, QwenTts, TtsModel,
        Manifest as TtsManifest, ModelAsset as TtsModelAsset, ModelCache as TtsModelCache,
        DEFAULT_CHANNELS,
    };

    fn check_transcribe(
        stt: &mut RemoteStt,
        path: &Path,
        cancel: &syllabix_stt::Cancel,
    ) -> syllabix_stt::Result<syllabix_stt::Transcript> {
        transcribe_wav(stt, path, cancel)
    }

    fn check_speak(
        tts: &mut PocketTts,
        text: &str,
        path: &Path,
        cancel: &syllabix_tts::Cancel,
    ) -> syllabix_tts::Result<()> {
        speak_to_wav(tts, text, path, cancel)
    }

    fn check_stt_settings(base: &str, model: &str, language: &str) -> RemoteSttSettings {
        RemoteSttSettings::new(base, model, language)
    }

    fn check_tts_settings(base: &str, model: &str, voice: &str) -> RemoteTtsSettings {
        RemoteTtsSettings::new(base, model, voice)
    }

    let _ = MoonshineStt::load;
    let _ = FluidGgufStt::load;
    let _ = check_transcribe;
    let _ = check_stt_settings;
    let _ = check_custom_stt_cache;
    let _ = check_custom_tts_cache;
    let _ = ModelCache::v0;
    let _ = HttpFetcher::new;

    fn check_custom_stt_cache(root: std::path::PathBuf) -> ModelCache {
        let manifest = Manifest {
            version: 1,
            assets: Vec::<ModelAsset>::new(),
        };
        ModelCache::new(root, manifest)
    }

    fn check_custom_tts_cache(root: std::path::PathBuf) -> TtsModelCache {
        let manifest = TtsManifest {
            version: 1,
            assets: Vec::<TtsModelAsset>::new(),
        };
        TtsModelCache::new(root, manifest)
    }
    let _ = DEFAULT_SAMPLE_RATE_HZ;
    let _ = SttModel::MoonshineStreamingSmall;
    let _ = SttModel::FluidSpeechProGgufQ8;

    let _ = PocketTts::load;
    let _ = check_speak;
    let _ = check_tts_settings;
    let _ = QwenTts::load;
    let _ = syllabix_tts::IndicMioTts::load;
    let _ = TtsModel::PocketTts;
    let _ = TtsModel::IndicMio;
    let _ = DEFAULT_CHANNELS;
}
EOF

  echo "speech-consumer crate at ${work}"
}

check() {
  if [[ ! -f "${work}/Cargo.toml" ]]; then
    echo "missing ${work}/Cargo.toml; run $0 --prepare first" >&2
    exit 1
  fi
  (
    cd "${work}"
    cargo check
  )
}

case "${mode}" in
  prepare) prepare ;;
  check) check ;;
  all)
    prepare
    check
    ;;
esac