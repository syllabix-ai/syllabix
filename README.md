# Syllabix

Local voice agent. One native binary. Apache-2.0.

v0 target: a stranger downloads `syllabix`, runs it, and is in a voice conversation on their laptop in under 3 minutes — no Python, pip, or API key.

This repository is the product. Founder docs live in [`syllabix-ai/syllabix_founder_documents`](https://github.com/syllabix-ai/syllabix_founder_documents). The launch contract is `V0_LAUNCH.md`.

```bash
cargo run -p syllabix -- --help
cargo run -p syllabix -- init     # optional syllabix.yaml
cargo run -p syllabix -- run      # zero-config mic + speakers with full-duplex AEC
cargo run -p syllabix -- run --barge-in     # interrupt TTS when the user speaks
# per-turn timeline + WAVs are yaml diagnostics (see "Diagnostics" below)
```

## 3-minute path

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
chmod +x syllabix
./syllabix run
```

Windows: download `syllabix-Windows-x86_64.exe` from the same Release. First `run` fills `~/.cache/syllabix/models/v1` (or `%LOCALAPPDATA%\syllabix\cache\models\v1`). A second `run` must work with the network blocked.

GitHub Release files are the four sequence-22 names plus `SHA256SUMS`. `.github/workflows/release.yml` publishes them on `v*` tags. Pull requests do not package dist binaries. If Actions cannot run, build each target with `scripts/package-release.sh` and attach with `scripts/publish-release.sh v0.1.0`. After a Release is published, validate the documented download path against it: `SMOKE_RELEASE_URL=https://github.com/syllabix-ai/syllabix/releases/download/<tag> scripts/smoke-setup.sh syllabix-Linux-x86_64` (repeat per OS you can touch).

| Target | Artifact |
| --- | --- |
| Linux x64 | `syllabix-Linux-x86_64` |
| macOS Apple Silicon | `syllabix-Darwin-arm64` |
| macOS Intel | `syllabix-Darwin-x86_64` |
| Windows x64 | `syllabix-Windows-x86_64.exe` |

```bash
./scripts/package-release.sh                         # host triple
./scripts/package-release.sh x86_64-unknown-linux-gnu
./scripts/check-clean-artifact.sh dist/syllabix-Linux-x86_64
./scripts/smoke-offline-setup.sh dist/syllabix-Linux-x86_64
./scripts/check-repro.sh                             # two isolated dist builds; run when dist packaging changes
./scripts/publish-release.sh v0.1.0                  # attach dist/ when Actions cannot run
```

`cargo build -p syllabix --profile dist` is thin-LTO, one codegen unit, debuginfo stripped. Default `release` is unchanged for `ci-local.sh`. The executable does **not** pack Silero / Whisper / Llama / Kokoro weights; first `run` fetches them into `$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1` (`%LOCALAPPDATA%\syllabix\cache\models\v1` on Windows), verifies SHA-256, and reuses the cache offline. `--help` / `init` do not need the cache.

From this checkout, `cargo run -p syllabix -- run` talks on a machine with a microphone and speakers. WebRTC AEC3 echo control is on by default and calibrates automatically from the samples sent to the speakers while the microphone stays open. If the terminal reports a lost speaker reference or the laptop still self-interrupts, use headphones and include the device names from the startup line in a bug report.

## CLI

| Command | Now | Launch |
| --- | --- | --- |
| `syllabix run` | Zero-config local mic/speaker conversation, full-duplex AEC, and TUI timings | Same |
| `syllabix run --barge-in` | Opt-in. VAD keeps running during TTS; user SpeechStart stops playback, flushes queued audio, and cancels LLM/TTS. Off by default. With diagnostics enabled, interrupted turns are dumped too. Whisper utterances always include 200 ms of post-AEC preroll. | Same |
| `syllabix init [dir]` | Optional `syllabix.yaml` scaffold | Same |

There is no `serve` or `bench`. The default `run` needs no yaml, no API key, and no network after the first-run cache fills. If `syllabix.yaml` is present, it must name a known stack. `pipeline.stt.model` selects the whisper.cpp weights: `small` (default), `medium`, `large-v3-turbo`, or the published quantizations `medium-q5_0` / `large-v3-turbo-q5_0`; first run fetches only the selected id. `pipeline.stt.language` is a whisper-supported ISO code (`en`, `fr`, `de`, `ja`, …) or `auto` — with `auto`, the detected language shows in the TUI and diagnostics sidecar, and the agent replies in that language. Optional `pipeline.vad` keys (`threshold`, `min_speech_ms`, `end_silence_ms`, `preroll_ms`) tune Silero; omit them for the launch defaults. A minimal example lives at [`examples/demo-agent.yaml`](examples/demo-agent.yaml).

### Diagnostics: per-turn timeline and WAVs

Turn diagnostics are yaml-only — there is no CLI flag:

```yaml
name: demo-agent
# ... pipeline as above ...
diagnostics:
  timestamps: true            # write target/turn-debug/turn-*/turn.json sidecars
  audio: true                 # additionally write capture/clean/utterance/tts WAVs (implies timestamps)
  directory: target/turn-debug   # optional; this default needs no key
```

Default `run` writes nothing: with both keys false the directory is not even created. The full field guide — timeline anchors, a sample `summarize-timelines.py` report, and how to read it — lives in [`docs/diagnostics.md`](docs/diagnostics.md); segment formulas and the p50/p95 protocol are in [`docs/reference-profiles.md`](docs/reference-profiles.md) §10.

### Optional: BYO-key online LLM (v0.1)

The LLM slot has two execution models: `provider: local` (default) runs weights in-process from the first-run cache; `provider: online` streams from any OpenAI-compatible `chat/completions` endpoint. Audio never leaves the machine either way — with `online`, only transcript text reaches the server you choose:

```yaml
name: demo-agent
pipeline:
  vad:
    provider: silero
  stt:
    provider: whisper.cpp
    model: small
    language: en
  llm:
    provider: online                          # local (default) | online
    model: gpt-4o-mini                        # any id the endpoint serves
    base_url: https://api.openai.com/v1       # required for online
  tts:
    provider: local                           # local only today
    model: kokoro                             # kokoro (default) | qwen3-0.6 | qwen3-1.7
    language: en                              # optional; Qwen3-TTS voice language
```

Endpoint examples: `https://api.openai.com/v1` (OpenAI), `https://api.groq.com/openai/v1` (Groq), `http://127.0.0.1:11434/v1` (Ollama), or any vLLM / llama-server URL. Nothing is defaulted for you: `online` without an explicit `base_url` fails at config load, and `local` rejects the field.

**TTS models:** `tts.provider` uses the same posture words as the LLM — `local` runs weights in-process; `online` is reserved for a future cloud TTS row and fails fast. Under `local`, `tts.model` picks the voice engine: `kokoro` (default, ~310 MB ONNX) or one of two Apache-2.0 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) backbones — `qwen3-0.6` (~344 MB fetch) and `qwen3-1.7` (~1.5 GB with its speech tokenizer), fetched on first use only when selected. The Qwen engines are LM-based, so they read numbers and currency the way a listener expects (`100` → "one hundred"), speak ten languages via `tts.language` (en, zh, de, it, pt, es, fr, ja, ko, ru), and hold **one steady voice across every sentence**: at load the engine synthesizes a short reference clip from a pinned seed and conditions all speech on it, so the voice no longer drifts between utterances or runs. Kokoro stays the zero-config default.

- **The API key comes from the environment only:** it is never read from `syllabix.yaml` or a `.env` file, so project folders stay shareable and secret-free. Two ways to supply it:

  ```bash
  # one-off run — nothing is persisted:
  SYLLABIX_LLM_API_KEY=sk-… syllabix run

  # current shell session (add to your shell profile to persist):
  export SYLLABIX_LLM_API_KEY=sk-…
  ```

  With `provider: online` set and no key exported, `run` fails fast before opening any device.
- **Keyless loopback endpoints:** Ollama, llama-server, and vLLM ignore auth — pass any placeholder value, e.g. `SYLLABIX_LLM_API_KEY=ollama syllabix run` with `base_url: http://127.0.0.1:11434/v1`. Your transcript still never leaves the machine.
- Barge-in cancels the in-flight stream through the same cancel path as local generation. A failed turn speaks a short fallback ("Sorry, I could not reach the language model.") instead of hanging — there is no auto-retry. Hard connect/idle timeouts bound every turn.
- VAD, AEC, STT, and TTS stay local in this mode; no cloud STT/TTS exists. Diagnostics sidecars (`diagnostics:` in yaml) record the endpoint, model id, and the provider's request id for diagnosis.

## Develop

Requires Rust 1.91+, CMake, and a C++ compiler. whisper.cpp and llama.cpp share one `ggml` compiled into the binary (Darwin Metal + Accelerate; Linux/Windows portable CPU). Linux contributors also need ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/ci-local.sh   # Linux stand-in for GitHub Actions (fmt through tests, then Linux dist + clean-machine + README smoke)
./scripts/check-repro.sh  # two dist builds; run when the dist profile or packaging script changes
# `cargo llvm-cov --workspace --fail-under-lines 85` skips native inference (`cfg(coverage)`). Run `cargo test` for Whisper/Llama/Kokoro.
```

Reference hardware profiles and measurement protocols (AEC zero-false-turns gate, barge-in p95, TTS→ASR intelligibility): [`docs/reference-profiles.md`](docs/reference-profiles.md).

Model weights are not in git and are not packed into the `dist` executable. A versioned manifest lists Silero, the whisper.cpp STT menu (`small` default, `medium`, `large-v3-turbo`, `medium-q5_0`, `large-v3-turbo-q5_0`), Llama 3.2 1B (default), Qwen3.5-0.8B, Qwen3.5-2B, and Kokoro. Zero-config `run` fetches only the selected ids (whisper `small` and `llama-3.2-1b` unless yaml sets `pipeline.stt.model` / `pipeline.llm.model`). Thinking is off unless yaml sets `pipeline.llm.thinking: true` (Qwen). CPU vs Metal tok/s for the three GGUFs is in `vendor/llama-bench.md`, not here. The cache writes into `$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1`, verifies SHA-256, and reuses files offline. `syllabix run` fills that cache on first launch.

CI records and plays a WAV fixture (no microphone). That path also soaks 30 minutes of *audio time* through bounded queues faster than real time. A six-turn native loop test runs Silero → whisper.cpp → llama.cpp → Kokoro through fixture capture/playback. The first STT/LLM/TTS/VAD test run fetches weights into the model cache.

On a laptop with a mic and speakers:

```bash
cargo run -p syllabix -- run
cargo test -p syllabix-core --test audio_io hardware_record_and_play_if_devices_exist -- --ignored --nocapture
```

PR 18's echo gate needs a quiet laptop with its built-in microphone and speakers selected. Do not wear headphones or speak during this command. It allows 10 seconds for automatic calibration, then plays the versioned speech fixture continuously for 1 minute and requires Silero to detect zero false user turns:

```bash
cargo test -p syllabix-core --test audio_io hardware_aec_1_minute_playback_has_zero_false_turns -- --ignored --exact --nocapture
```

The microphone remains open throughout the test. Muting capture during playback does not pass this gate. The test writes `render.wav`, `capture.wav`, `clean.wav`, and `sidecar.json` to `$SYLLABIX_AEC_DEBUG_DIR` or `target/aec-debug`.

To dump a live conversation for diagnosis (listen to `utterance.wav` against STT text and `tts.wav` against the LLM reply), drop this next to your run and start it — see [`docs/diagnostics.md`](docs/diagnostics.md) for the full field guide:

```yaml
# syllabix.yaml
diagnostics:
  timestamps: true
  audio: true
  directory: target/turn-debug
```

```bash
cargo run -p syllabix --release -- run
```

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `error: No speakers found` / no microphone | `syllabix run` needs input **and** output devices at launch. Check OS sound settings, then rerun. On a headless box there is nothing to talk to — failing fast is correct. |
| The agent interrupts itself on laptop speakers | Full-duplex AEC3 is on by default and calibrates automatically for ~10 s; let calibration finish before speaking. If it still self-interrupts, use headphones and include the device names from the startup line in a bug report. |
| First `run` is slow | It fetches Silero, Whisper `small`, Llama 3.2 1B, and Kokoro into the model cache with progress lines. Later runs reuse the cache and never touch the network. |
| Replies are cut off mid-sentence when you speak over them | That is barge-in — but only with `run --barge-in`, which keeps VAD listening during TTS. Without the flag the agent finishes its sentence first; that is the default, not a bug. |
| The agent speaks Qwen's reasoning aloud | It should not: `<think>…</think>` is stripped before TTS and hidden in the TUI. Thinking stays off unless yaml sets `pipeline.llm.thinking: true`. If you hear chain-of-thought, capture it with diagnostics enabled (`turn.json` keeps the full `llm_text`) and file a bug. |
| STT text does not match what you said | Enable `diagnostics: {timestamps: true, audio: true}`, then listen to `utterance.wav` (what Whisper received, including the 200 ms onset preroll) versus `clean.wav` (post-AEC). If `utterance.wav` sounds wrong but `clean.wav` sounds right, report the sidecar `stt_text` plus both files. |
| `SYLLABIX_LLM_API_KEY is required…` at `run` start | Your yaml selects `pipeline.llm.provider: online`. Supply the key for one run (`SYLLABIX_LLM_API_KEY=sk-… syllabix run`) or export it (`export SYLLABIX_LLM_API_KEY=sk-…`). Keys are read from the environment only — never from yaml or a `.env` file. Switch the provider back to `local` for the on-device LLM. |
| Every reply is "Sorry, I could not reach the language model." | The cloud turn failed (bad key → HTTP 401, endpoint down, or idle timeout) and the turn spoke its fallback instead of hanging. Check the key, the `base_url`, and the endpoint's status; diagnostics sidecars carry the endpoint and request id. |
| Linux build fails linking ALSA | Contributors need `libasound2-dev` (a declared OS library). Users of the Release binary never compile anything. |
| macOS asks to approve microphone access | Grant it once in System Settings → Privacy & Security → Microphone; the binary requests access through CoreAudio. |
| Downloaded binary won't verify | Re-download the artifact and `SHA256SUMS` from the same Release; the checksum command must pass before you run anything. |

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
