# Syllabix

Local voice agent. One native binary. Apache-2.0.

v0 target: a stranger downloads `syllabix`, runs it, and is in a voice conversation on their laptop in under 3 minutes — no Python, pip, or API key.

This repository is the product. Founder docs live in [`syllabix-ai/syllabix_founder_documents`](https://github.com/syllabix-ai/syllabix_founder_documents). The launch contract is `V0_LAUNCH.md`.

```bash
cargo run -p syllabix -- --help
cargo run -p syllabix -- run    # exits 2 until the conversation loop lands
```

## Intended 3-minute path (not shipping yet)

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
chmod +x syllabix
./syllabix run
```

GitHub Releases are not published yet. Do not expect a spoken reply from this checkout.

## CLI

| Command | Now | Launch |
| --- | --- | --- |
| `syllabix run` | Not implemented | Zero-config local mic/speaker conversation |
| `syllabix init [dir]` | Not implemented | Optional `syllabix.yaml` scaffold |

There is no `serve`, `bench`, cloud provider, or API key in v0.

## Develop

Requires Rust 1.83+, CMake, and a C++ compiler. whisper.cpp and llama.cpp share one CPU `ggml` compiled into the binary. Linux contributors also need ALSA headers (`libasound2-dev`).

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/ci-local.sh   # Linux stand-in for GitHub Actions
```

Model weights are not in git. A versioned manifest lists Silero, whisper.cpp `small`, llama-3.2-1b, and Kokoro. The cache writes into `$SYLLABIX_CACHE_DIR` or `~/.cache/syllabix/models/v1`, verifies SHA-256, and reuses files offline. `syllabix run` does not fetch yet.

CI records and plays a WAV fixture (no microphone). That path also soaks 30 minutes of *audio time* through bounded queues faster than real time. whisper.cpp `small` is compiled into the binary; the first STT test run fetches `ggml-small.bin` into the model cache.

On a laptop with a mic and speakers:

```bash
cargo test -p syllabix-core --test audio_io hardware_record_and_play_if_devices_exist -- --ignored --nocapture
```

## License

Apache-2.0. See [LICENSE](LICENSE).
