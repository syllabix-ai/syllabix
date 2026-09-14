# Install on Linux

Artifact: `syllabix-Linux-x86_64`.

```bash
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-$(uname -s)-$(uname -m) -o syllabix
curl -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
chmod +x syllabix
./syllabix run
```

`uname -s` / `uname -m` must be `Linux` / `x86_64`. There is no ARM64 Release artifact.

## Audio

`run` needs ALSA/Pulse/PipeWire devices the same way any desktop app does. Headless boxes with no capture or playback fail fast (`No speakers found` / no microphone) — that is expected.

Users of the Release binary do **not** compile anything and do not install `libasound2-dev`. That package is only for building from source.

## Models

First `run` fetches the default stack (~2.2 GB) into `~/.cache/syllabix/models/v1` (or `$SYLLABIX_CACHE_DIR/models/v1`). The Linux artifact uses portable CPU for whisper.cpp and llama.cpp.

Cache and other OS: [../install.md](../install.md).
