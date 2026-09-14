# FAQ

## Does it work on Apple Silicon and Intel Macs?

Both have Release artifacts (`syllabix-Darwin-arm64`, `syllabix-Darwin-x86_64`). Apple Silicon uses Metal + Accelerate for Whisper and the local LLM. Intel Macs run the portable CPU path. Unsigned downloads need a Gatekeeper exception until artifacts are notarized — [install/macos.md](install/macos.md).

## Does it work on Windows and Linux?

Yes: `syllabix-Windows-x86_64.exe` and `syllabix-Linux-x86_64`. Those builds use CPU for llama.cpp / whisper.cpp. Linux users of the **binary** do not install ALSA headers; contributors building from source need `libasound2-dev`.

## How much RAM and disk do I need?

Plan on 16 GB RAM and ~3 GB disk for the default stack. 8 GB is tight. Qwen TTS ids need several more gigabytes. [requirements.md](requirements.md).

## Why is the first run slow? Is it three minutes to first speech?

First `run` downloads ~2.2 GB and verifies SHA-256. Cold-start wall time (binary on disk → first spoken reply) is **not published**. Do not plan a three-minute gate until [reference-profiles.md](reference-profiles.md) records it.

## Do I need Python, pip, Ollama, or an API key?

Not for default `run`. An online LLM is yaml opt-in and then needs `SYLLABIX_LLM_API_KEY`. Ollama is only relevant if you set `provider: online` and a local `base_url`.

## Is there a REST API or `serve` command?

No. [api.md](api.md).

## The agent talks over itself / barges in on its own voice.

AEC3 calibrates ~10 s with the mic open. Wait before speaking. If laptop speakers still false-trigger Silero, use headphones and include startup device names in a bug report. [troubleshooting.md](troubleshooting.md).

## Replies stop when I talk. Is that a bug?

Only if you did **not** pass `--barge-in`. With the flag, talking over the agent is the feature. Without it, the agent should finish the sentence.

## Can I clone my voice or dub a video?

No. Pocket TTS uses a fixed English voice. This is a conversation agent, not a studio. [compare.md](compare.md).

## Does Syllabix send audio to the cloud?

Not on the default local stack. `provider: online` sends **transcript text** to `base_url`. PCM stays local.

## How do I see what Whisper heard?

Enable diagnostics in `syllabix.yaml` and listen to `utterance.wav` vs `clean.wav`. [diagnostics.md](diagnostics.md).

## How do I remove downloaded models?

Delete the cache directory (`~/.cache/syllabix/models/v1`, or `%LOCALAPPDATA%\syllabix\cache\models\v1`, or `$SYLLABIX_CACHE_DIR/models/v1`). The binary itself is the file you downloaded.

## Can I use it commercially?

The application is Apache-2.0. Downloaded GGUF / ONNX weights keep upstream licenses. Review those terms for the ids you select. Attribution for vendored native code is in [NOTICE](../NOTICE).
