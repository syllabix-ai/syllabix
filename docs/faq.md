# FAQ

Symptom fixes (won't verify, Gatekeeper, no mic, self-interrupt): [troubleshooting](troubleshooting.md).

## Why is the first `run` so large / slow?

Default `run` fetches Silero, Whisper `small`, LFM2.5-2.6B, and Pocket TTS (~2.2 GB), then checks SHA-256. How long that takes depends on network bandwidth. Later runs reuse the cache offline. `--help` and `init` download nothing.

## Memory is tight. Can I use a smaller model?

Yes. `syllabix init`, then either set `pipeline.llm.model` to `lfm2.5-350m` or `lfm2.5-230m`, or set `pipeline.llm.provider: online` with `base_url` and `SYLLABIX_LLM_API_KEY` so the LLM runs in the cloud (transcript text only; mic audio stays local). Catalogue and keys: [engines](engines.md), [syllabix.yaml](syllabix-yaml.md).

## Why is barge-in off by default?

The default finishes the agent's sentence so open-speaker echo cannot cut it off mid-word. Pass `run --barge-in` when you want talk-over, or enable it during the conversation via the terminal UI (`b`). AEC is still on in both modes.

## Do I need headphones?

Not if AEC holds on your laptop speakers. Let the ~10 s calibration finish before talking. If the agent still treats its own voice as yours, use headphones and report the device names from the startup line.

## Where do API keys go?

Nowhere in yaml, and nowhere in a `.env` file. Only `SYLLABIX_LLM_API_KEY` in the environment, and only when yaml sets `pipeline.llm.provider: online`. Default local `run` needs no key.

## Download or compile?

Download a Release binary if you just want to talk — that is the usual install. Compile (`cargo run -p syllabix -- run`) if you are changing the code. To put a tagged CLI on your `PATH` without cloning, `cargo install --git … --tag vX.Y.Z syllabix`. Details: [install](install.md).

## Can I change the voice / model without rebuilding?

Yes. `syllabix init`, edit `pipeline.*.model` in `syllabix.yaml`. Catalogue: [engines](engines.md). Unknown ids fail at load.

## Does audio leave the machine?

Not on the default local stack once weights are cached. An online LLM sends transcript text only; the mic stream stays local.
