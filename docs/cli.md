# CLI

Syllabix has two user commands: `run` and `init`.

```text
syllabix --help
syllabix run [--barge-in]
syllabix init [DIR]
```

`--help` and `init` do not download weights.

## `run`

Start a local voice conversation. Zero config: no yaml file and no API key.

| Flag | Default | Effect |
| --- | --- | --- |
| (none) | — | Finish the current sentence before listening. |
| `--barge-in` | off | Keep VAD running during TTS; user speech stops playback. Press `b` in the terminal UI to toggle during a conversation. |

`syllabix.yaml` in the current working directory is loaded if present. Missing file is not an error. Diagnostics are yaml-only — there is no `--turn-debug` flag. See [diagnostics](diagnostics.md).

Online stages read keys from the environment at `run` start: `SYLLABIX_LLM_API_KEY`, `SYLLABIX_STT_API_KEY`, `SYLLABIX_TTS_API_KEY`. Never yaml. See [syllabix.yaml](syllabix-yaml.md#local-vs-online).

## `init [DIR]`

Write `syllabix.yaml` with launch defaults. `DIR` defaults to `.`. Refuses to overwrite an existing file. Keys: [syllabix.yaml](syllabix-yaml.md).

## Hidden commands

`bench` and the `bench-*-worker` helpers are contributor-only performance tools. They are hidden from `--help`. See [contributing](contributing.md#contribute-component-performance-evidence).
