# `syllabix.yaml`

Optional project file for Syllabix. **`syllabix run` needs no yaml.** Defaults talk with zero config.

| Fact | Detail |
| --- | --- |
| Name | `syllabix.yaml` (only this name is loaded) |
| Where | Current working directory when you `run` |
| Create | `syllabix init [dir]`, or copy [`examples/demo-agent.yaml`](../examples/demo-agent.yaml) |
| Missing file | Quietly uses defaults — not an error |
| Unknown keys | Fail at load |
| Not in this file | API keys, barge-in (`run --barge-in`), sample rate / frame size |

`init` writes the defaults so you can edit them. It refuses to overwrite an existing file.

## Sample

```yaml
name: demo-agent                 # agent label
pipeline:
  vad:                           # speech start/end
    provider: silero
    threshold: 0.5
    min_speech_ms: 100
    end_silence_ms: 350
    preroll_ms: 200
  stt:                           # speech → text
    provider: local
    model: whisper-small
    language: en
  llm:                           # text → reply
    provider: local
    model: lfm2.5-2.6b
    thinking: false
    system_prompt: "You are a smart assistant. This is a spoken conversation. Reply in spoken {language}, the way a person talks: brief, clear, and natural. Do not use markdown, lists, headings, or emoji."
  tts:                           # reply → speech
    provider: local
    model: pocket-tts
    language: en

# optional (init omits these):
# auto-timeout:                  # idle mic-mute / exit
# diagnostics:                   # turn timelines / WAVs
# skills:                        # local instruction packages (harness only)
```

First `run` fetches only the selected model ids (~2.2 GB for the default stack). Model catalogue (ids, sources, machine needs): [engines.md](engines.md). The rest of this page walks every key in order.

## `name`

Display label for the agent. Default: `demo-agent`.

## VAD — speech start / end

`pipeline.vad` decides when the user has started talking and when the turn is over.
Leave these alone unless you are diagnosing turn-taking.

### `provider`

Only `silero`. Required.

### `threshold`

Default `0.5`. Range `(0, 1]` (`0` fails at load).

Silero scores every audio frame from `0` to `1`. A frame counts as **speech**
when `score >= threshold`.

- Closer to `0` — very sensitive; almost everything counts as speech (fan noise, rustling).
- `0.5` — default.
- Closer to `1` — almost nothing counts as speech; soft speech is ignored; the agent may never open a turn or transcribe anything.

### `min_speech_ms`

Default `100`.

How long the score must stay **at or above** `threshold` before Syllabix opens
a turn and starts feeding audio to STT. Shorter blips are ignored and never
transcribed.

- Lower it — turns start sooner; brief noises are more likely to open a false turn.
- Raise it — short sounds are ignored; you must speak a bit longer before transcription starts.

### `end_silence_ms`

Default `350`.

How long the score must stay **below** `threshold` after speech before the turn
ends and STT finalizes the utterance.

- Lower it — the turn ends sooner after you stop; the agent may cut you off mid-sentence.
- Raise it — the agent waits longer after pauses; turns feel slower once you finish.

### `preroll_ms`

Default `200`.

Milliseconds of audio kept before VAD detected speech start, prepended to the
STT utterance so the beginning of the first word is not clipped.

Increase this if the agent often skips or gets your first few words wrong.
Recommended to be higher than `min_speech_ms`.

## STT — speech → text

### `provider`

Only `local` (in-process). Required. Other providers are a placeholder for future support and fail at load.

### `model`

Ids and sources: [engines](engines.md#stt). Unknown ids fail at load. First `run` fetches only the id you pick. Measured latency and memory: [workload benchmark](workload_benchmark/workload_benchmark.md#stt).

### `language`

Whisper-supported ISO code (`en`, `fr`, `de`, `ja`, …) or `auto`.

- With a fixed code, STT runs in that language.
- With `auto`, the detected language shows in the TUI and diagnostics sidecar, and the LLM replies in that language via `{language}` in `system_prompt`.

Moonshine models are English-only: `language` must be `en`; `auto` and other codes fail at load. Partials show automatically while VAD owns the turn. There is no separate partials key — Silero `end_silence_ms` remains the one turn-end control. Medium uses a larger INT8 ONNX export (~590 MB vs ~360 MB for small) with the same runtime.

## LLM — text → reply

### `provider`

`local` (default) or `online`.

- `local` — GGUF in-process via llama.cpp.
- `online` — OpenAI-compatible `chat/completions`; audio stays on the machine; only transcript text leaves.

### `model`

**Local** ids and sources: [engines](engines.md#llm). Unknown local ids fail at load.

**Online:** any id the endpoint serves (`gpt-4o-mini`, …).

Measured latency and memory for each local LLM id: [workload benchmark](workload_benchmark/workload_benchmark.md#llm).

### `thinking`

Default `false`. Set `true` only with local `qwen3.5-2b`. Rejected for every other model.

### `system_prompt`

Optional spoken persona for both `local` and `online`. Default: the sample prompt above.

`{language}` is replaced at generate time with the STT language’s English name (`English`, `French`, …). Empty or non-string values fail at load. `init` writes this key so you can edit it.

### `base_url`

Required when `provider: online`. Forbidden when `provider: local`. No default value for `base_url`.

Examples: `https://api.openai.com/v1`, `https://api.groq.com/openai/v1`, `http://127.0.0.1:11434/v1` (Ollama), any vLLM / llama-server URL.

```yaml
pipeline:
  llm:
    provider: online
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1
```

### API key (environment, not yaml)

```bash
SYLLABIX_LLM_API_KEY=sk-… syllabix run
# or: export SYLLABIX_LLM_API_KEY=sk-…
```

Missing/empty with `provider: online` fails before devices or weights load. Keyless loopback still needs a placeholder: `SYLLABIX_LLM_API_KEY=ollama syllabix run`.

### `developer_harness`

Default off. `init` does not write it. Explicit opt-in for the developer tool loop. Valid only with `provider: online` or local `lfm2.5-2.6b`. The ordinary spoken path stays tool-free.

The harness uses a generic shell through the host sandbox: macOS Seatbelt and Linux Bubblewrap, with Landlock fallback. The model may request less filesystem authority per call, but the session ceiling is fixed at process start and cannot be widened in-session. The shell uses a scrubbed environment, bounded output, cancellation, and a workspace-contained working directory. It has no wall-clock deadline; cancellation still stops the active command.

### `developer_permissions`

Optional session ceiling. Valid only when `developer_harness: true`. Unknown keys fail at load.

| Key | Values | Default |
| --- | --- | --- |
| `filesystem` | `read-only` \| `workspace-write` \| `danger-full-access` | `read-only` |
| `network` | `none` \| `allow` | `none` |
| `secrets` | `none` only | `none` |

```yaml
pipeline:
  llm:
    developer_harness: true
    developer_permissions:
      filesystem: workspace-write
      network: none
      secrets: none
```

Omitting `developer_permissions` uses `read-only`, `none`, and `none`.

## TTS — reply → speech

### `provider`

Only `local` (in-process). Required. `online` is a placeholder for future support and fails at load.

### `model`

Ids and sources: [engines](engines.md#tts). Unknown ids fail at load. Non-default ids are fetched on first use. Pocket TTS accepts no user voice data.

Measured latency and memory for each TTS id: [workload benchmark](workload_benchmark/workload_benchmark.md#tts).

### `language`

Optional. Qwen engines use it as the voice language (`en`, `zh`, `de`, `it`, `pt`, `es`, `fr`, `ja`, `ko`, `ru`). Kokoro ignores it (fixed ONNX voice).

### `compute`

Optional and valid only for Qwen. Values: `auto` (default), `cpu`, or `metal`.
On Apple Silicon, `auto` runs after STT and LLM are resident: it attempts a
complete Metal voice-anchor/audio warm-up, destroys that context if reservation
or inference fails, then reloads Qwen on CPU. Other platforms select CPU.
Forced `metal` fails clearly outside macOS. Pocket TTS remains the default and
never pays this probe cost.

```yaml
pipeline:
  tts:
    provider: local
    model: qwen3-0.6
    language: en
    compute: auto
```

The selected `metal` or `cpu` backend is recorded as `tts_backend` in turn
diagnostics and as `backend` in component benchmark JSONL. Qwen TTS stays
`provider: local`; it can pair with `pipeline.llm.provider: online` (audio
never leaves the machine). That combination does not load a local LLM GGUF,
so the Metal TTS probe competes only with resident STT.

## Skills — local instruction packages

Optional top-level block. Valid only when `pipeline.llm.developer_harness` is `true`. `init` omits it. The product-shipped root beside the executable supplies default skills; repository and global roots are extra sources of local instructions.

### `roots`

List of `{ path, source }` entries.

| `source` | `path` |
| --- | --- |
| `repository` | Workspace-relative or absolute |
| `global` | Absolute only |

```yaml
skills:
  roots:
    - path: skills
      source: repository
    - path: /Users/me/.config/syllabix/skills
      source: global
```

Each root contains one directory per skill with a UTF-8 `SKILL.md`. Front matter currently accepts only name, description, and harmless declarative inputs; the Markdown body is shown to the harness as labelled `default` or `custom` reference text. Unknown metadata, invalid skills, and duplicate names are retained only as diagnostics and are never shown to the model. Phase 6 rejects entrypoints and executes no skill code. Skill text never grants filesystem, network, or secret access.

## Auto-timeout — idle mic-mute / exit

Optional top-level block. Omit for defaults. `0` disables that timer.

### `mic_mute_ms`

Default `180000` (3 minutes). After this much idle listen time, capture is muted until you press `m`.

### `exit_ms`

Default `600000` (10 minutes). After this much idle listen time, `run` exits. When both timers are positive, `exit_ms` must be greater than `mic_mute_ms`.

```yaml
auto-timeout:
  mic_mute_ms: 180000
  exit_ms: 600000
```

The idle clock runs while listening, stops on speech start, and restarts after TTS (and after skipped turns). Any keypress resets an armed clock.

## Diagnostics — turn timelines / WAVs

Yaml-only. Default `run` writes nothing.

### `timestamps`

Default `false`. When `true`, writes per-turn timeline sidecars.

### `audio`

Default `false`. When `true`, also writes turn WAVs and implies `timestamps: true`.

### `directory`

Output folder. Default `target/turn-debug`.

```yaml
diagnostics:
  timestamps: true
  audio: true
  directory: target/turn-debug
```

When timestamps or audio is on, each turn writes files under `directory`. What those files contain (timeline events, WAV names) is documented in [diagnostics.md](diagnostics.md).

## Related

| Page | For |
| --- | --- |
| [Docs index](README.md) | User, config, and contributor map |
| [engines.md](engines.md) | Shipped model ids |
| [cli.md](cli.md) | `run`, `--barge-in`, `init` |
| [diagnostics.md](diagnostics.md) | Turn timeline field guide |
| [install.md](install.md) | Checksums, cache, Gatekeeper, SmartScreen, compile |
| [troubleshooting.md](troubleshooting.md) | Mic, echo, keys, first-run fetch |
| [faq.md](faq.md) | Large first run, barge-in default, keys |
| [reference-profiles.md](reference-profiles.md) | How we measure conversation quality |
| [workload_benchmark/workload_benchmark.md](workload_benchmark/workload_benchmark.md) | Per-model STT / LLM / TTS timings |
