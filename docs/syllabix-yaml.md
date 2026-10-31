# `syllabix.yaml`

Optional project file for Syllabix. **`syllabix run` needs no yaml.** Defaults talk with zero config.

| Fact | Detail |
| --- | --- |
| Name | `syllabix.yaml` (only this name is loaded) |
| Where | Current working directory when you `run` |
| Create | `syllabix init [dir]`, or copy [`examples/demo-fully-self-hosted-voice-agents.yaml`](../examples/demo-fully-self-hosted-voice-agents.yaml) |
| Missing file | Quietly uses defaults — not an error |
| Unknown keys | Fail at load |
| Not yaml keys | API keys, barge-in (`run --barge-in`), sample rate / frame size |

`init` writes the defaults so you can edit them. It refuses to overwrite an existing file.

## Sample

```yaml
name: demo-agent                 # agent label
pipeline:
  vad:                           # speech start/end
    provider: silero
    threshold: 0.6
    min_speech_ms: 384
    end_silence_ms: 800
    preroll_ms: 200
  stt:                           # speech → text
    provider: local
    model: whisper-small
    language: en
  llm:                           # text → reply
    provider: local
    model: lfm3-1.7b
    thinking: false
    system_prompt: "You are a smart assistant. This is a spoken conversation. Reply in spoken {language}, the way a person talks: brief, clear, and natural. Do not use markdown, lists, headings, or emoji."
  tts:                           # reply → speech
    provider: local
    model: pocket-tts
    language: en
# optional (init omits these):
# agent_harness:             # agent backend + sandbox shell environment
# auto-timeout:                  # idle mic-mute / exit
# diagnostics:                   # turn timelines / WAVs
# skills:                        # local instruction packages (harness only)
```

First `run` fetches only the selected model ids (~2.5 GB for the default stack). Model catalogue (ids, sources, machine needs): [engines.md](engines.md). The rest of this page walks every key in order.

## Local vs online

Each of STT, LLM, and TTS is `local` (in-process, the default) or `online` (an OpenAI-compatible `base_url`). Default `run` stays fully local. API keys never go in yaml. `base_url` may be a hosted Audio API or [`syllabix serve`](cli.md#serve) on the LAN (`http://127.0.0.1:8091/v1` STT, `:8092` TTS).

| Stage | `local` | `online` |
| --- | --- | --- |
| STT | In-process. Ids: [engines](engines.md#stt). | OpenAI-compatible `audio/transcriptions`. `base_url` ([same rules](#base_url)). Key: `SYLLABIX_STT_API_KEY`. |
| LLM | GGUF in-process (named id or `.gguf` path). | `provider: openai-compatible` — OpenAI-compatible `chat/completions`. Key: `SYLLABIX_LLM_API_KEY` (self-hosted endpoints may use a placeholder; [API key](#api-key-environment-not-yaml)). See [`base_url`](#base_url). |
| TTS | In-process. Ids: [engines](engines.md#tts). | OpenAI-compatible `audio/speech`. `base_url` ([same rules](#base_url)). Key: `SYLLABIX_TTS_API_KEY`. |

## `name`

Display label for the agent. Default: `demo-agent`.

## VAD — speech start / end

Voice activity detection (`pipeline.vad`) decides when the user has started
talking and when the turn is over. `threshold`, `min_speech_ms`, and
`end_silence_ms` are the editable controls; the small margin that separates a
quiet moment inside the turn from a possible end of turn is fixed. Leave these
alone unless you are diagnosing turn-taking.

### `provider`

Only `silero`. Required.

### `threshold`

Default `0.6`. Range `(0, 1]` (`0` fails at load).

Silero scores every audio frame from `0` to `1`. A frame counts as **speech**
when `score >= threshold`.

- Closer to `0` — very sensitive; almost everything counts as speech (fan noise, rustling).
- `0.6` — default.
- Closer to `1` — almost nothing counts as speech; soft speech is ignored; the agent may never open a turn or transcribe anything.

### `min_speech_ms`

Default `384`.

How long the score must stay **at or above** `threshold` before Syllabix opens
a turn and starts feeding audio to STT. Shorter blips are ignored and never
transcribed.

- Lower it — turns start sooner; brief noises are more likely to open a false turn.
- Raise it — short sounds are ignored; you must speak a bit longer before transcription starts.

### `end_silence_ms`

Default `800`.

How long the score must stay **below** `threshold` after speech before the turn
ends and STT finalizes the utterance.

Briefly quiet moments inside a longer utterance — a soft consonant, a pause for
breath — do not count toward this window. Only stretches clearly below
`threshold` count, so a small dip holds the turn open instead of ending it.

- Lower it — the turn ends sooner after you stop; the agent may cut you off mid-sentence.
- Raise it — the agent waits longer after pauses; turns feel slower once you finish.

In a noisy room, raise `end_silence_ms` (or `threshold`) before lowering
`min_speech_ms`. Background noise keeps the score near the threshold, and this
window is the usual control for keeping short noise bursts from ending a turn.

### `preroll_ms`

Default `200`.

Milliseconds of audio kept before VAD detected speech start, prepended to the
STT utterance so the beginning of the first word is not clipped.

Increase this if the agent often skips or gets your first few words wrong.

## STT — speech → text

### `provider`

`local` (default, in-process) or `online`.

- `local` — in-process. Ids: [engines](engines.md#stt).
- `online` — OpenAI-compatible transcriptions endpoint. Posts each finished utterance; no STT weights are fetched. See [Local vs online](#local-vs-online). `base_url` uses the [LLM rules](#base_url).

### `model`

**Local** ids: [engines](engines.md#stt). Unknown local ids fail at load. First `run` fetches only the id you pick.

**Online:** any id the endpoint serves.

Measured latency and memory for local ids: [workload benchmark](workload_benchmark/workload_benchmark.md#stt).

### `language`

Whisper-supported ISO code (`en`, `fr`, `de`, `ja`, …) or `auto`.

- With a fixed code, STT runs in that language.
- With `auto`, the detected language shows in the TUI and diagnostics sidecar, and the LLM replies in that language via `{language}` in `system_prompt`.

Moonshine models are English-only: `language` must be `en`; `auto` and other codes fail at load. Partials show automatically while VAD owns the turn. There is no separate partials key — Silero `end_silence_ms` remains the one turn-end control. Medium uses a larger INT8 ONNX export (~590 MB vs ~360 MB for small) with the same runtime.

### `compute`

Optional and valid only for Qwen3-ASR (`qwen3-asr-0.6`). Values: `auto` (default), `cpu`, `metal`, or `vulkan`.
Same placement rules as TTS `pipeline.tts.compute`: Apple Silicon `auto` tries Metal then CPU; Linux builds with `SYLLABIX_GGML_VULKAN=1` try Vulkan when a device is present, then CPU. Forced `metal` / `vulkan` fail clearly when unavailable. Whisper and Moonshine reject this key.

```yaml
pipeline:
  stt:
    provider: local
    model: qwen3-asr-0.6
    language: en
    compute: auto
```

The selected backend is recorded as `backend` in component benchmark JSONL.

## LLM — text → reply

### `provider`

With `agent_harness.agent: syllabix` or `none` (default), only `local` and `openai-compatible` are valid.

- `local` — GGUF in-process via llama.cpp. `model` is a catalogue id or a path to a `.gguf` file.
- `openai-compatible` — any OpenAI-style `chat/completions` endpoint (`base_url` + `model`). Audio stays on the machine; only transcript text leaves. The old value `online` was renamed to `openai-compatible`.
- Any other value (`openai`, `anthropic`, `ollama`, `openrouter`, …) is a [Goose](engines.md) provider name and requires `agent_harness.agent: goose`, unless you use `openai-compatible` with that vendor's OpenAI-compatible endpoint instead.

### `model`

**Local** ids and sources: [engines](engines.md#llm), or a `.gguf` path. Unknown local ids fail at load.

**OpenAI-compatible (syllabix/none) or Goose:** any id the endpoint or provider serves.

Measured latency and memory for each local LLM id: [workload benchmark](workload_benchmark/workload_benchmark.md#llm).

### `thinking`

Default `false`. Set `true` only with local `qwen3.5-2b`. Rejected for every other model.

### `system_prompt`

Optional spoken persona for `local` and `openai-compatible`. Default: the sample prompt above.

`{language}` is replaced at generate time with the STT language’s English name (`English`, `French`, …). Empty or non-string values fail at load. `init` writes this key so you can edit it.

### `base_url`

Required when `provider: openai-compatible`. Forbidden when `provider: local`. Optional for Goose `ollama` (`OLLAMA_HOST`). No default value for `base_url`.

Examples: `https://api.openai.com/v1`, `https://api.groq.com/openai/v1`, `http://127.0.0.1:11434/v1` (Ollama via `openai-compatible`), any vLLM / llama-server URL.

```yaml
pipeline:
  llm:
    provider: openai-compatible
    model: gpt-4o-mini
    base_url: https://api.openai.com/v1
```

Anthropic through the Syllabix reply path (OpenAI-compatible API; key from the environment):

```yaml
pipeline:
  llm:
    provider: openai-compatible
    model: claude-sonnet-4-5
    base_url: https://api.anthropic.com/v1/
```

Use `SYLLABIX_LLM_API_KEY` with your Anthropic API key (never put keys in yaml).

### API key (environment, not yaml)

```bash
SYLLABIX_LLM_API_KEY=sk-… syllabix run
# or: export SYLLABIX_LLM_API_KEY=sk-…
```

Missing/empty with `agent_harness.agent: none` or `syllabix` and `provider: openai-compatible` fails before devices or weights load. With `agent_harness.agent: goose`, Syllabix maps `SYLLABIX_LLM_API_KEY` to the Goose provider's expected variable when applicable. Keyless loopback still needs a placeholder: `SYLLABIX_LLM_API_KEY=ollama syllabix run`.

### `developer_permissions`

Optional session ceiling under `pipeline.llm`. Valid only when `agent_harness.agent: syllabix`. Unknown keys fail at load.

| Key | Values | Default |
| --- | --- | --- |
| `filesystem` | `read-only` \| `workspace-write` \| `danger-full-access` | `read-only` |
| `network` | `none` \| `allow` | `none` |
| `secrets` | `none` only | `none` |

```yaml
agent_harness:
  agent: syllabix
pipeline:
  llm:
    developer_permissions:
      filesystem: workspace-write
      network: none
      secrets: none
```

Omitting `developer_permissions` uses `read-only`, `none`, and `none`.

## Agent backend and shell environment (`agent_harness`)

Optional top-level `agent_harness:` block. `init` omits it. Unknown keys fail at load.

### `agent`

Default `none` (omitted). Values: `none` \| `syllabix` \| `goose`.

- `none` — plain voice chat with no sandboxed tools (the default `init` path).
- `syllabix` — built-in agent harness with sandboxed shell tools. Works with `provider: local` (any catalogue id or `.gguf` path whose GGUF embeds a Jinja `chat_template` that supports tools—LFM, Qwen3.5, Llama 3.x, Gemma 4 / Nemotron Bolt, and similar) or `openai-compatible`. Templates without tool support fail closed with a spoken error at harness start; you do not need a separate `llama-server` for local tool turns. The old yaml value `builtin` was renamed to `syllabix`; the removed `pipeline.llm.developer_harness` flag is replaced by `agent_harness.agent: syllabix`.
- `goose` — Goose runs the reply backend. Uses `pipeline.llm` as the single place to pick provider and model. Syllabix writes them into `{SYLLABIX_CACHE_DIR}/goose/config/config.yaml` (merging with keys you added by hand) before starting Goose. Selecting `goose` downloads a pinned Goose CLI into the model cache and checks its SHA-256 before unpacking. Goose can search the web and fetch pages. Details: [engines](engines.md).

With `agent: syllabix`, the harness uses a generic shell through the host sandbox: macOS Seatbelt and Linux Bubblewrap, with Landlock fallback. The model may request less filesystem authority per call, but the session ceiling is fixed at process start and cannot be widened in-session. Bounded output, cancellation, and a workspace-contained working directory apply to every command. It has no wall-clock deadline; cancellation still stops the active command.

With barge-in on (`run --barge-in`), speaking while Goose is answering cancels that answer.

### `inherit_current_session`

Default `true`. When `true`, each command the agent runs (syllabix or Goose) starts with the environment of the shell that launched Syllabix, like a forked shell — the agent can read every variable you have exported there. Set `inherit_current_session: false` for a clean session: the environment is cleared and only safe defaults (for example `PATH` and git non-interactive settings) are set. File and network sandbox rules are unchanged either way. Secret values are never written to diagnostics or trajectory logs.

```yaml
agent_harness:
  agent: syllabix
  inherit_current_session: false
```

## TTS — reply → speech

### `provider`

`local` (default, in-process) or `online`.

- `local` — in-process. Ids: [engines](engines.md#tts).
- `online` — OpenAI-compatible speech endpoint. Posts each finished sentence (`stream_format=audio`); no TTS weights are fetched. See [Local vs online](#local-vs-online). `base_url` uses the [LLM rules](#base_url).

### `model`

**Local** ids: [engines](engines.md#tts). Unknown local ids fail at load. Non-default ids are fetched on first use. Pocket TTS accepts no user voice data.

**Online:** any id the endpoint serves.

Measured latency and memory for each TTS id: [workload benchmark](workload_benchmark/workload_benchmark.md#tts).

### `language`

Optional. Qwen engines use it as the voice language (`en`, `zh`, `de`, `it`, `pt`, `es`, `fr`, `ja`, `ko`, `ru`). Kokoro ignores it (fixed ONNX voice).

### `compute`

Optional and valid only for Qwen. Values: `auto` (default), `cpu`, `metal`, or `vulkan`.
On Apple Silicon, `auto` runs after STT and LLM are resident: it attempts a
complete Metal voice-anchor/audio warm-up, destroys that context if reservation
or inference fails, then reloads Qwen on CPU. On a Linux build compiled with
`SYLLABIX_GGML_VULKAN=1`, `auto` tries Vulkan only when a Vulkan device is
present, the same way, and falls back to CPU. Other platforms select CPU.
Forced `metal` fails clearly outside macOS. Forced `vulkan` fails when Vulkan
was not compiled in (including Darwin) or no Vulkan device is found.
Pocket TTS remains the default and never pays this probe cost.

```yaml
pipeline:
  tts:
    provider: local
    model: qwen3-0.6
    language: en
    compute: auto
```

The selected `metal`, `vulkan`, or `cpu` backend is recorded as `tts_backend` in turn
diagnostics and as `backend` in component benchmark JSONL. Qwen TTS stays
`provider: local`; it can pair with `pipeline.llm.provider: openai-compatible` (audio
never leaves the machine). That combination does not load a local LLM GGUF,
so the Metal TTS probe competes with resident STT (including a GPU-backed Qwen ASR).

## Skills — local instruction packages

Optional top-level block. Valid only when `agent_harness.agent` is `syllabix` or `goose`. `init` omits it. The product-shipped root beside the executable supplies default skills; repository and global roots are extra sources of local instructions.

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

### `trajectory`

Default `false`. When `true`, also writes each turn's agent trajectory
(`user`, tool calls and results, assistant reply, outcome) as
`trajectory.jsonl` next to `turn.json`, with secrets redacted. Implies
`timestamps: true`. Written the same for `agent_harness.agent: syllabix` and
`agent_harness.agent: goose`.

### `directory`

Output folder. Default `target/turn-debug`.

```yaml
diagnostics:
  timestamps: true
  audio: true
  trajectory: true
  directory: target/turn-debug
```

When any of these is on, each turn writes files under `directory`. What those files contain (timeline events, WAV names, trajectory fields) is documented in [diagnostics.md](diagnostics.md).

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
