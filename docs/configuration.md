# Configuration

`syllabix run` works with **no yaml**. Use `syllabix init [dir]` (or copy [`examples/demo-agent.yaml`](../examples/demo-agent.yaml)) when you want a project folder.

Unknown keys fail at load.

## How yaml relates to the binary

The Release artifact is one native executable. It does not contain your project
yaml. On every `run`, the binary looks in the **current working directory** for
`syllabix.yaml`:

```text
syllabix (binary)
  └─ run
       ├─ cwd/syllabix.yaml present? → parse → AgentConfig
       └─ missing?                  → AgentConfig::v0() (hardcoded)
            └─ load models / open mic / talk
```

Weights still live in the model cache (fetched by selected ids). Yaml only
chooses stack options for that process. Edit the file, then **rerun**
`syllabix run` — a running session does not reload config.

## TUI configure (`c`)

While `syllabix run` is in the transcript view, press **`c`** (not Ctrl+C):

1. The conversation stops and raw mode leaves the terminal.
2. If `cwd/syllabix.yaml` is missing, Syllabix writes a **commented** template
   (same file as `syllabix init`) listing model menus and optional blocks.
3. The file opens in `$VISUAL`, then `$EDITOR`, then a platform default
   (`nano` on macOS/Linux, `notepad` on Windows).
4. Save, quit the editor, then rerun `syllabix run` to load the new stack.

`init` and `c` share the same commented template; neither overwrites an existing
yaml.

## Default stack (zero-config)

| Layer | Default |
| --- | --- |
| VAD | Silero (threshold 0.5, min speech 100 ms, end silence 350 ms, Whisper preroll 200 ms) |
| STT | whisper.cpp `small`, language `en` |
| LLM | llama.cpp, LFM2.5-2.6B, thinking off (LFM always thinks internally; think tags are stripped before TTS) |
| TTS | Kokoro |
| Echo | Full-duplex AEC3, on |

Launch contract: the default is the best M4 16 GB combination for minimal
delight (whisper `small` + `lfm2.5-2.6b` + Pocket TTS; evidence in
`docs/eval/runs/` and `docs/reference-profiles.md`).

`run --barge-in` is a CLI flag, not a yaml key. Off by default.

## Shape

```yaml
name: demo-agent
pipeline:
  vad:
    provider: silero
    threshold: 0.5          # optional; (0, 1]
    min_speech_ms: 100      # optional
    end_silence_ms: 350     # optional
    preroll_ms: 200         # optional Whisper preroll
  stt:
    provider: local
    model: whisper-small
    language: en
  llm:
    provider: local
    model: lfm2.5-2.6b
    thinking: false
    # optional; omit for the launch default. `{language}` → STT language name.
    system_prompt: "You are a smart assistant. This is a spoken conversation. Reply in spoken {language}, the way a person talks: brief, clear, and natural. Do not use markdown, lists, headings, or emoji."
  tts:
    provider: local
    model: pocket-tts
    language: en            # optional; Qwen3-TTS voice language
```

Omit any VAD tunable to keep the launch default. First `run` fetches only the selected model ids.

## STT

`pipeline.stt.provider` is `local`. Its `model` selects the in-process STT engine.

| `model` | Notes |
| --- | --- |
| `whisper-small` | Default |
| `whisper-medium` | |
| `whisper-large-v3-turbo` | |
| `whisper-medium-q5_0` | Published quantization |
| `whisper-large-v3-turbo-q5_0` | Published quantization |
| `moonshine-streaming-small` | English-only streaming ASR with automatic partials |
| `moonshine-streaming-medium` | Larger English-only streaming ASR with automatic partials |

`language` is a whisper-supported ISO code (`en`, `fr`, `de`, `ja`, …) or `auto`. With `auto`, the detected language shows in the TUI and diagnostics sidecar, and the agent replies in that language.

### Moonshine streaming (small / medium)

```yaml
pipeline:
  stt:
    provider: local
    model: moonshine-streaming-small   # or moonshine-streaming-medium
    language: en
```

Both Moonshine sizes are English-only: `en` is required; `auto` and other
language codes fail at config load. They display partial transcript text
automatically while VAD owns an active turn. There is no partials setting and
no second endpointing timer: Silero's `end_silence_ms` remains the one turn-end
control, after which Moonshine finalizes the same utterance. Medium uses a
larger INT8 ONNX export (~590 MB vs ~360 MB for small) with the same runtime.

## LLM

`pipeline.llm.provider` is `local` (default) or `online`.

### Local

| `model` | Notes |
| --- | --- |
| `lfm2.5-2.6b` | Default; first-run cache (LiquidAI LFM2.5 QAD Q4_0) |
| `llama-3.2-1b` | Fetched when selected |
| `qwen3.5-0.8b` | Fetched when selected |
| `qwen3.5-2b` | Fetched when selected |

`thinking: true` enables chain-of-thought on `qwen3.5-2b` only (rejected for every other model). Thinking text is never spoken and is hidden in the TUI. Unknown local ids fail at load.

Optional `system_prompt` sets the spoken persona for both `local` and `online`. `{language}` is replaced at generate time with the STT language’s English name (`English`, `French`, …). Omit the key (or run with no yaml) for the launch default above. `syllabix init` writes it so you can edit it. Empty or non-string values fail at load.

### Online (BYO key)

Audio never leaves the machine. Only transcript text reaches the endpoint you set.

```yaml
pipeline:
  llm:
    provider: online
    model: gpt-4o-mini                 # any id the endpoint serves
    base_url: https://api.openai.com/v1  # required for online; forbidden for local
```

`base_url` examples: `https://api.openai.com/v1`, `https://api.groq.com/openai/v1`, `http://127.0.0.1:11434/v1` (Ollama), or any vLLM / llama-server URL. Nothing is defaulted: `online` without `base_url` fails at load; `local` rejects the field.

Set the key in the environment:

```bash
SYLLABIX_LLM_API_KEY=sk-… syllabix run
# or: export SYLLABIX_LLM_API_KEY=sk-…
```

Missing or empty key with `provider: online` fails at `run` start before devices or weights load. Keyless loopback (Ollama, llama-server, vLLM) still needs a placeholder: `SYLLABIX_LLM_API_KEY=ollama syllabix run`.

## TTS

`pipeline.tts.provider` is `local` (in-process) or `online` (reserved; fails fast).

| `model` (under `local`) | Notes |
| --- | --- |
| `pocket-tts` | Default English Pocket TTS ONNX graph set + fixed Alba voice (~150 MB) |
| `kokoro` | Kokoro ONNX + default voice (~310 MB); fetched when selected |
| `qwen3-0.6` | Qwen3-TTS 0.6B + speech tokenizer; fetched when selected |
| `qwen3-1.7` | Qwen3-TTS 1.7B + speech tokenizer; fetched when selected |

Pocket TTS uses the pinned English SentencePiece tokenizer and the shipped fixed Alba voice; it accepts no user voice data or voice-registration input. Qwen engines read numbers and currency as words (`100` → “one hundred”), speak ten languages via optional `tts.language` (`en`, `zh`, `de`, `it`, `pt`, `es`, `fr`, `ja`, `ko`, `ru`), and pin one voice across sentences. Set `model: kokoro` to use Kokoro instead.

## Auto-timeout

Optional top-level block. When omitted, both timers use the launch defaults.
`0` disables that timer.

```yaml
auto-timeout:
  mic_mute_ms: 180000   # 3 minutes; 0 disables auto mic-mute
  exit_ms: 600000       # 10 minutes; 0 disables idle exit
```

The idle clock runs while the agent is listening. It stops on user speech start and restarts after TTS playback finishes (and after skipped turns that never speak). Any keypress resets an armed clock. Auto mic-mute drops capture frames until you press `u` to listen again. When both timers are positive, `exit_ms` must be greater than `mic_mute_ms`.

## Diagnostics

Yaml-only; default `run` writes nothing.

```yaml
diagnostics:
  timestamps: true
  audio: true                 # implies timestamps; writes WAVs
  directory: target/turn-debug
```

Field guide: [diagnostics.md](diagnostics.md).
