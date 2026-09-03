# Configuration

`syllabix run` works with **no yaml**. Use `syllabix init [dir]` (or copy [`examples/demo-agent.yaml`](../examples/demo-agent.yaml)) when you want a project folder.

Unknown keys fail at load.

## Default stack (zero-config)

| Layer | Default |
| --- | --- |
| VAD | Silero (threshold 0.5, min speech 100 ms, end silence 350 ms, Whisper preroll 200 ms) |
| STT | whisper.cpp `small`, language `en` |
| LLM | llama.cpp, Llama 3.2 1B, thinking off |
| TTS | Kokoro |
| Echo | Full-duplex AEC3, on |

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
    provider: whisper.cpp
    model: small
    language: en
  llm:
    provider: local
    model: llama-3.2-1b
    thinking: false
    # optional; omit for the launch default. `{language}` → STT language name.
    system_prompt: "You are a smart assistant. This is a spoken conversation. Reply in spoken {language}, the way a person talks: brief, clear, and natural. Do not use markdown, lists, headings, or emoji."
  tts:
    provider: local
    model: kokoro
    language: en            # optional; Qwen3-TTS voice language
```

Omit any VAD tunable to keep the launch default. First `run` fetches only the selected model ids.

## STT

`pipeline.stt.provider` is `whisper.cpp`.

| `model` | Notes |
| --- | --- |
| `small` | Default |
| `medium` | |
| `large-v3-turbo` | |
| `medium-q5_0` | Published quantization |
| `large-v3-turbo-q5_0` | Published quantization |

`language` is a whisper-supported ISO code (`en`, `fr`, `de`, `ja`, …) or `auto`. With `auto`, the detected language shows in the TUI and diagnostics sidecar, and the agent replies in that language.

## LLM

`pipeline.llm.provider` is `local` (default) or `online`.

### Local

| `model` | Notes |
| --- | --- |
| `llama-3.2-1b` | Default; first-run cache |
| `qwen3.5-0.8b` | Fetched when selected |
| `qwen3.5-2b` | Fetched when selected |
| `lfm2.5-2.6b` | LiquidAI LFM2.5 QAD Q4_0; fetched when selected |

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
| `kokoro` | Default (~310 MB) |
| `pocket-tts` | English Pocket TTS ONNX graph set + fixed Alba voice (~150 MB); fetched when selected |
| `qwen3-0.6` | Qwen3-TTS 0.6B + speech tokenizer; fetched when selected |
| `qwen3-1.7` | Qwen3-TTS 1.7B + speech tokenizer; fetched when selected |

Pocket TTS uses the pinned English SentencePiece tokenizer and the shipped fixed Alba voice; it accepts no user voice data or voice-registration input. Qwen engines read numbers and currency as words (`100` → “one hundred”), speak ten languages via optional `tts.language` (`en`, `zh`, `de`, `it`, `pt`, `es`, `fr`, `ja`, `ko`, `ru`), and pin one voice across sentences. Kokoro stays the zero-config default.

## Diagnostics

Yaml-only; default `run` writes nothing.

```yaml
diagnostics:
  timestamps: true
  audio: true                 # implies timestamps; writes WAVs
  directory: target/turn-debug
```

Field guide: [diagnostics.md](diagnostics.md).
