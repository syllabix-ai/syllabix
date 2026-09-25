# Remote STT and TTS serve

Proposal: keep `syllabix run` as the conversation runtime on every
endpoint, and add a **serve** command that is a model daemon (Ollama-shaped)
for speech-to-text and text-to-speech. LLM-as-a-service already exists
(`pipeline.llm.provider: online` against Ollama, vLLM, llama-server, or any
OpenAI-compatible endpoint). ASR and TTS do not have that ecosystem yet, so
Syllabix ships those workers.

This is usability for a fleet of endpoints (desk, laptop, speaker, later
other UIs): one weight cache, not ten copies of Whisper, Moonshine, or
Pocket/Qwen. It is not a new conversation product. Capture, AEC, VAD,
barge-in, playback, and session history stay on the device that has the
human.

Status: design only. Default `run` is unchanged until the PR list lands.

## Topology

```text
  endpoint 1 (mic, AEC, VAD, barge-in, speakers, TUI or other UI)
  endpoint 2
  …
  endpoint N
           \  transcript text  →  LLM serve  (already: provider online)
           \  utterance PCM    →  syllabix serve stt
           /  token/text stream → syllabix serve tts  → PCM back
```

Each endpoint is a thin Syllabix runtime. One (or a few) machines hold
models. Users should not install Whisper, Moonshine, Qwen-ASR, or
Pocket/Qwen-TTS on every desk.

Forwarding ALSA/Pulse to a GPU box and running `syllabix run` there is out
of scope. Serve **inference**, not the conversation. AEC and barge-in need
capture and playback on the same clock as the speakers.

## What stays on every endpoint vs what is served

| Stay on every endpoint | Serve once (sub-command / worker) |
| --- | --- |
| Capture / playback | STT weights |
| AEC3 | TTS weights |
| VAD, turn start/end, barge-in cancel | LLM (already external) |
| Session / history / yaml persona | |
| Idle mute, local TUI or other UI | |

Do not serve VAD or AEC. Silero on-device is cheap. Whisper, Moonshine,
Qwen-ASR, and Qwen-TTS are the RAM/VRAM that should not be replicated.
`serve stt --model` is any yaml STT id (`build_stt`), including
`moonshine-streaming-*`.

Barge-in is a **local** cancel of a **remote** job.

## Difficulty

The pipeline, `Stt` / `Tts` traits, YAML hooks, HTTP client (`ureq` +
threads in `openai.rs`), WAV helpers, and isolated STT/TTS loaders
(`build_stt` / `build_tts`, bench workers) already exist. There is **no
HTTP server and no async runtime** in the repo.

| Slice | Difficulty | Why |
| --- | --- | --- |
| Client: remote finalize STT | Small | Whisper and Qwen3-ASR are `transcribe` on one utterance. `write_wav` exists. `openai.rs` already has cancel, timeouts, warm-up, `validate_base_url`. **No pipeline-loop change** if `LiveStt` grows a remote arm. |
| Server: `serve stt` (finalize) | Moderate | Wrap `build_stt` (any finalize id). New work is bind, HTTP, auth, single-flight queue + cancel. First HTTP stack in the workspace. |
| Moonshine over the same worker | Moderate–larger | Moonshine is a **menu id**, not a side stack. The `Stt` trait already has `start_turn` / `push_frame` / `transcribe` / `cancel_turn` and `supports_partials`. The worker holds one `MoonshineStt`; the wire is a turn session (same idea as TTS generations), not a WAV POST. Partials are why you serve it instead of collapsing it to Whisper. |
| Client: remote TTS, full utterance | Small–moderate | Buffer tokens until `is_last`, POST text, play WAV. Fits `Tts` but waits for the whole LLM turn before first audio. |
| Client + server: token-stream TTS | Larger | Pipeline already calls `synthesize_chunk` per token; Pocket/Qwen sentence state lives in the engine. That state must sit on the **server** for one generation, with streaming audio back and barge-in cancel. |
| Multi-client fairness beyond a mutex | Later | One in-flight decode plus a short queue is enough for desks that rarely overlap. |

If a change edits turn-taking in the `run` loop, the split is wrong.

`TtsProvider::Online` already exists and fails at load (“not supported
yet”). STT has only `local` today.

## Non-goals (this series)

Moonshine is **in** the series (PR 3 below). It is not a v1 cut.

- A CUDA-specific SKU (worker uses whatever `build_stt` / `build_tts` already compile)
- Public internet bind as the default
- Promising full OpenAI **Realtime** (WebRTC, `session.update`) or every
  Audio field (`translations`, diarization, `srt`/`vtt`)
- Moving AEC, VAD, or `run` onto the server
- WebRTC / SIP / SFU

## CLI

```text
syllabix serve stt  [--bind 127.0.0.1:8091] [--model whisper-small]
syllabix serve stt  --model moonshine-streaming-small
syllabix serve tts  [--bind 127.0.0.1:8092] [--model pocket-tts]
```

- Default bind **localhost**.
- Auth: `SYLLABIX_SERVE_TOKEN` in a header. Required unless bind is loopback
  and the token is unset (dev only).
- Model ids are the existing yaml menu. Load weights once at start via
  `build_stt` / `build_tts`.
- Not a flag on `run`. Hidden `bench-*-worker` is the load template, not
  the UX.

## Yaml

Mirror the LLM `online` shape. Pick one provider name (`online`) and keep
it consistent with `pipeline.llm`.

```yaml
pipeline:
  stt:
    provider: online
    model: whisper-small      # or moonshine-streaming-small, qwen3-asr-0.6, … — must match the worker
    base_url: http://127.0.0.1:8091/v1
  llm:
    provider: online
    model: lfm2.5-2.6b
    base_url: http://127.0.0.1:11434/v1
  tts:
    provider: online
    model: pocket-tts
    base_url: http://127.0.0.1:8092/v1
```

- `provider: local` remains the default `run` path.
- `base_url` required for online, forbidden for local — copy
  `resolve_llm_base_url`.
- Tokens from the environment, never yaml:
  `SYLLABIX_STT_API_KEY` / `SYLLABIX_TTS_API_KEY`, or one
  `SYLLABIX_INFERENCE_API_KEY`.
- When STT/TTS are online, **do not fetch those weights on the endpoint**
  (same as cloud `build_llm`).

`GET /health` returns model id and busy/idle. Client warm-up matches the
cloud LLM path.

## Protocol

Stay in the existing thread model. Add a small blocking HTTP server (for
example `tiny_http`, or a short `TcpListener` parser). Do not pull
Axum/Hyper unless the process model is being redesigned.

### Wire format: OpenAI Audio where it exists

Chat already uses OpenAI-compatible `POST /v1/chat/completions`. Audio has
a **smaller** clone-set. Use it for the shapes everyone already speaks;
do not invent `/v1/stt/transcribe` for those.

**De facto standard (cloned widely):** OpenAI Audio REST under `/v1`.

| Job | Method | Who clones it | Compatible models |
| --- | --- | --- | --- |
| File / utterance STT | `POST /v1/audio/transcriptions` (multipart `file`, `model`, `language`, `response_format`) | Groq, Azure Whisper-style, whisper.cpp `whisper-server`, LocalAI, many gateways. Response `{ "text": "..." }` (`json`) or `verbose_json` (`text`, `language`, `duration`, `segments`). | OpenAI: `whisper-1`, `gpt-transcribe`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`. Groq: `whisper-large-v3`, `whisper-large-v3-turbo`. whisper.cpp: request `whisper-1` (often ignored; weights chosen at process start). Syllabix worker: yaml ids `whisper-small`, `whisper-medium`, `whisper-large-v3-turbo`, `whisper-medium-q5_0`, `whisper-large-v3-turbo-q5_0`, `qwen3-asr-0.6`. |
| Full-text TTS | `POST /v1/audio/speech` (JSON `model`, `input`, `voice`, `response_format`) | Groq PlayAI, LocalAI, Kokoro-FastAPI, openedai-speech. Body is audio bytes (`mp3` default on OpenAI; we default `pcm` or `wav` for the voice loop). Optional `stream_format`: `audio` (chunked bytes) or `sse`. | OpenAI: `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`. Groq: `playai-tts`, `playai-tts-arabic`. Kokoro-FastAPI: `kokoro` plus a voice name. Syllabix worker: yaml ids `pocket-tts`, `kokoro`, `qwen3-0.6`, `qwen3-1.7`. `voice` is accepted for SDK compatibility; Pocket’s fixed voice may ignore it. |

Auth is `Authorization: Bearer …`, same as LLM online. `base_url` is the
`/v1` root, same as `pipeline.llm.base_url`.

**Not a clone-set** (do not pretend these are OpenAI-compatible):

| Job | What majors actually use | Compatible models | Syllabix |
| --- | --- | --- | --- |
| Live mic STT with partials | OpenAI **Realtime** (WebSocket/WebRTC, `input_audio_buffer.append`). Deepgram live WS, AssemblyAI streaming — each proprietary. Groq STT is still the **file** endpoint. | OpenAI: `gpt-live-transcribe`. Deepgram: `nova-3` / `nova-2`. AssemblyAI: streaming model ids on their WS. No Groq live-STT model on `/audio/transcriptions`. | Yaml `moonshine-streaming-small`, `moonshine-streaming-medium` via `Stt::push_frame`. Full Realtime is a product fork. Expose a **small** turn session on the same worker (below). Optional later: emit OpenAI-shaped `transcript.text.delta` events on that session without the Realtime session object. |
| TTS fed by LLM token chunks | OpenAI speech takes one `input` string; streaming is **output audio**, not input tokens. ElevenLabs / Cartesia have their own input-stream websockets. | No OpenAI `/audio/speech` model accepts a token stream. ElevenLabs: e.g. `eleven_multilingual_v2`. Cartesia: Sonic family. | Same yaml TTS ids (`pocket-tts`, `kokoro`, `qwen3-*`) on a generation session that matches `Tts::synthesize_chunk`. Sentence-buffer + repeated `/v1/audio/speech` is the compatible fallback (extra RTT). |

**Decision:** `serve` **is** OpenAI Audio for finalize STT and full-text
TTS so curl, the OpenAI Python client, and other tools work. Moonshine
partials and token-chunk TTS use a thin Syllabix session beside that,
not a fake `/audio/*` that no client expects.

VAD stays on the endpoint for every path. The worker never decides turn
start/end.

### STT — two shapes, one worker

`serve stt` loads one yaml model via `build_stt`.

**Finalize** (Whisper, Qwen3-ASR): OpenAI transcriptions after VAD closes
the utterance.

```http
POST {base_url}/audio/transcriptions
Authorization: Bearer …
Content-Type: multipart/form-data

file=@utterance.wav
model=whisper-small
language=en
response_format=verbose_json
```

- `file`: WAV (`write_wav` already in-tree). `model` must match the
  worker (or be ignored like whisper.cpp, with a warning).
- Success `verbose_json`: `{ "text", "language", "duration"? }`. `json`
  is `{ "text" }` only; the client then keeps yaml language.
- Cancel: drop the client read; the server watches disconnect and fires
  `Cancel`.
- Ignore or 400 unused OpenAI fields we will not honor in v1
  (`timestamp_granularities`, `diarized_json`, `/audio/translations`).

Client `RemoteStt::transcribe`: concat `utterance.pcm()`, `write_wav`,
POST, map to `Transcript`. Check `cancel` like `openai.rs`. The same
client can point `base_url` at Groq or whisper-server for experiments.

**Streaming** (Moonshine: `supports_partials`): not OpenAI
`/audio/transcriptions` (that API has no `push_frame`). One turn session:

- `POST /v1/stt/turns` → `{ turn_id }` maps to `start_turn`
- `POST /v1/stt/turns/:id/frames` — one post-AEC `AudioFrame` (or a
  short batch) maps to `push_frame`; response may include a partial
  string
- `POST /v1/stt/turns/:id/finalize` maps to `transcribe`
- `POST /v1/stt/turns/:id/cancel` maps to `cancel_turn`

`RemoteStt` implements the full `Stt` trait. When `supports_partials`
is true, the pipeline already calls `push_frame`; the remote adapter
must not collapse that to a late transcriptions POST or live partials
disappear.

**Concurrency:** one in-flight turn or finalize (`Mutex`). Extra
requests wait on a short queue (for example 8) then `503`. A Moonshine
session holds that slot until finalize or cancel.

### TTS

**OpenAI speech (compatible, full `input`):**

```http
POST {base_url}/audio/speech
Content-Type: application/json

{ "model": "pocket-tts", "input": "Hello.", "voice": "alloy", "response_format": "pcm" }
```

`voice` is accepted for client compatibility; Pocket’s fixed voice may
ignore it. Response is audio bytes. `stream_format=audio` may chunk
PCM as it is produced.

This is the curl/SDK path and the sentence-buffer fallback: the endpoint
`RemoteTts` can flush on sentence/`is_last` and call this once (honest
extra latency).

**Token-chunk session (matches `Tts::synthesize_chunk`, not in OpenAI
Audio):**

- `POST /v1/tts/generations` → `{ generation_id }`
- `POST /v1/tts/generations/:id/chunks` with `TokenChunk` JSON (`index`,
  `text`, `is_last`)
- Chunked or SSE body of PCM frames (`SynthesizedAudio` metadata + samples)
- `POST .../cancel` maps to existing `Cancel`

The server holds one `LiveTts` and one active generation (or a small map
by id). That reuses `synthesize_chunk_into` with no engine fork.

Prefer the generation session for `run --barge-in` and time-to-first-audio.
Keep `/v1/audio/speech` implemented on the same process so the worker is
still an OpenAI TTS endpoint.

LAN/Tailscale can send 16 kHz mono PCM. A WAN codec (Opus) is later.
`response_format=opus` can wait.

Do not implement OpenAI Realtime (WebRTC SFU, `session.update`, audio
output from an agent) in this series.

## Code layout

| Piece | Where |
| --- | --- |
| Yaml `SttProvider::Online`, `stt_base_url`; TTS `Online` actually builds | `config.rs`, `defaults.rs` |
| `RemoteStt` / `RemoteTts` | New modules next to `openai.rs` |
| `LiveStt::Remote` / `LiveTts::Remote` | `real.rs` `build_*` — skip the model cache |
| HTTP server + job mutex | `crates/syllabix/src/serve.rs` (CLI crate, like `bench.rs`) |
| Shared request types | `syllabix-core` so client and server cannot drift |

Loopback tests (required): serve on `127.0.0.1` — finalize PCM fixture
and a Moonshine turn (`push_frame` partials + finalize). No microphone.

**Copy:** `openai.rs` timeouts/cancel/warm-up; `validate_base_url`;
`build_stt` / `build_tts`; `Cancel`; WAV; clap subcommands.

**Invent:** listen socket, PCM serialization, job mutex.

## PR list

Squash to one commit per PR. Do not combine the HTTP server, STT client,
and streaming TTS in one change. STT alone is a complete fleet feature.

| PR | Title (clean, no prefix) | Scope | Done when |
| --- | --- | --- | --- |
| 1 | Accept online STT yaml like LLM | Config only: `pipeline.stt.provider: online` + `base_url`; reject missing URL; runtime still fails clearly (same posture as TTS online today). Tests in `config.rs`. | Yaml round-trips; local still default; no network. |
| 2 | Serve finalize STT and remote transcribe client | `syllabix serve stt` for Whisper / Qwen3-ASR; OpenAI `POST /v1/audio/transcriptions`; `RemoteStt::transcribe`; skip cache; loopback fixture; bind localhost; token rules. | Yaml online STT works; `curl` + OpenAI-shaped multipart against the worker returns `{ "text" }`. |
| 3 | Serve Moonshine turns and remote partials | Same `serve stt` with `--model moonshine-streaming-*`; Syllabix turn session (not Realtime); `RemoteStt` implements `start_turn` / `push_frame` / `cancel_turn`; loopback partials. | Yaml `moonshine-streaming-small` + `provider: online` shows live partials; barge-in cancel drops the turn. |
| 4 | Serve TTS OpenAI speech and generation client | `POST /v1/audio/speech` on `serve tts`; generation session for `synthesize_chunk`; `RemoteTts` implements `Tts`; barge-in cancel. | OpenAI client can fetch a WAV; yaml online TTS + local STT works; first audio can start before the LLM finishes via the session. |
| 5 | Document serve for a fleet of endpoints | User page: bind, token, tunnel/Tailscale, “weights on the worker,” Moonshine vs finalize STT. Default `run` story unchanged. CLI + engines + yaml cross-links. Architecture “what leaves the machine” updated for online STT/TTS. | Docs match shipped commands. |

Out of this series: queue metrics, multi-model workers (two engines in
one process), GPU backend matrix, Windows-as-worker.

## Suggested order of work inside PRs 2–4

1. Blocking HTTP listen + `/health` with no models (unit-testable).
2. Load one Whisper via `build_stt`; OpenAI `transcriptions` on a fixture WAV.
3. Wire yaml `online` to `RemoteStt` (finalize).
4. Moonshine turn session on the same server (`build_stt` already
   selects the engine).
5. `/v1/audio/speech` then TTS generation sessions.

## Related

- [Architecture](architecture.md) — local cascade and network boundary today
- [CLI](cli.md) — `run` / `init` only
- [syllabix.yaml](syllabix-yaml.md) — LLM `online` already; TTS `online` rejected
- [Compare](compare.md) — no `serve`, SIP, or SFU in the repo today
- [Engines](engines.md) — model ids the worker should load
