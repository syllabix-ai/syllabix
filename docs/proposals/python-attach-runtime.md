# Proposal — Python attach to the native runtime

**Status:** Post-launch recommendation. Not a v0 change. `syllabix run`
stays one native binary with no Python, no `serve`, and no API key on the
default path.

**Lifecycle:** temporary; remove this file once the proposal is rejected or
superseded by the product and documentation PRs that ship the attach surface.

**Decision requested:** after v0 launches as-is, approve an embed path whose
DX is `import syllabix` in an existing Python app, while Python remains
optional for using Syllabix at all.

## 1. Outcome

A developer can add a voice session to a Python program without forking
Syllabix and without reimplementing VAD, AEC, or barge-in:

```python
import syllabix

with syllabix.run(barge_in=True) as session:
    for event in session:
        if event.name == "user":
            print(event.text)
```

`pip install syllabix` is enough. The package starts or attaches to the
**same native runtime** that `syllabix run` uses. Audio, echo control, and
interrupt stay in that process.

Default users still download the GitHub Release and run `./syllabix run`.
They never install Python.

## 2. What not to build

Do not split the product into “Syllabix Client” (Python VAD, interruption)
and “Syllabix Server” (models). That redraws the conversational physics:
barge-in is cancel + flush on the same duplex path that AEC3 and Silero
already share. Moving VAD to an SDK and weights to a remote process makes
interrupt a network feature and fights the privacy contract (utterance
audio would leave the box).

Do not make Python required for the next default user path.

Do not put ggml into CPython for the first cut (PyO3 wheels, GIL vs audio
threads, manylinux). In-process FFI is a later option if attach-stdio
misses a measured latency budget.

Do not ship `syllabix serve` on `0.0.0.0` in order to satisfy `import`.
Localhost-only attach is enough. A remote hosted voice API is a different
product and a different privacy doc.

Do not grow a Python audio stack (numpy/torch/onnx, a second Silero).

## 3. Roles

| Role | Owns | Lives in |
| --- | --- | --- |
| Runtime | Devices, AEC3, VAD, barge-in cancel/flush, STT/LLM/TTS, model cache | Native `syllabix` binary (today’s `run` loop without the TUI) |
| Shell | TUI | Existing CLI when stdout is a terminal |
| Attach client | Start/stop session, subscribe to events, send mute / barge-in / cancel | Python first; same protocol for Node/Go/Rust later |

“Client/server” later means **process roles**, not a cloud model farm.
An optional long-lived runtime on loopback is a sidecar for apps that
must not load ggml in-process. It is still the same machine until we
explicitly name a remote mode.

## 4. Native attach surface

Today `run_live` already accepts `events: Option<Sender<LoopEvent>>` and
`RuntimeControls`. Headless `run` discards those events. Tomorrow expose
them:

```text
syllabix attach --stdio
```

Parent process owns stdin/stdout. No HTTP, no bind address. Stderr stays
human (model-fetch progress), same as `run`.

Line-oriented JSON, one object per line, schema version in every message:

```json
{"v":1,"type":"event","name":"partial","turn":3,"text":"hello"}
{"v":1,"type":"cmd","name":"set","barge_in":true}
{"v":1,"type":"cmd","name":"cancel"}
{"v":1,"type":"cmd","name":"mute"}
{"v":1,"type":"cmd","name":"shutdown"}
```

Events map `LoopEvent` as it exists: `ready`, `user`, `partial`,
`assistant`, `thinking`, `aec`, `playback`, `timings`, plus `error`.
Commands map to `RuntimeControls` and `Cancel`. The first `ready` event
includes `protocol_version`; a client whose major does not match fails
closed.

Config remains `syllabix.yaml` in the working directory (and the env
vars the binary already honors). The Python package must not invent a
second yaml dialect.

Optional later: `syllabix attach --socket <path>` so several clients
share one long-lived runtime. Do not start there.

## 5. `import syllabix`

The PyPI package is a thin guest: find or unpack the native binary, spawn
`attach --stdio`, codec the JSONL stream, expose `Session`.

Binary resolution, in order:

1. `SYLLABIX_BIN` if set
2. `syllabix` on `PATH`
3. Platform artifact vendored in the wheel (same names as GitHub
   Releases: `syllabix-Linux-x86_64`, `syllabix-Darwin-arm64`, …)
4. If missing: download the matching Release, verify `SHA256SUMS`, cache
   under the package dir or `$SYLLABIX_CACHE_DIR`

Prefer wheels that already contain the binary for the four release
targets so first `run()` does not need network for the executable.
Models stay in `~/.cache/syllabix/models/v1` (`%LOCALAPPDATA%\syllabix\cache\models\v1`
on Windows). The Python package does not duplicate the manifest.

Package layout stays small (no native extension on the first cut):

```text
syllabix/
  __init__.py      # Session, run(), attach()
  protocol.py      # JSONL encode/decode
  binary.py        # find / unpack / verify native
```

`Session.close()` sends `shutdown` and waits. atexit and kill are
backstops so a crashed interpreter does not leave a mic process.

Suggested DX:

```python
import syllabix

session = syllabix.run()          # spawn native attach
session.on_partial(print)
session.barge_in = True
session.mute()
session.cancel()                  # same flush path as CLI --barge-in
session.close()
```

Iteration (`for event in session`) is the primitive; callbacks are sugar.

## 6. Order of work (after launch)

1. Treat `LoopEvent` / `RuntimeControls` as the session ABI. Additive
   changes only; version the JSON schema.
2. Implement `syllabix attach --stdio` with golden JSON fixtures (no
   weights in CI).
3. Publish `pip install syllabix`: vendored binary + `Session`.
4. Consider a loopback socket sidecar or PyO3 only if a measured embed
   case requires it.

Release CI already publishes four artifacts plus `SHA256SUMS`. A
follow-up job can copy those files into platform wheels so Python and
the README download path stay one compile graph.

## 7. Tests

- Rust: encode/decode of the attach codec; cancel/flush still covered by
  existing `barge_in_*` / `cancel_*` tests.
- Python: spawn a fake `attach --stdio` that emits canned events (no
  devices, no weights).
- Live: one ignored hardware test on profile A (open speakers), same
  echo and barge-in gates as `syllabix run`. If attach cannot pass those,
  the SDK does not ship.

## 8. Success

- `./syllabix run` unchanged: no Python, no server, offline after cache.
- `import syllabix` in an app starts a real duplex session.
- Interruption and AEC are still native runtime properties, not SDK
  features.
- Other languages can bind the same stdio protocol without a second
  engine.
