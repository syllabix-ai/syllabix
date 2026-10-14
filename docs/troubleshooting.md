# Troubleshooting

| Symptom | What to do |
| --- | --- |
| Downloaded app won't run | Missing `chmod +x`, macOS Gatekeeper, or Windows SmartScreen. See the rows below. |
| macOS: “cannot be opened because Apple cannot check it for malicious software” | Right-click → Open, or `xattr -d com.apple.quarantine syllabix`. See [install.md](install.md). |
| Windows SmartScreen warning | More info → Run anyway. |
| Hash doesn't match `SHA256SUMS` | Re-download both files from the same Release and [check again](#check-that-your-download-isnt-damaged-optional). |
| macOS asks to approve microphone access | Grant it once in System Settings → Privacy & Security → Microphone. |
| `error: No speakers found` / no microphone | `syllabix run` needs input **and** output devices at launch. Check OS sound settings, then rerun. On a headless box, ensure both devices are available. |
| The agent interrupts itself on laptop speakers | Full-duplex AEC3 is on by default and calibrates automatically for ~10 s; let calibration finish before speaking. If it still self-interrupts, use headphones and include the device names from the startup line in a bug report. |
| First `run` is slow | It fetches Silero, Whisper `small`, LFM2.5-2.6B, and Pocket TTS (~2.2 GB) into the model cache with progress lines. Later runs reuse the cache and never touch the network. |
| Replies are cut off mid-sentence when you speak over them | This is barge-in and applies only with `run --barge-in`. Without the flag, the agent finishes its sentence first. |
| The agent speaks Qwen's reasoning aloud | It should not: `<think>…</think>` is stripped before TTS and hidden in the TUI. Thinking stays off unless yaml sets `pipeline.llm.thinking: true`. If you hear chain-of-thought, enable diagnostics (`turn.json` keeps the full `llm_text`) and file a bug. |
| STT text does not match what you said | Enable `diagnostics: {timestamps: true, audio: true}`, then listen to `utterance.wav` (what Whisper received, including the 200 ms onset preroll) versus `clean.wav` (post-AEC). If `utterance.wav` sounds wrong but `clean.wav` sounds right, report the sidecar `stt_text` plus both files. |
| `SYLLABIX_LLM_API_KEY is required…` at `run` start | Yaml selects `pipeline.llm.provider: online`. Supply the key for one run (`SYLLABIX_LLM_API_KEY=sk-… syllabix run`) or export it. Keys are never read from yaml. Switch the provider back to `local` for the on-device LLM. |
| `SYLLABIX_STT_API_KEY is required…` / `SYLLABIX_TTS_API_KEY is required…` | Yaml selects online STT or TTS. Export that stage's key. Keys are never read from yaml. |
| `SYLLABIX_SERVE_TOKEN is required…` | `syllabix serve` on a non-loopback bind needs that env var. Loopback may omit it. [CLI serve](cli.md#serve). |
| Every reply is "Sorry, I could not reach the language model." | The cloud turn failed (bad key → HTTP 401, endpoint down, or idle timeout) and spoke its fallback instead of hanging. Check the key, `base_url`, and the endpoint; diagnostics sidecars carry the endpoint and request id. |
| Linux build fails linking ALSA | Contributors need `libasound2-dev`. Users of the Release binary never compile anything. |

## Check that your download isn't damaged (optional)

Check the SHA-256 of the file for your platform against the latest release's SHA256SUMS.

| Platform | File |
| --- | --- |
| Linux x86_64 | [`syllabix-Linux-x86_64`](https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Linux-x86_64) |
| macOS Apple Silicon | [`syllabix-Darwin-arm64`](https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Darwin-arm64) |
| macOS Intel | [`syllabix-Darwin-x86_64`](https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Darwin-x86_64) |
| Windows x64 | [`syllabix-Windows-x86_64.exe`](https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Windows-x86_64.exe) |

Latest checksums: [`SHA256SUMS`](https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS)
