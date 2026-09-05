# Troubleshooting

| Symptom | What to do |
| --- | --- |
| Downloaded binary won't verify | Re-download the artifact and `SHA256SUMS` from the same Release; the checksum must pass before you run. |
| macOS: “cannot be opened because Apple cannot check it for malicious software” | After checksum verify: right-click → Open, or `xattr -d com.apple.quarantine syllabix`. See [install.md](install.md). |
| Windows SmartScreen warning | More info → Run anyway, after the hash matches `SHA256SUMS`. |
| macOS asks to approve microphone access | Grant it once in System Settings → Privacy & Security → Microphone. |
| `error: No speakers found` / no microphone | `syllabix run` needs input **and** output devices at launch. Check OS sound settings, then rerun. On a headless box there is nothing to talk to — failing fast is correct. |
| The agent interrupts itself on laptop speakers | Full-duplex AEC3 is on by default and calibrates automatically for ~10 s; let calibration finish before speaking. If it still self-interrupts, use headphones and include the device names from the startup line in a bug report. |
| First `run` is slow | It fetches Silero, Whisper `small`, LFM2.5-2.6B, and Pocket TTS (~2.2 GB) into the model cache with progress lines. Later runs reuse the cache and never touch the network. |
| Replies are cut off mid-sentence when you speak over them | That is barge-in — but only with `run --barge-in`. Without the flag the agent finishes its sentence first; that is the default, not a bug. |
| The agent speaks Qwen's reasoning aloud | It should not: `<think>…</think>` is stripped before TTS and hidden in the TUI. Thinking stays off unless yaml sets `pipeline.llm.thinking: true`. If you hear chain-of-thought, enable diagnostics (`turn.json` keeps the full `llm_text`) and file a bug. |
| STT text does not match what you said | Enable `diagnostics: {timestamps: true, audio: true}`, then listen to `utterance.wav` (what Whisper received, including the 200 ms onset preroll) versus `clean.wav` (post-AEC). If `utterance.wav` sounds wrong but `clean.wav` sounds right, report the sidecar `stt_text` plus both files. |
| `SYLLABIX_LLM_API_KEY is required…` at `run` start | Yaml selects `pipeline.llm.provider: online`. Supply the key for one run (`SYLLABIX_LLM_API_KEY=sk-… syllabix run`) or export it. Keys are never read from yaml. Switch the provider back to `local` for the on-device LLM. |
| Every reply is "Sorry, I could not reach the language model." | The cloud turn failed (bad key → HTTP 401, endpoint down, or idle timeout) and spoke its fallback instead of hanging. Check the key, `base_url`, and the endpoint; diagnostics sidecars carry the endpoint and request id. |
| Linux build fails linking ALSA | Contributors need `libasound2-dev`. Users of the Release binary never compile anything. |
