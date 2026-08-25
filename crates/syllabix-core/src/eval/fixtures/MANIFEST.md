# Fixed evaluation question corpus

These nine PCM16 WAV files were rendered once with the shipped native Qwen3-TTS path and are embedded in `syllabix bench`. Every benchmark profile uses the same bytes, so changing the pipeline TTS does not change the microphone input.

- Generator model: `qwen3-1.7` (`Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf`)
- Matching projector: `mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf`
- Language: `en`
- Sampler seed: `42`
- Voice: built-in fixed self-voice anchor
- Output: 16 kHz, mono, signed PCM16 WAV
- Model and generated corpus posture: Apache-2.0

| File | Prompt | SHA-256 |
|---|---|---|
| `greeting_001_turn_1.wav` | Hi, let us start with something simple: what can you do? | `2cc5cdc204f54597320ba5f0410c7ce26bbd1f965fb1daaa9bf698d7591eb139` |
| `greeting_001_turn_2.wav` | Tell me something interesting about yourself. | `514ae34bcce73d5945ead463a497c6e7aa5c96884cd96f557f3ce2d7cb274efe` |
| `space_fact_001_turn_1.wav` | Tell me a short fun fact about space please. | `65ad19697b793f7cd68b4e0dc163ee29015613db8e2c60b056965cccdf1215cd` |
| `space_fact_001_turn_2.wav` | Tell me which planet is the biggest one in our solar system. | `9800b9d284b4748478d1bb95f98a218eff90840c87fc58db571282d0d2f0fd71` |
| `arithmetic_001_turn_1.wav` | Hey, what is twenty plus thirty exactly? | `9d9b10150521238e98fdd38b5276836eba2a8a3a81510175e7d54947d5ae86d7` |
| `arithmetic_001_turn_2.wav` | What is the answer when you multiply five by five? | `16b95b8d7e9b3d7e8593183d3871624e4e850353d111023b30994cb017112821` |
| `history_001_turn_1.wav` | My favorite color is blue. Please remember that forever. | `70997dba2a57dbdb77fb03463266a93a0de27b1ecc79d3095e95c71bd6aaedc7` |
| `history_001_turn_2.wav` | So tell me, what is my favorite color again? | `dc8423d22f950a3f323a89c6a65da904bef853ff888246eaf463238f12a1cf85` |
| `long_sentence_001_turn_1.wav` | I am testing how you handle a longer sentence so here is one that keeps going for quite a while before asking how you are doing today | `1941e0a4ee6710e1f11e0aae1fa5aeba0de3e4d2b243503297c0dd9929250184` |
