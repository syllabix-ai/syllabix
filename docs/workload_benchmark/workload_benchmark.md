# Workload benchmark

One table per model; rows split by machine configuration. Each row averages that machine's test cases; `passed` is (x/y) cases.

## STT

### `moonshine-streaming-medium` (`stt-moonshine-streaming-medium.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `3/4` | `406.0` | `406.0` | `0.056` | `0.98` | `1049.0` | `1180.7` |
| `linux-x86_64-xeon-8c-15.6gib` | `3/4` | `681.5` | `681.5` | `0.091` | `0.98` | `1051.1` | `1146.8` |
| `macos-aarch64-m4-10c-16.0gib` | `3/4` | `305.5` | `305.5` | `0.044` | `0.98` | `1150.7` | `1448.2` |

### `moonshine-streaming-small` (`stt-moonshine-streaming-small.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `3/4` | `225.75` | `225.75` | `0.035` | `0.92` | `619.2` | `701.1` |
| `linux-x86_64-xeon-8c-15.6gib` | `3/4` | `413.25` | `413.25` | `0.059` | `0.897` | `620.2` | `758.7` |
| `macos-aarch64-m4-10c-16.0gib` | `4/4` | `173.5` | `173.5` | `0.025` | `0.991` | `704.8` | `927.2` |

### `whisper-large-v3-turbo` (`stt-whisper-large-v3-turbo.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/4` | `11638.25` | `11638.25` | `2.901` | `1.0` | `1637.5` | `1767.2` |
| `linux-x86_64-xeon-8c-15.6gib` | `4/4` | `17015.0` | `17015.0` | `4.231` | `1.0` | `1637.9` | `1767.6` |
| `macos-aarch64-m4-10c-16.0gib` | `4/4` | `3087.5` | `3087.5` | `0.842` | `1.0` | `1801.3` | `1801.3` |

### `whisper-large-v3-turbo-q5_0` (`stt-whisper-large-v3-turbo-q5_0.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/4` | `14501.25` | `14501.25` | `3.621` | `1.0` | `635.8` | `771.0` |
| `linux-x86_64-xeon-8c-15.6gib` | `4/4` | `21928.0` | `21928.0` | `5.532` | `1.0` | `636.1` | `765.9` |
| `macos-aarch64-m4-10c-16.0gib` | `4/4` | `1365.5` | `1365.5` | `0.332` | `1.0` | `724.7` | `815.1` |

### `whisper-medium` (`stt-whisper-medium.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/4` | `7481.0` | `7481.0` | `1.802` | `0.964` | `1702.0` | `1821.6` |
| `linux-x86_64-xeon-8c-15.6gib` | `4/4` | `10937.0` | `10937.0` | `2.629` | `0.964` | `1703.0` | `1822.7` |
| `macos-aarch64-m4-10c-16.0gib` | `4/4` | `3135.25` | `3135.25` | `0.746` | `0.964` | `1810.2` | `1889.8` |

### `whisper-medium-q5_0` (`stt-whisper-medium-q5_0.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/4` | `8804.25` | `8804.25` | `2.161` | `0.964` | `753.7` | `873.2` |
| `linux-x86_64-xeon-8c-15.6gib` | `4/4` | `13140.75` | `13140.75` | `3.286` | `0.964` | `754.1` | `873.9` |
| `macos-aarch64-m4-10c-16.0gib` | `4/4` | `1284.75` | `1284.75` | `0.303` | `0.964` | `819.4` | `913.1` |

### `whisper-small` (`stt-whisper-small.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/4` | `2409.0` | `2409.0` | `0.585` | `0.964` | `581.5` | `663.5` |
| `linux-x86_64-xeon-8c-15.6gib` | `4/4` | `3797.5` | `3797.5` | `0.902` | `0.964` | `582.8` | `683.7` |
| `macos-aarch64-m4-10c-16.0gib` | `4/4` | `884.5` | `884.5` | `0.263` | `0.964` | `695.8` | `777.7` |

## TTS

### `kokoro` (`tts-kokoro.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `3/3` | `1051.667` | `1051.667` | `0.17` | `6900.0` | `1.0` | `702.1` | `1082.4` |
| `linux-x86_64-xeon-8c-15.6gib` | `3/3` | `1508.0` | `1508.0` | `0.258` | `6900.0` | `1.0` | `707.5` | `1100.8` |
| `macos-aarch64-m4-10c-16.0gib` | `2/3` | `1919.333` | `1919.333` | `0.291` | `6900.0` | `1.0` | `863.2` | `1312.5` |

### `pocket-tts` (`tts-pocket-tts.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `3/3` | `2939.667` | `546.0` | `0.513` | `5813.333` | `1.0` | `310.2` | `596.2` |
| `linux-x86_64-xeon-8c-15.6gib` | `3/3` | `5692.667` | `1014.0` | `0.977` | `5893.333` | `1.0` | `320.3` | `657.5` |
| `macos-aarch64-m4-10c-16.0gib` | `3/3` | `1955.0` | `291.0` | `0.317` | `5600.0` | `1.0` | `337.6` | `703.7` |

### `qwen3-0.6` (`tts-qwen3-0.6.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `3/3` | `8039.333` | `5427.333` | `1.262` | `6720.0` | `1.0` | `6131.5` | `6209.4` |
| `linux-x86_64-xeon-8c-15.6gib` | `3/3` | `11138.0` | `7713.0` | `1.906` | `5866.667` | `1.0` | `6134.6` | `6212.4` |
| `macos-aarch64-m4-10c-16.0gib` | `3/3` | `15352.667` | `10347.667` | `2.73` | `5520.0` | `1.0` | `5099.8` | `5099.8` |

### `qwen3-1.7` (`tts-qwen3-1.7.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `3/3` | `10185.667` | `7211.0` | `1.58` | `6586.667` | `1.0` | `7868.6` | `7946.7` |
| `linux-x86_64-xeon-8c-15.6gib` | `2/3` | `14300.667` | `9830.667` | `2.163` | `6800.0` | `1.0` | `7869.6` | `7946.9` |
| `macos-aarch64-m4-10c-16.0gib` | `2/3` | `12027.667` | `8592.0` | `2.061` | `6053.333` | `1.0` | `5158.1` | `5358.8` |

## LLM

### `lfm2.5-2.6b` (`llm-lfm2.5-2.6b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/5` | `2986.6` | `1177.8` | `63.821` | `16.386` | `4810.2` | `4820.1` |
| `linux-x86_64-xeon-8c-15.6gib` | `4/5` | `4034.6` | `1783.2` | `42.607` | `12.183` | `4898.4` | `4931.6` |
| `macos-aarch64-m4-10c-16.0gib` | `4/5` | `676.0` | `213.8` | `354.062` | `59.548` | `3564.1` | `3580.3` |

### `lfm2.5-230m` (`llm-lfm2.5-230m.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/5` | `295.4` | `185.4` | `416.133` | `313.568` | `1777.8` | `1789.8` |
| `linux-x86_64-xeon-8c-15.6gib` | `4/5` | `390.6` | `193.2` | `399.235` | `122.037` | `1778.0` | `1790.0` |
| `macos-aarch64-m4-10c-16.0gib` | `4/5` | `101.0` | `34.0` | `2296.972` | `463.502` | `1715.5` | `1720.7` |

### `lfm2.5-350m` (`llm-lfm2.5-350m.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `5/5` | `305.0` | `231.0` | `333.792` | `344.432` | `1911.9` | `1928.0` |
| `linux-x86_64-xeon-8c-15.6gib` | `5/5` | `449.0` | `268.2` | `287.574` | `263.604` | `1912.9` | `1928.9` |
| `macos-aarch64-m4-10c-16.0gib` | `5/5` | `80.0` | `43.6` | `1792.51` | `468.679` | `1782.4` | `1787.6` |

### `llama-3.2-1b` (`llm-llama-3.2-1b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `5/5` | `1182.2` | `625.6` | `115.86` | `228.784` | `5389.6` | `5406.6` |
| `linux-x86_64-xeon-8c-15.6gib` | `5/5` | `1431.4` | `752.2` | `96.657` | `221.266` | `5388.2` | `5406.5` |
| `macos-aarch64-m4-10c-16.0gib` | `5/5` | `285.4` | `121.8` | `597.035` | `132.676` | `4905.2` | `4919.0` |

### `qwen3.5-0.8b` (`llm-qwen3.5-0.8b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `0/5` | `3411.4` | `517.2` | `136.257` | `39.857` | `3902.9` | `3925.1` |
| `linux-x86_64-xeon-8c-15.6gib` | `0/5` | `2952.8` | `607.2` | `117.115` | `28.887` | `3903.4` | `3925.5` |
| `macos-aarch64-m4-10c-16.0gib` | `0/5` | `1325.4` | `102.8` | `688.265` | `100.335` | `3770.8` | `3780.1` |

### `qwen3.5-2b` (`llm-qwen3.5-2b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-xeon-4c-15.6gib` | `4/5` | `2515.6` | `912.4` | `81.73` | `19.825` | `4968.2` | `4993.6` |
| `linux-x86_64-xeon-8c-15.6gib` | `5/5` | `2730.0` | `1085.2` | `68.852` | `17.849` | `4968.5` | `4993.6` |
| `macos-aarch64-m4-10c-16.0gib` | `4/5` | `571.2` | `176.0` | `425.573` | `59.409` | `4508.6` | `4518.5` |

## Machines

- `linux-x86_64-xeon-4c-15.6gib`: `linux / x86_64` · Intel(R) Xeon(R) Processor · 4 cores · 15.6 GiB RAM · binary 0.1.0 (`d9b271b8fbd6`)
- `linux-x86_64-xeon-8c-15.6gib`: `linux / x86_64` · Intel(R) Xeon(R) Processor · 8 cores · 15.6 GiB RAM · binary 0.1.0 (`d9b271b8fbd6`)
- `macos-aarch64-m4-10c-16.0gib`: `macos / aarch64` · Apple M4 · 10 cores · 16.0 GiB RAM · binary 0.1.0 (`c0a70d92efcd`)

*Autogenerated by `scripts/summarize-benchmarks.py` — do not edit by hand.*
