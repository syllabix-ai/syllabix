# Workload benchmark

One table per model; rows split by machine configuration. Each row averages that machine's test cases; `passed` is (x/y) cases.

## STT

### `moonshine-streaming-medium` (`stt-moonshine-streaming-medium.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `3/4` | `3078.25` | `3078.25` | `0.454` | `0.98` | `1061.1` | `1297.6` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `3/4` | `470.0` | `470.0` | `0.063` | `0.98` | `1058.2` | `1194.8` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `3/4` | `681.5` | `681.5` | `0.091` | `0.98` | `1051.1` | `1146.8` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `3/4` | `305.5` | `305.5` | `0.044` | `0.98` | `1150.7` | `1448.2` |

### `moonshine-streaming-small` (`stt-moonshine-streaming-small.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/4` | `1947.75` | `1947.75` | `0.291` | `0.991` | `630.1` | `845.9` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/4` | `257.25` | `257.25` | `0.038` | `0.991` | `624.2` | `721.7` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `3/4` | `413.25` | `413.25` | `0.059` | `0.897` | `620.2` | `758.7` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/4` | `173.5` | `173.5` | `0.025` | `0.991` | `704.8` | `927.2` |

### `qwen3-asr-0.6` (`stt-qwen3-asr-0.6.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/4` | `9734.25` | `9734.25` | `1.202` | `0.991` | `7436.1` | `8286.6` |

### `whisper-large-v3-turbo` (`stt-whisper-large-v3-turbo.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/4` | `187.5` | `187.5` | `0.042` | `1.0` | `317.3` | `436.0` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/4` | `14868.0` | `14868.0` | `3.865` | `1.0` | `1639.7` | `1775.0` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `4/4` | `17015.0` | `17015.0` | `4.231` | `1.0` | `1637.9` | `1767.6` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/4` | `3087.5` | `3087.5` | `0.842` | `1.0` | `1801.3` | `1801.3` |

### `whisper-large-v3-turbo-q5_0` (`stt-whisper-large-v3-turbo-q5_0.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/4` | `187.25` | `187.25` | `0.042` | `1.0` | `235.6` | `354.4` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/4` | `16539.25` | `16539.25` | `4.099` | `1.0` | `638.1` | `767.7` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `4/4` | `21928.0` | `21928.0` | `5.532` | `1.0` | `636.1` | `765.9` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/4` | `1365.5` | `1365.5` | `0.332` | `1.0` | `724.7` | `815.1` |

### `whisper-medium` (`stt-whisper-medium.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/4` | `235.25` | `235.25` | `0.048` | `0.964` | `295.1` | `408.3` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/4` | `8740.5` | `8740.5` | `2.111` | `0.964` | `1704.5` | `1824.0` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `4/4` | `10937.0` | `10937.0` | `2.629` | `0.964` | `1703.0` | `1822.7` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/4` | `3135.25` | `3135.25` | `0.746` | `0.964` | `1810.2` | `1889.8` |

### `whisper-medium-q5_0` (`stt-whisper-medium-q5_0.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/4` | `593.5` | `593.5` | `0.078` | `0.964` | `228.9` | `380.1` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/4` | `10490.5` | `10490.5` | `2.631` | `0.964` | `756.0` | `875.5` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `4/4` | `13140.75` | `13140.75` | `3.286` | `0.964` | `754.1` | `873.9` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/4` | `1284.75` | `1284.75` | `0.303` | `0.964` | `819.4` | `913.1` |

### `whisper-small` (`stt-whisper-small.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/4` | `1488.75` | `1488.75` | `0.149` | `0.964` | `268.0` | `410.9` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/4` | `3075.75` | `3075.75` | `0.766` | `0.964` | `584.8` | `685.3` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `4/4` | `3797.5` | `3797.5` | `0.902` | `0.964` | `582.8` | `683.7` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/4` | `884.5` | `884.5` | `0.263` | `0.964` | `695.8` | `777.7` |

## TTS

### `kokoro` (`tts-kokoro.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `2/3` | `10537.667` | `10537.667` | `1.591` | `6900.0` | `1.0` | `714.3` | `1216.3` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `3/3` | `1115.0` | `1115.0` | `0.183` | `6900.0` | `1.0` | `705.5` | `1069.2` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `3/3` | `1508.0` | `1508.0` | `0.258` | `6900.0` | `1.0` | `707.5` | `1100.8` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `2/3` | `1919.333` | `1919.333` | `0.291` | `6900.0` | `1.0` | `863.2` | `1312.5` |

### `pocket-tts` (`tts-pocket-tts.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `3/3` | `13254.333` | `2383.0` | `2.468` | `5520.0` | `1.0` | `323.9` | `616.2` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `3/3` | `2466.333` | `493.333` | `0.471` | `5333.333` | `1.0` | `317.6` | `483.1` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `3/3` | `5692.667` | `1014.0` | `0.977` | `5893.333` | `1.0` | `320.3` | `657.5` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `3/3` | `1955.0` | `291.0` | `0.317` | `5600.0` | `1.0` | `337.6` | `703.7` |

### `qwen3-0.6` (`tts-qwen3-0.6.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `3/3` | `2207.667` | `1420.0` | `0.4` | `6560.0` | `1.0` | `1156.1` | `1170.4` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `3/3` | `9931.333` | `6899.0` | `1.59` | `6666.667` | `1.0` | `6135.4` | `6213.4` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `3/3` | `11138.0` | `7713.0` | `1.906` | `5866.667` | `1.0` | `6134.6` | `6212.4` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `3/3` | `15352.667` | `10347.667` | `2.73` | `5520.0` | `1.0` | `5099.8` | `5099.8` |

### `qwen3-1.7` (`tts-qwen3-1.7.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_rtf | avg_generated_audio_ms | avg_word_match | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `3/3` | `1775.333` | `1117.333` | `0.302` | `5973.333` | `1.0` | `1797.3` | `1820.8` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `3/3` | `11011.667` | `7683.0` | `1.654` | `6800.0` | `1.0` | `7870.9` | `7948.4` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `2/3` | `14300.667` | `9830.667` | `2.163` | `6800.0` | `1.0` | `7869.6` | `7946.9` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `2/3` | `12027.667` | `8592.0` | `2.061` | `6053.333` | `1.0` | `5158.1` | `5358.8` |

## LLM

### `lfm2.5-2.6b` (`llm-lfm2.5-2.6b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/5` | `742.8` | `615.2` | `705.774` | `215.925` | `780.0` | `827.0` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/5` | `3124.2` | `1454.6` | `51.902` | `15.66` | `4899.2` | `4932.3` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `4/5` | `4034.6` | `1783.2` | `42.607` | `12.183` | `4898.4` | `4931.6` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/5` | `676.0` | `213.8` | `354.062` | `59.548` | `3564.1` | `3580.3` |

### `lfm2.5-230m` (`llm-lfm2.5-230m.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/5` | `82.4` | `47.6` | `1645.048` | `592.396` | `383.1` | `396.9` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/5` | `305.0` | `223.2` | `346.015` | `282.086` | `1779.1` | `1790.9` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `4/5` | `390.6` | `193.2` | `399.235` | `122.037` | `1778.0` | `1790.0` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/5` | `101.0` | `34.0` | `2296.972` | `463.502` | `1715.5` | `1720.7` |

### `lfm2.5-350m` (`llm-lfm2.5-350m.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `5/5` | `75.0` | `51.2` | `1529.594` | `640.623` | `383.2` | `397.1` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `5/5` | `365.0` | `289.0` | `267.128` | `332.568` | `1913.6` | `1929.6` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `5/5` | `449.0` | `268.2` | `287.574` | `263.604` | `1912.9` | `1928.9` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `5/5` | `80.0` | `43.6` | `1792.51` | `468.679` | `1782.4` | `1787.6` |

### `llama-3.2-1b` (`llm-llama-3.2-1b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/5` | `561.4` | `496.8` | `418.705` | `440.157` | `780.5` | `838.4` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `5/5` | `1347.4` | `744.4` | `97.488` | `227.667` | `5392.0` | `5408.8` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `5/5` | `1431.4` | `752.2` | `96.657` | `221.266` | `5388.2` | `5406.5` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `5/5` | `285.4` | `121.8` | `597.035` | `132.676` | `4905.2` | `4919.0` |

### `qwen3.5-0.8b` (`llm-qwen3.5-0.8b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `0/5` | `1149.8` | `598.8` | `467.949` | `217.943` | `938.6` | `991.9` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `0/5` | `2233.8` | `561.6` | `125.587` | `41.39` | `3904.7` | `3927.9` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `0/5` | `2952.8` | `607.2` | `117.115` | `28.887` | `3903.4` | `3925.5` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `0/5` | `1325.4` | `102.8` | `688.265` | `100.335` | `3770.8` | `3780.1` |

### `qwen3.5-2b` (`llm-qwen3.5-2b.csv`)
| machine | passed | avg_elapsed_ms | avg_first_output_ms | avg_prompt_tps | avg_gen_tps | avg_mem_after_mib | avg_mem_peak_mib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090` | `4/5` | `242.4` | `119.4` | `635.137` | `182.138` | `1342.1` | `1362.2` |
| `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu` | `4/5` | `2218.0` | `1028.4` | `72.514` | `19.428` | `4970.2` | `4996.5` |
| `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu` | `5/5` | `2730.0` | `1085.2` | `68.852` | `17.849` | `4968.5` | `4993.6` |
| `macos-aarch64-apple-m4-10c-16.0gib-cpu` | `4/5` | `571.2` | `176.0` | `425.573` | `59.409` | `4508.6` | `4518.5` |

## Machines

- `linux-x86_64-amd-ryzen-threadripper-pro-3955wx-16-cores-5c-503.5gib-vulkan-nvidia-geforce-rtx-3090`: `linux / x86_64` · AMD Ryzen Threadripper PRO 3955WX 16-Cores · 5 cores · 503.5 GiB RAM · ggml `vulkan` · GPU NVIDIA GeForce RTX 3090 (24.2 GiB VRAM) · binary 0.1.0 (`06986721f559`)
- `linux-x86_64-intel-r-xeon-r-processor-4c-15.6gib-cpu`: `linux / x86_64` · Intel(R) Xeon(R) Processor · 4 cores · 15.6 GiB RAM · ggml `cpu` · binary 0.1.0 (`3eb307b95331`)
- `linux-x86_64-intel-r-xeon-r-processor-8c-15.6gib-cpu`: `linux / x86_64` · Intel(R) Xeon(R) Processor · 8 cores · 15.6 GiB RAM · ggml `cpu` · binary 0.1.0 (`d9b271b8fbd6`)
- `macos-aarch64-apple-m4-10c-16.0gib-cpu`: `macos / aarch64` · Apple M4 · 10 cores · 16.0 GiB RAM · ggml `cpu` · binary 0.1.0 (`c0a70d92efcd`)

*Autogenerated by `scripts/summarize-benchmarks.py` — do not edit by hand.*
