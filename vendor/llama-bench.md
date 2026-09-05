# llama-bench (v0 GGUFs)

Not the product README. Scores for the three first-run cache GGUFs at the vendored llama.cpp pin.

| | |
| --- | --- |
| Pin | `ece963f41b0b02d7a0d61436ae365762c073a4c8` (`vendor/llama.cpp`) |
| Tool | upstream `llama-bench` at that pin (tools are not vendored into Syllabix) |
| Tests | `pp512` prompt, `tg32` decode (voice cares about **tg**) |
| Flags | `-ngl 0 -t 4 -p 512 -n 32 -r 3` unless a row says otherwise |
| Build | `Release`, `GGML_NATIVE=ON`, Metal/CUDA/BLAS off on Linux |

These numbers are **llama-bench**, not `syllabix`. Linux/Windows `syllabix` stays portable CPU (`n_gpu_layers=0`, max 4 threads). Darwin `syllabix` compiles Metal + embedded metallib + Accelerate, uses `n_gpu_layers=-1`, and keeps the 4-thread cap. M4 llama-bench with Metal+Accelerate is ~100 t/s tg on 0.8B; compare a Darwin `syllabix` build to the tables below.

GGUF files (SHA-256 in `crates/syllabix-core/src/models/manifest.rs`):

| id | file |
| --- | --- |
| `llama-3.2-1b` | `Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| `qwen3.5-0.8b` | `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` |
| `qwen3.5-2b` | `Qwen_Qwen3.5-2B-Q4_K_M.gguf` |

Cache: `$SYLLABIX_CACHE_DIR/models/v1` or `~/.cache/syllabix/models/v1`.

## Linux x86_64 (this agent)

- Host: Linux 6.12 KVM, 4× Intel Xeon (family 6 model 207), 1 thread/core
- Date: 2026-08-21
- `nproc=4`, so `-t 4` is also “all cores”

| model | size | params | backend | threads | test | t/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | CPU | 4 | pp512 | 470.43 ± 176.90 |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | CPU | 4 | tg32 | 34.61 ± 0.71 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | CPU | 4 | pp512 | 521.59 ± 5.06 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | CPU | 4 | tg32 | 43.65 ± 1.01 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | CPU | 4 | pp512 | 277.01 ± 8.23 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | CPU | 4 | tg32 | 19.40 ± 0.84 |

`build: ece963f (303)`

On this host, Qwen3.5 0.8B decode is faster than Llama 3.2 1B. That does **not** contradict a slow Mac `syllabix run`: this table is AVX-512 CPU llama-bench with native kernels. Apple Silicon + portable ggml + no Metal is a different machine.

## macOS Apple Silicon (M4)

- Host: MTL0 **Apple M4**, `MTLGPUFamilyApple9`, unified memory, recommended working set 12.7 GiB. Tensor API disabled (pre-M5).
- Date: 2026-08-21
- Build: same pin, `Release`, `GGML_NATIVE=ON`, **Metal + embedded metallib** (`build: ece963f41 (10450)`)
- llama-bench reports backend **`MTL,BLAS`** on every row, including `-ngl 0`, because this binary was built with Metal and Accelerate. Darwin `syllabix` now compiles that same pair (`n_gpu_layers=-1`). Linux `syllabix` does not.
- zsh treated `#` comments as commands (`command not found: #`); the three tables below are still A / B / C in order.

Commands (zsh-safe, no `#` lines):

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout ece963f41b0b02d7a0d61436ae365762c073a4c8
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build --target llama-bench -j
CACHE="${SYLLABIX_CACHE_DIR:-$HOME/.cache/syllabix}/models/v1"
MODELS="$CACHE/Llama-3.2-1B-Instruct-Q4_K_M.gguf,$CACHE/Qwen_Qwen3.5-0.8B-Q4_K_M.gguf,$CACHE/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
BENCH=./build/bin/llama-bench

$BENCH -ngl 0 -t 4 -p 512 -n 32 -r 3 -o md -m "$MODELS"
$BENCH -ngl 0 -t "$(sysctl -n hw.ncpu)" -p 512 -n 32 -r 3 -o md -m "$MODELS"
$BENCH -ngl 99 -p 512 -n 32 -r 3 -o md -m "$MODELS"
```

Voice cares about **tg32**. On this M4, Qwen 0.8B decode is ~100 t/s with Metal in the bench binary (~84 t/s Llama 1B at `-ngl 0 -t 4`; ~106 t/s Llama at `-ngl 99`). Extra CPU threads (`-t 10`) **hurt** decode. Prefill (`pp512`) is the big `-ngl 99` win (about 4×). Syllabix felt slow because it never compiles this Metal/BLAS path.

### A. `-ngl 0 -t 4`

| model | size | params | backend | threads | test | t/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | MTL,BLAS | 4 | pp512 | 341.54 ± 3.32 |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | MTL,BLAS | 4 | tg32 | 84.51 ± 0.37 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | MTL,BLAS | 4 | pp512 | 459.57 ± 8.41 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | MTL,BLAS | 4 | tg32 | 100.58 ± 0.57 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | MTL,BLAS | 4 | pp512 | 193.03 ± 8.31 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | MTL,BLAS | 4 | tg32 | 47.58 ± 0.10 |

### B. `-ngl 0 -t 10` (`hw.ncpu`)

| model | size | params | backend | threads | test | t/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | MTL,BLAS | 10 | pp512 | 438.82 ± 2.76 |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | MTL,BLAS | 10 | tg32 | 55.50 ± 15.22 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | MTL,BLAS | 10 | pp512 | 394.86 ± 9.12 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | MTL,BLAS | 10 | tg32 | 56.01 ± 9.39 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | MTL,BLAS | 10 | pp512 | 230.10 ± 12.07 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | MTL,BLAS | 10 | tg32 | 33.33 ± 0.82 |

### C. `-ngl 99` (Metal layers; 4 threads)

| model | size | params | backend | threads | test | t/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | MTL,BLAS | 4 | pp512 | 1254.62 ± 48.94 |
| llama 1B Q4_K - Medium | 762.81 MiB | 1.24 B | MTL,BLAS | 4 | tg32 | 106.25 ± 5.54 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | MTL,BLAS | 4 | pp512 | 1802.32 ± 29.66 |
| qwen35 0.8B Q4_K - Medium | 542.31 MiB | 772.85 M | MTL,BLAS | 4 | tg32 | 103.30 ± 0.48 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | MTL,BLAS | 4 | pp512 | 807.80 ± 16.32 |
| qwen35 2B Q4_K - Medium | 1.29 GiB | 1.94 B | MTL,BLAS | 4 | tg32 | 56.72 ± 0.43 |

## Darwin `syllabix` checks (sequence 26)

Linux CI cannot compile Metal. On Apple Silicon, from this checkout:

```bash
cargo test -p syllabix-native n_gpu_layers_matches_os -- --nocapture
cargo test --workspace   # launch stack; extra yaml models via SYLLABIX_NATIVE_MODELS
# system info from the unit test should mention Metal; Whisper GPU is on.
# Live tok/s vs the M4 tables above (tg32, default llama-3.2-1b, 4 threads):
# enable diagnostics in syllabix.yaml (`diagnostics: {timestamps: true, audio: true}`)
# then read llm_first_token offsets from target/turn-debug/turn-*/turn.json.
cargo run -p syllabix --release -- run
```

KleidiAI SME did not link (`___arm_tpidr2_save`). It stays off. Default GGUF is `lfm2.5-2.6b`. Metal + Accelerate are the Darwin path.
