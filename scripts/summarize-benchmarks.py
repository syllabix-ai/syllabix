#!/usr/bin/env python3
"""Summarize kept benchmark runs into per-model CSVs and workload_benchmark.md.

Reads docs/eval/runs/*.jsonl (one fingerprinted record per model/case/machine)
and writes, fully regenerated on every run:

  docs/workload_benchmark/<component>-<model>.csv   one row per machine,
      each averaged across that machine's test cases, with a passed (x/y) column
  docs/workload_benchmark/workload_benchmark.md     one table per model,
      rows split by machine configuration

Regenerate after a run lands:  python3 scripts/summarize-benchmarks.py

Also supports --fingerprint <run.jsonl> (prints the filename stem for a raw
bench capture; used by scripts/contribute-performance.sh).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

STT_COLS = ["machine", "passed", "avg_elapsed_ms", "avg_first_output_ms",
            "avg_rtf", "avg_word_match", "avg_mem_after_mib", "avg_mem_peak_mib"]
TTS_COLS = ["machine", "passed", "avg_elapsed_ms", "avg_first_output_ms",
            "avg_rtf", "avg_generated_audio_ms", "avg_word_match",
            "avg_mem_after_mib", "avg_mem_peak_mib"]
LLM_COLS = ["machine", "passed", "avg_elapsed_ms", "avg_first_output_ms",
            "avg_prompt_tps", "avg_gen_tps", "avg_mem_after_mib", "avg_mem_peak_mib"]


def mib(b: float) -> float:
    return round(b / (1024 * 1024), 1)


def fingerprint_id(fp: dict) -> str:
    return hashlib.sha256(json.dumps(fp, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def machine_label(fp: dict) -> str:
    cpu = "m4" if "M4" in fp["cpu"] else "xeon"
    ram = round(fp["ram_bytes"] / 1024**3, 1)
    return f"{fp['os']}-{fp['arch']}-{cpu}-{fp['cpu_cores']}c-{ram}gib"


def machine_detail(fp: dict) -> str:
    return (f"`{fp['os']} / {fp['arch']}` · {fp['cpu']} · {fp['cpu_cores']} cores · "
            f"{round(fp['ram_bytes'] / 1024**3, 1)} GiB RAM · "
            f"binary {fp['binary_version']} (`{fp['git_sha'][:12]}`)")


def load(runs: Path) -> tuple[list[dict], list[dict]]:
    records, machines = [], {}
    for path in sorted(runs.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    for r in records:
        machines[machine_label(r["fingerprint"])] = r["fingerprint"]
    return records, [machines[k] for k in sorted(machines)]


def mean(vals: list) -> float:
    return round(statistics.fmean(vals), 3)


def summarize(records: list[dict], comp: str, model: str) -> tuple[list[str], list[list]]:
    """One row per machine: test cases averaged, correctness kept as passed (x/y)."""
    rows = [r for r in records if r["component"] == comp and r["model"] == model]
    if comp == "stt":
        cols = STT_COLS

        def row(label: str, rs: list[dict]) -> list:
            return [label, f"{sum(r['passed'] for r in rs)}/{len(rs)}",
                    mean([r["elapsed_ms"] for r in rs]), mean([r["first_output_ms"] for r in rs]),
                    mean([r["real_time_factor"] for r in rs]),
                    mean([r.get("word_match", 0) for r in rs]),
                    mib(statistics.fmean([r["memory_after_load_bytes"] for r in rs])),
                    mib(statistics.fmean([r["memory_peak_bytes"] for r in rs]))]
    elif comp == "tts":
        cols = TTS_COLS

        def row(label: str, rs: list[dict]) -> list:
            wm = [r["word_match"] for r in rs if "word_match" in r]
            return [label, f"{sum(r['passed'] for r in rs)}/{len(rs)}",
                    mean([r["elapsed_ms"] for r in rs]), mean([r["first_output_ms"] for r in rs]),
                    mean([r["real_time_factor"] for r in rs]),
                    mean([r["generated_audio_ms"] for r in rs]),
                    mean(wm) if wm else "n/a",
                    mib(statistics.fmean([r["memory_after_load_bytes"] for r in rs])),
                    mib(statistics.fmean([r["memory_peak_bytes"] for r in rs]))]
    else:
        cols = LLM_COLS

        def row(label: str, rs: list[dict]) -> list:
            return [label, f"{sum(r['passed'] for r in rs)}/{len(rs)}",
                    mean([r["elapsed_ms"] for r in rs]), mean([r["first_output_ms"] for r in rs]),
                    mean([r["prompt_tokens_per_second"] for r in rs]),
                    mean([r["generated_tokens_per_second"] for r in rs]),
                    mib(statistics.fmean([r["memory_after_load_bytes"] for r in rs])),
                    mib(statistics.fmean([r["memory_peak_bytes"] for r in rs]))]
    labels = sorted({machine_label(r["fingerprint"]) for r in rows})
    return cols, [row(label, [r for r in rows if machine_label(r["fingerprint"]) == label])
                  for label in labels]


def md_table(cols: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(f"`{v}`" for v in r) + " |" for r in rows]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("docs/eval/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/workload_benchmark"))
    parser.add_argument("--markdown", type=Path,
                        default=Path("docs/workload_benchmark/workload_benchmark.md"))
    parser.add_argument("--fingerprint", type=Path, help="print the run id for one JSONL file")
    args = parser.parse_args()

    if args.fingerprint:
        recs = [json.loads(l) for l in args.fingerprint.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len({json.dumps(r["fingerprint"], sort_keys=True) for r in recs}) != 1:
            print("every record must share one fingerprint", file=sys.stderr)
            return 1
        print(fingerprint_id(recs[0]["fingerprint"]))
        return 0

    records, machines = load(args.runs)
    if not records:
        print("no records found", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({(r["component"], r["model"]) for r in records})
    table: dict[tuple[str, str], tuple[list[str], list[list]]] = {}
    for comp, model in models:
        cols, rows = summarize(records, comp, model)
        table[(comp, model)] = (cols, rows)
        with (args.out_dir / f"{comp}-{model}.csv").open("w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(cols)
            w.writerows(rows)
    print(f"wrote {len(table)} per-model CSVs to {args.out_dir}/")

    detail = {machine_label(m): machine_detail(m) for m in machines}
    lines = ["# Workload benchmark",
             "",
             "One table per model; rows split by machine configuration. "
             "Each row averages that machine's test cases; `passed` is (x/y) cases.",
             ""]
    for comp, title in (("stt", "STT"), ("tts", "TTS"), ("llm", "LLM")):
        lines += [f"## {title}", ""]
        for (c, model), (cols, rows) in sorted(table.items()):
            if c != comp:
                continue
            lines.append(f"### `{model}` (`{comp}-{model}.csv`)")
            lines += md_table(cols, rows) + [""]
    lines += ["## Machines", ""]
    lines += [f"- `{label}`: {detail[label]}" for label in sorted(detail)]
    lines += ["",
              "*Autogenerated by `scripts/summarize-benchmarks.py` — do not edit by hand.*",
              ""]
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
