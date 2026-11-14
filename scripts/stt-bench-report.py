#!/usr/bin/env python3
"""Print STT WER and latency for `bench-asr-worker --manifest` JSONL files.

One row per model × dataset × noise condition: corpus WER (sum errors / sum
ref_words), utterance count, p50/p95 elapsed_ms bucketed by clip length
(<3 s, 3–8 s, >8 s), mean RTF, this-worker peak RSS, cold-start, and fingerprint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

BUCKETS = (("<3s", 0, 3000), ("3-8s", 3000, 8001), (">8s", 8001, None))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q / 100 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def fingerprint_label(fp: dict) -> str:
    ram = fp.get("ram_bytes")
    ram_gib = f"{round(ram / 1024**3, 1)}GiB" if ram else "unknown-ram"
    backend = fp.get("ggml_backend") or "cpu"
    gpu = fp.get("gpu_name")
    label = (
        f"{fp.get('os')}/{fp.get('arch')} {fp.get('cpu')} "
        f"{fp.get('cpu_cores')}c {ram_gib} ggml={backend}"
    )
    if gpu:
        label = f"{label} gpu={gpu}"
    return label


def audio_ms(row: dict) -> float:
    value = row.get("audio_ms")
    if value is None:
        value = row.get("input_audio_ms") or 0
    return float(value)


def in_bucket(ms: float, start: int, end: int | None) -> bool:
    if ms < start:
        return False
    if end is None:
        return True
    return ms < end


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def fmt_ms(values: list[float], q: float) -> str:
    if not values:
        return "-"
    return f"{percentile(values, q):.1f}"


def snr_label(noise: object, snr: object) -> str:
    if noise == "clean" or snr is None:
        return "-"
    return f"{float(snr):.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path, help="bench-asr-worker JSONL files")
    args = parser.parse_args()
    records = load_records(args.jsonl)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("wer") is None:
            continue
        key = (
            record.get("model"),
            record.get("dataset") or "-",
            record.get("noise"),
            record.get("snr_db"),
        )
        groups[key].append(record)
    if not groups:
        print("no WER records", file=sys.stderr)
        return 1
    print(
        f"{'model':<28} {'dataset':<24} {'n_utt':>5} {'noise':<8} {'snr':>6} "
        f"{'wer':>8} {'ref_w':>7} "
        f"{'p50_<3':>8} {'p95_<3':>8} {'p50_3-8':>8} {'p95_3-8':>8} "
        f"{'p50_>8':>8} {'p95_>8':>8} {'mean_rtf':>9} {'peak_mib':>9} "
        f"{'load_ms':>8} {'first_ms':>8} {'thr':>4}  fingerprint"
    )
    for key in sorted(
        groups,
        key=lambda item: (
            str(item[0]),
            str(item[1]),
            str(item[2]),
            float(item[3]) if item[3] is not None else -1.0,
        ),
    ):
        rows = groups[key]
        errors = sum(row.get("errors") or 0 for row in rows)
        ref_words = sum(row.get("ref_words") or 0 for row in rows)
        wer = errors / ref_words if ref_words else 0.0
        n_utt = rows[0].get("n_utt") or len(rows)
        rtfs = [
            float(row["real_time_factor"])
            for row in rows
            if row.get("real_time_factor") is not None
        ]
        peaks = [row.get("memory_peak_bytes") or 0 for row in rows]
        peak_mib = max(peaks) / (1024 * 1024)
        load = rows[0].get("load_ms")
        first = rows[0].get("first_call_ms")
        threads = rows[0].get("threads")
        bucket_ms = {}
        for name, start, end in BUCKETS:
            elapsed = [
                float(row["elapsed_ms"])
                for row in rows
                if in_bucket(audio_ms(row), start, end)
            ]
            bucket_ms[name] = elapsed
        fp = fingerprint_label(rows[0].get("fingerprint") or {})
        model, dataset, noise, snr = key
        print(
            f"{str(model):<28} {str(dataset):<24} {int(n_utt):5d} {str(noise):<8} "
            f"{snr_label(noise, snr):>6} {wer:8.4f} {ref_words:7d} "
            f"{fmt_ms(bucket_ms['<3s'], 50):>8} {fmt_ms(bucket_ms['<3s'], 95):>8} "
            f"{fmt_ms(bucket_ms['3-8s'], 50):>8} {fmt_ms(bucket_ms['3-8s'], 95):>8} "
            f"{fmt_ms(bucket_ms['>8s'], 50):>8} {fmt_ms(bucket_ms['>8s'], 95):>8} "
            f"{(sum(rtfs) / len(rtfs) if rtfs else float('nan')):9.3f} "
            f"{peak_mib:9.1f} "
            f"{(f'{load:.0f}' if load is not None else '-'):>8} "
            f"{(f'{first:.0f}' if first is not None else '-'):>8} "
            f"{(str(threads) if threads is not None else '-'):>4}  {fp}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
