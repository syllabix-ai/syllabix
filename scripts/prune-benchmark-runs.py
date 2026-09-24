#!/usr/bin/env python3
"""Delete superseded benchmark runs, keeping the newest full-matrix run per machine.

A machine is identified by (os, arch, cpu, cpu_cores, ram_bytes) plus optional
compute-backend fields (ggml_backend, gpu_name, gpu_vram_bytes) — the parts of
the bench fingerprint that describe hardware, ignoring binary_version/git_sha.
Schema 3 runs omit the backend keys; missing keys compare as None so old CPU
runs stay grouped together. CPU vs Vulkan (or Metal) on the same host stay
distinct.
For each machine the run with the most records wins (the full STT x LLM x TTS
matrix grows as models are added); ties break toward the lexicographically
largest git_sha. Everything else (legacy single-axis runs, superseded matrix
runs) is deleted.

Usage:
    scripts/prune-benchmark-runs.py          # dry run, prints keep/delete
    scripts/prune-benchmark-runs.py --apply  # actually delete
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MACHINE_KEYS = (
    "os",
    "arch",
    "cpu",
    "cpu_cores",
    "ram_bytes",
    "ggml_backend",
    "gpu_name",
    "gpu_vram_bytes",
)


def load_runs(runs: Path) -> list[tuple[Path, dict, int]]:
    found = []
    for path in sorted(runs.glob("*.jsonl")):
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError) as error:
            print(f"skip {path.name}: unreadable ({error})", file=sys.stderr)
            continue
        if not records:
            print(f"skip {path.name}: empty", file=sys.stderr)
            continue
        fps = {json.dumps(r.get("fingerprint"), sort_keys=True) for r in records}
        if len(fps) != 1:
            print(f"skip {path.name}: mixed fingerprints", file=sys.stderr)
            continue
        found.append((path, records[0]["fingerprint"], len(records)))
    return found


def pick_keep(found: list[tuple[Path, dict, int]]) -> tuple[set[Path], set[Path]]:
    by_machine: dict[tuple, list[tuple[Path, dict, int]]] = {}
    for path, fp, n in found:
        by_machine.setdefault(tuple(fp.get(k) for k in MACHINE_KEYS), []).append((path, fp, n))
    keep, delete = set(), set()
    for machine, group in sorted(by_machine.items()):
        group.sort(key=lambda item: (item[2], item[1].get("git_sha", "")), reverse=True)
        keeper = group[0]
        keep.add(keeper[0])
        for path, fp, n in group[1:]:
            delete.add(path)
        label = "/".join(str(x) for x in machine[:3])
        print(f"machine {label}: keep {keeper[0].name} ({keeper[2]} records, {keeper[1].get('git_sha', '')[:12]})")
        for path, fp, n in group[1:]:
            print(f"  delete {path.name} ({n} records, {fp.get('git_sha', '')[:12]})")
    return keep, delete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("docs/eval/runs"))
    parser.add_argument("--apply", action="store_true", help="actually delete superseded runs")
    args = parser.parse_args()
    found = load_runs(args.runs)
    if not found:
        print("no runs found", file=sys.stderr)
        return 1
    _, delete = pick_keep(found)
    if not args.apply:
        print(f"\ndry run: {len(delete)} file(s) would be deleted; re-run with --apply")
        return 0
    for path in sorted(delete):
        path.unlink()
        print(f"deleted {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
