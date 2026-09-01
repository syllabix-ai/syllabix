#!/usr/bin/env python3
"""Validate immutable benchmark runs and render the contribution ledger.

The benchmark binary owns measurement.  This tool deliberately only reads
committed JSONL: it makes the submission filename deterministic, rejects a
second submission for the same machine/build fingerprint, and writes views
which CI publishes as an artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


SCHEMA_VERSION = 3
FINGERPRINT_KEYS = {
    "os", "arch", "cpu", "cpu_cores", "ram_bytes", "binary_version", "git_sha"
}
COMMON_KEYS = {"schema_version", "fingerprint", "component", "model", "case", "elapsed_ms", "passed"}
COMPONENT_KEYS = {
    "stt": {
        "first_output_ms", "output_units", "word_match", "word_match_threshold",
        "input_audio_ms", "real_time_factor", "transcript", "memory_before_load_bytes",
        "memory_after_load_bytes", "memory_peak_bytes",
    },
    "tts": {
        "first_output_ms", "output_units", "generated_audio_ms", "real_time_factor",
        "transcript", "memory_before_load_bytes", "memory_after_load_bytes",
        "memory_peak_bytes",
    },
    "llm": {
        "first_output_ms", "output_units", "prompt_tokens", "generated_tokens",
        "prompt_tokens_per_second", "generated_tokens_per_second", "response",
        "memory_before_load_bytes", "memory_after_load_bytes", "memory_peak_bytes",
    },
}
ALL_RECORD_KEYS = COMMON_KEYS | set().union(*COMPONENT_KEYS.values())
CSV_COLUMNS = [
    "fingerprint", "os", "arch", "cpu", "cpu_cores", "ram_bytes", "binary_version",
    "git_sha", "component", "model", "case", "passed", "elapsed_ms",
    "first_output_ms", "real_time_factor", "prompt_tokens_per_second",
    "generated_tokens_per_second", "word_match", "generated_audio_ms",
    "memory_before_load_bytes", "memory_after_load_bytes", "memory_peak_bytes",
]


def fingerprint_id(fingerprint: dict) -> str:
    """Return the stable filename stem for a benchmark fingerprint."""
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def fail(message: str) -> None:
    raise ValueError(message)


def read_run(path: Path) -> tuple[str, list[dict]]:
    records: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        fail(f"{path}: cannot read: {error}")
    if not lines:
        fail(f"{path}: empty run file")
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"{path}:{number}: invalid JSON: {error.msg}")
        if not isinstance(record, dict):
            fail(f"{path}:{number}: record must be an object")
        if record.get("schema_version") != SCHEMA_VERSION:
            fail(f"{path}:{number}: expected schema_version {SCHEMA_VERSION}")
        component = record.get("component")
        if component not in COMPONENT_KEYS:
            fail(f"{path}:{number}: component must be stt, llm, or tts")
        missing = COMMON_KEYS - record.keys()
        if missing:
            fail(f"{path}:{number}: missing common fields: {', '.join(sorted(missing))}")
        unknown = record.keys() - ALL_RECORD_KEYS
        if unknown:
            fail(f"{path}:{number}: unknown fields: {', '.join(sorted(unknown))}")
        fingerprint = record.get("fingerprint")
        if not isinstance(fingerprint, dict) or set(fingerprint) != FINGERPRINT_KEYS:
            fail(f"{path}:{number}: fingerprint fields do not match schema")
        for key in ("os", "arch", "cpu", "binary_version", "git_sha"):
            if not isinstance(fingerprint[key], str) or not fingerprint[key].strip():
                fail(f"{path}:{number}: fingerprint.{key} must be a non-empty string")
        if not isinstance(fingerprint["cpu_cores"], int) or fingerprint["cpu_cores"] < 1:
            fail(f"{path}:{number}: fingerprint.cpu_cores must be a positive integer")
        if fingerprint["ram_bytes"] is not None and (
            not isinstance(fingerprint["ram_bytes"], int) or fingerprint["ram_bytes"] < 1
        ):
            fail(f"{path}:{number}: fingerprint.ram_bytes must be null or a positive integer")
        if not isinstance(record["model"], str) or not record["model"]:
            fail(f"{path}:{number}: model must be a non-empty string")
        if not isinstance(record["case"], str) or not record["case"]:
            fail(f"{path}:{number}: case must be a non-empty string")
        if not isinstance(record["elapsed_ms"], int) or record["elapsed_ms"] < 0:
            fail(f"{path}:{number}: elapsed_ms must be a non-negative integer")
        if not isinstance(record["passed"], bool):
            fail(f"{path}:{number}: passed must be boolean")
        for key in record.keys() & {
            "first_output_ms", "output_units", "prompt_tokens", "generated_tokens",
            "input_audio_ms", "generated_audio_ms", "memory_before_load_bytes",
            "memory_after_load_bytes", "memory_peak_bytes",
        }:
            if not isinstance(record[key], int) or record[key] < 0:
                fail(f"{path}:{number}: {key} must be a non-negative integer")
        for key in record.keys() & {
            "prompt_tokens_per_second", "generated_tokens_per_second", "word_match",
            "word_match_threshold", "real_time_factor",
        }:
            if not isinstance(record[key], (int, float)) or isinstance(record[key], bool):
                fail(f"{path}:{number}: {key} must be numeric")
        records.append(record)
    fingerprints = {fingerprint_id(record["fingerprint"]) for record in records}
    if len(fingerprints) != 1:
        fail(f"{path}: every record must have the same fingerprint")
    return fingerprints.pop(), records


def collect(runs: Path) -> list[tuple[str, Path, list[dict]]]:
    if not runs.is_dir():
        return []
    found: list[tuple[str, Path, list[dict]]] = []
    seen: set[str] = set()
    for path in sorted(runs.glob("*.jsonl")):
        run_id, records = read_run(path)
        if path.stem != run_id:
            fail(f"{path}: filename must be {run_id}.jsonl")
        if run_id in seen:
            fail(f"{path}: duplicate fingerprint {run_id}")
        seen.add(run_id)
        found.append((run_id, path, records))
    return found


def render(found: list[tuple[str, Path, list[dict]]], csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for run_id, _, records in found:
            for record in records:
                row = {column: record.get(column, "") for column in CSV_COLUMNS}
                row["fingerprint"] = run_id
                row.update(record["fingerprint"])
                writer.writerow(row)

    lines = [
        "# Performance contribution ledger",
        "",
        "Generated by `scripts/validate-performance-ledger.py`; do not edit by hand.",
        "",
        "| Fingerprint | OS / architecture | Binary | Records | Passed |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for run_id, _, records in found:
        fingerprint = records[0]["fingerprint"]
        passed = sum(record["passed"] for record in records)
        lines.append(
            f"| `{run_id}` | {fingerprint['os']} / {fingerprint['arch']} | "
            f"{fingerprint['binary_version']} ({fingerprint['git_sha']}) | {len(records)} | {passed} |"
        )
    if not found:
        lines.append("| — | — | — | 0 | 0 |")
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("docs/eval/runs"))
    parser.add_argument("--fingerprint", type=Path, help="print the fingerprint id for one JSONL file")
    parser.add_argument("--csv", type=Path, default=Path("docs/eval/results.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("docs/eval/results.md"))
    args = parser.parse_args()
    try:
        if args.fingerprint:
            run_id, _ = read_run(args.fingerprint)
            print(run_id)
            return 0
        render(collect(args.runs), args.csv, args.markdown)
    except ValueError as error:
        print(f"ledger validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
