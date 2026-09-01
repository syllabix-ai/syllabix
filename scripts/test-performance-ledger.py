#!/usr/bin/env python3
"""Fast schema/renderer checks; benchmark weights are never loaded."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "scripts" / "validate-performance-ledger.py"
FINGERPRINT = {"os": "linux", "arch": "x86_64", "cpu": "Example CPU", "cpu_cores": 4,
               "ram_bytes": 8_000_000_000, "binary_version": "0.1.0", "git_sha": "abc123"}


def record():
    return {"schema_version": 3, "fingerprint": FINGERPRINT, "component": "stt", "model": "small",
            "case": "fixture", "elapsed_ms": 12, "first_output_ms": 12, "output_units": 5,
            "word_match": 1.0, "word_match_threshold": 0.8, "input_audio_ms": 100,
            "real_time_factor": 0.12, "transcript": "hello", "memory_before_load_bytes": 1,
            "memory_after_load_bytes": 2, "memory_peak_bytes": 3, "passed": True}


class LedgerTests(unittest.TestCase):
    def test_renders_valid_fingerprinted_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            raw = temporary / "raw.jsonl"
            raw.write_text(json.dumps(record()) + "\n")
            fingerprint = subprocess.check_output([sys.executable, str(VALIDATOR), "--fingerprint", str(raw)], text=True).strip()
            runs = temporary / "runs"
            runs.mkdir()
            (runs / f"{fingerprint}.jsonl").write_text(raw.read_text())
            result = subprocess.run([sys.executable, str(VALIDATOR), "--runs", str(runs), "--csv", str(temporary / "results.csv"), "--markdown", str(temporary / "results.md")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("fingerprint,os,arch,cpu", (temporary / "results.csv").read_text())
            self.assertIn(fingerprint, (temporary / "results.md").read_text())

    def test_rejects_filename_and_schema_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            runs = temporary / "runs"
            runs.mkdir()
            bad = record()
            bad["surprise"] = True
            (runs / "not-a-fingerprint.jsonl").write_text(json.dumps(bad) + "\n")
            result = subprocess.run([sys.executable, str(VALIDATOR), "--runs", str(runs)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown fields", result.stderr)


if __name__ == "__main__":
    unittest.main()
