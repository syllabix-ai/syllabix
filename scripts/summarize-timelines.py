#!/usr/bin/env python3
"""Summarize row-34 turn-timeline sidecars into per-segment p50/p95.

Contributor-only tool for the reference-Air capture protocol in
docs/reference-profiles.md §10. It is never shipped in the binary and users
never need it.

Usage:
    python3 scripts/summarize-timelines.py [TURN_DIR]

TURN_DIR defaults to target/turn-debug. Reads turn-*/turn.json files that the
`diagnostics:` yaml block produced and prints one table of p50/p95 per named
timeline segment (milliseconds), plus a raw-anchor table.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# (name, start anchor, end anchor) — signed differences allowed so
# llm_end_to_first_pcm can expose TTS/generation overlap vs whole-reply buffering.
SEGMENTS = [
    ("user_speech_ms", "speech_start_ms", "speech_end_ms"),
    ("vad_queue_wait_ms", "speech_end_ms", "stt_queued_ms"),
    ("stt_decode_ms", "stt_queued_ms", "stt_done_ms"),
    ("stt_total_ms", "speech_end_ms", "stt_done_ms"),
    ("handoff_llm_ms", "stt_done_ms", "llm_start_ms"),
    ("llm_ttft_ms", "llm_start_ms", "llm_first_token_ms"),
    ("llm_stream_ms", "llm_first_token_ms", "llm_last_token_ms"),
    ("tts_lead_ms", "llm_first_token_ms", "tts_first_pcm_ms"),
    ("llm_end_to_first_pcm_ms", "llm_last_token_ms", "tts_first_pcm_ms"),
    ("tts_synthesis_ms", "tts_first_pcm_ms", "tts_last_pcm_ms"),
    ("playback_handoff_ms", "tts_first_pcm_ms", "playback_first_ms"),
    ("audible_latency_ms", "speech_end_ms", "playback_first_ms"),
    ("spoken_duration_ms", "playback_first_ms", "playback_done_ms"),
    ("total_turn_ms", "speech_end_ms", "playback_done_ms"),
]


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile; matches the n=20 convention in §9."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q / 100 * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def load_sidesars(root: Path) -> list[dict]:
    turns = sorted(root.glob("turn-*/turn.json"))
    if not turns:
        sys.exit(f"no turn-*/turn.json under {root} — enable diagnostics first")
    docs = []
    for path in turns:
        try:
            docs.append(json.loads(path.read_text()))
        except json.JSONDecodeError as err:
            print(f"warning: skipping {path}: {err}", file=sys.stderr)
    return docs


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "target/turn-debug")
    docs = load_sidesars(root)
    completed = sum(1 for d in docs if d.get("outcome") == "completed")

    print(f"turns: {len(docs)} read from {root} ({completed} completed, "
          f"{len(docs) - completed} cancelled/skipped)\n")

    header = f"{'segment':<26}{'n':>4}{'p50':>10}{'p95':>10}{'min':>8}{'max':>8}"
    print(header)
    print("-" * len(header))
    for name, start_key, end_key in SEGMENTS:
        values = [
            d["timeline"][end_key] - d["timeline"][start_key]
            for d in docs
            if d.get("timeline", {}).get(start_key) is not None
            and d.get("timeline", {}).get(end_key) is not None
        ]
        if not values:
            print(f"{name:<26}{0:>4}{'—':>10}{'—':>10}{'—':>8}{'—':>8}")
            continue
        p50 = percentile(values, 50)
        p95 = percentile(values, 95)
        print(f"{name:<26}{len(values):>4}{p50:>10.0f}{p95:>10.0f}"
              f"{min(values):>8.0f}{max(values):>8.0f}")

    print("\nraw anchors (ms from speech_start):")
    anchors = [
        key.removesuffix("_ms")
        for key in docs[0].get("timeline", {})
        if key.endswith("_ms")
    ]
    print(f"{'anchor':<26}{'n':>4}{'p50':>10}{'p95':>10}")
    print("-" * 50)
    for anchor in anchors:
        values = [
            d["timeline"][f"{anchor}_ms"]
            for d in docs
            if d.get("timeline", {}).get(f"{anchor}_ms") is not None
        ]
        if not values:
            continue
        print(f"{anchor + '_ms':<26}{len(values):>4}"
              f"{percentile(values, 50):>10.0f}{percentile(values, 95):>10.0f}")


if __name__ == "__main__":
    main()
