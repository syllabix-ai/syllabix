#!/usr/bin/env python3
"""Build a local 16 kHz mono WAV+TSV corpus for `bench-asr-worker --manifest`.

Default source is LibriSpeech test-clean from OpenSLR (CC-BY-4.0): the first
200 utterances sorted by id. Extra TSV files (`wav_path<TAB>reference`) can be
merged with --extra-tsv; those paths are relative to the TSV. Do not add a
dataset here unless its license is already verified as permissive.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

OPENSIR_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
UTTERANCE_LIMIT = 200


def convert_audio(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dest),
        ]
    else:
        sox = shutil.which("sox")
        if not sox:
            raise SystemExit("ffmpeg or sox is required to convert audio to WAV")
        cmd = [sox, str(src), "-r", "16000", "-c", "1", "-b", "16", str(dest)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def collect_utterances(root: Path) -> list[tuple[str, Path, str]]:
    items: list[tuple[str, Path, str]] = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            flac = trans.with_name(f"{utt_id}.flac")
            if not flac.is_file():
                raise SystemExit(f"missing FLAC for {utt_id}: {flac}")
            items.append((utt_id, flac, text.strip()))
    items.sort(key=lambda item: item[0])
    return items[:UTTERANCE_LIMIT]


def download_archive(archive: Path) -> None:
    if archive.is_file():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {OPENSIR_URL}", file=sys.stderr)
    urllib.request.urlretrieve(OPENSIR_URL, archive)


def extract_archive(archive: Path, dest: Path) -> Path:
    extracted = dest / "LibriSpeech" / "test-clean"
    if extracted.is_dir():
        return extracted
    print(f"extracting {archive}", file=sys.stderr)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    if not extracted.is_dir():
        raise SystemExit(f"expected {extracted} after extract")
    return extracted


def parse_extra_tsv(path: Path) -> list[tuple[Path, str, str]]:
    items = []
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "\t" not in line:
            raise SystemExit(f"{path}:{index} is not wav_path<TAB>reference text")
        wav_s, rest = line.split("\t", 1)
        if "\t" in rest:
            text, dataset = rest.split("\t", 1)
            dataset = dataset.strip() or "extra"
        else:
            text, dataset = rest, "extra"
        wav = Path(wav_s.strip())
        if not wav.is_absolute():
            wav = path.parent / wav
        if not wav.is_file():
            raise SystemExit(f"{path}:{index} missing audio {wav}")
        items.append((wav, text.strip(), dataset))
    return items


def unique_wav_name(used: set[str], src: Path) -> str:
    name = src.name if src.suffix.lower() == ".wav" else f"{src.stem}.wav"
    if name not in used:
        return name
    stem = Path(name).stem
    n = 2
    while True:
        candidate = f"{stem}-{n}.wav"
        if candidate not in used:
            return candidate
        n += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="output folder for WAVs and manifest.tsv")
    parser.add_argument(
        "--extra-tsv",
        action="append",
        default=[],
        type=Path,
        help="additional TSV corpus to merge (paths relative to that TSV)",
    )
    args = parser.parse_args()
    out = args.folder
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "test-clean.tar.gz"
    download_archive(archive)
    speech = extract_archive(archive, out)
    utterances = collect_utterances(speech)
    if len(utterances) < UTTERANCE_LIMIT:
        raise SystemExit(f"found {len(utterances)} utterances, need {UTTERANCE_LIMIT}")
    wav_dir = out / "wav"
    used: set[str] = set()
    rows: list[tuple[str, str, str]] = []
    for utt_id, flac, text in utterances:
        name = f"{utt_id}.wav"
        wav = wav_dir / name
        if not wav.is_file():
            convert_audio(flac, wav)
        used.add(name)
        rows.append((name, text, "librispeech-test-clean"))
    for extra in args.extra_tsv:
        for src, text, dataset in parse_extra_tsv(extra):
            name = unique_wav_name(used, src)
            dest = wav_dir / name
            if not dest.is_file():
                convert_audio(src, dest)
            used.add(name)
            rows.append((name, text, dataset))
    manifest = out / "manifest.tsv"
    manifest.write_text(
        "".join(f"wav/{name}\t{text}\t{dataset}\n" for name, text, dataset in rows),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} utterances to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
