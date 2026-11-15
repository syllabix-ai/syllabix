#!/usr/bin/env python3
"""Tooling-only: capture MioCodec ISTFT golden files and speaker embeddings.

Pinned codec: Aratako/MioCodec-25Hz-24kHz @ 3a737f0de2c6324cb2fe40c1fbd1056c7add423d
Usage (from repo root, CPU torch):

  pip install 'torch' --index-url https://download.pytorch.org/whl/cpu
  pip install 'git+https://github.com/Aratako/MioCodec@3a737f0de2c6324cb2fe40c1fbd1056c7add423d'
  python3 scripts/make-miocodec-istft-fixture.py
"""

from __future__ import annotations

import argparse
import struct
import sys
import os
from pathlib import Path

import numpy as np

N_FFT = 1920
HOP = 480
N_FREQ = N_FFT // 2 + 1
SAME_TRIM = (N_FFT - HOP) // 2
SAMPLES_PER_TOKEN = 960
# Short fixed sequence used by the Rust golden test.
TOKENS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212]


def write_f32(path: Path, values: np.ndarray) -> None:
    path.write_bytes(np.asarray(values, dtype=np.float32).tobytes())


def periodic_hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def numpy_istft(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """mag/phase: [n_freq, n_frames]. Matches miocodec ISTFT padding='same'."""
    win = periodic_hann(N_FFT)
    spec = mag * np.exp(1j * phase)
    frames = np.fft.irfft(spec, n=N_FFT, axis=0, norm="backward") * win[:, None]
    n_frames = frames.shape[1]
    length = (n_frames - 1) * HOP + N_FFT
    y = np.zeros(length, dtype=np.float64)
    env = np.zeros(length, dtype=np.float64)
    for i in range(n_frames):
        y[i * HOP : i * HOP + N_FFT] += frames[:, i]
        env[i * HOP : i * HOP + N_FFT] += win.astype(np.float64) ** 2
    y = y[SAME_TRIM : length - SAME_TRIM]
    env = env[SAME_TRIM : length - SAME_TRIM]
    return (y / np.maximum(env, 1e-11)).astype(np.float32)


def capture_upstream(tokens, embedding, out_dir: Path) -> None:
    import torch
    from miocodec.module.istft_head import ISTFT

    captured = {}

    def hook(_mod, inputs, output):
        spec = inputs[0].detach().cpu()
        captured["spec"] = spec
        captured["pcm"] = output.detach().cpu()

    from miocodec import MioCodecModel

    model = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-24kHz", revision="3a737f0de2c6324cb2fe40c1fbd1056c7add423d"
    ).eval()
    head = model.decoder.istft_head if hasattr(model, "decoder") else None
    istft = None
    for module in model.modules():
        if isinstance(module, ISTFT):
            istft = module
            break
    if istft is None:
        raise SystemExit("could not find ISTFT module on MioCodecModel")
    handle = istft.register_forward_hook(hook)
    codes = torch.tensor(tokens, dtype=torch.long)
    emb = torch.tensor(embedding, dtype=torch.float32)
    with torch.no_grad():
        wav = model.decode(global_embedding=emb, content_token_indices=codes)
    handle.remove()
    spec = captured["spec"]
    if spec.ndim == 3:
        # [B, freq, time] or [B, time, freq]
        spec = spec[0]
    if spec.shape[0] != N_FREQ and spec.shape[-1] == N_FREQ:
        spec = spec.transpose(0, 1)
    mag = spec.abs().numpy().astype(np.float32)
    phase = spec.angle().numpy().astype(np.float32)
    pcm = captured["pcm"].squeeze().numpy().astype(np.float32)
    n_tokens = len(tokens)
    pcm = pcm[: n_tokens * SAMPLES_PER_TOKEN]
    write_f32(out_dir / "spec_mag.f32", mag.reshape(-1))
    write_f32(out_dir / "spec_phase.f32", phase.reshape(-1))
    write_f32(out_dir / "expected_pcm.f32", pcm)
    (out_dir / "tokens.txt").write_text(" ".join(str(t) for t in tokens) + "\n")
    print("upstream wav", tuple(wav.squeeze().shape), "peak", float(np.abs(pcm).max()))
    print("wrote", out_dir)


def capture_onnx_decoder(tokens, embedding, onnx_path: Path, out_dir: Path) -> None:
    """Pad with the last token, run the pinned decoder ONNX, iSTFT, trim to T×960."""
    import onnxruntime as ort

    pad = list(tokens) + [tokens[-1]] * (100 - len(tokens))
    ids = np.asarray(pad, dtype=np.int64).reshape(1, 100)
    emb = np.asarray(embedding, dtype=np.float32).reshape(1, 128)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    mag, phase = sess.run(None, {"content_token_indices": ids, "global_embedding": emb})[:2]
    mag = np.asarray(mag, dtype=np.float32).reshape(N_FREQ, 200)
    phase = np.asarray(phase, dtype=np.float32).reshape(N_FREQ, 200)
    pcm = numpy_istft(mag, phase)[: len(tokens) * SAMPLES_PER_TOKEN]
    write_f32(out_dir / "decoder_pcm.f32", pcm)
    print("onnx decoder pcm", pcm.shape, "peak", float(np.abs(pcm).max()))


def extract_voice(wav_path: Path, out_path: Path) -> np.ndarray:
    import torch
    from miocodec import MioCodecModel, load_audio

    model = MioCodecModel.from_pretrained(
        "Aratako/MioCodec-25Hz-24kHz", revision="3a737f0de2c6324cb2fe40c1fbd1056c7add423d"
    ).eval()
    wav = load_audio(str(wav_path), sample_rate=24000)
    # Crop rather than zero-pad: keep 5–10 s.
    max_samples = 10 * 24000
    min_samples = 5 * 24000
    n = wav.shape[-1]
    if n > max_samples:
        wav = wav[..., :max_samples]
    elif n < min_samples:
        print(f"warning: {wav_path} is only {n / 24000:.2f}s", file=sys.stderr)
    with torch.no_grad():
        feats = model.encode(wav)
        emb = feats.global_embedding.squeeze().cpu().numpy().astype(np.float32)
    write_f32(out_path, emb)
    print(out_path, "norm", float(np.linalg.norm(emb)))
    return emb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("crates/syllabix-native/tests/fixtures/miocodec_istft"))
    parser.add_argument("--voice", type=Path, action="append", default=[])
    parser.add_argument("--voice-out", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    embedding = np.zeros(128, dtype=np.float32)
    if args.voice:
        embedding = extract_voice(args.voice[0], args.out / "voice.f32")
        for wav, dest in zip(args.voice, args.voice_out):
            extract_voice(Path(wav), Path(dest))
    else:
        write_f32(args.out / "voice.f32", embedding)

    try:
        capture_upstream(TOKENS, embedding, args.out)
    except Exception as exc:
        raise SystemExit(
            f"upstream ISTFTHead capture failed (needed for the CI golden): {exc}"
        ) from exc

    onnx_path = Path(os.environ.get("MIOCODEC_DECODER_ONNX", ""))
    if onnx_path.is_file():
        capture_onnx_decoder(TOKENS, embedding, onnx_path, args.out)

    (args.out / "SOURCE.txt").write_text(
        "MioCodec ISTFT golden fixture\n"
        "codec: Aratako/MioCodec-25Hz-24kHz @ 3a737f0de2c6324cb2fe40c1fbd1056c7add423d\n"
        f"tokens: {' '.join(str(t) for t in TOKENS)}\n"
        "command: python3 scripts/make-miocodec-istft-fixture.py\n"
        "layout: spec_mag.f32 / spec_phase.f32 are little-endian f32, row-major [n_freq=961, n_frames]\n"
        "spec_* and expected_pcm.f32 come from an ISTFTHead forward hook.\n"
        "decoder_pcm.f32 is the pinned ONNX body + numpy iSTFT of last-token-padded codes, truncated to T×960.\n"
    )


if __name__ == "__main__":
    main()
