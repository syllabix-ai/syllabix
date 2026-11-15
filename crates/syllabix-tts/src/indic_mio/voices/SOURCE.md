# Indic-Mio speaker embeddings

Each `*.f32` file is 128 little-endian float32 values: a MioCodec
`global_embedding` extracted offline with `MioCodecModel.encode` on a
cropped (not zero-padded) 5–10 s clip. There is no reference encoder at
runtime.

Rasa (`ai4bharat/Rasa`) is gated. These embeddings are from
[google/fleurs](https://huggingface.co/datasets/google/fleurs)
(CC-BY-4.0) at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
FLEURS `gender` is `0` male / `1` female. Audio is 16 kHz WAV in the
dataset; encode resampled to 24 kHz.

| file | language | gender | FLEURS config | utterance id | clip | duration |
| --- | --- | --- | --- | --- | --- | --- |
| `hi-female.f32` | Hindi | female | `hi_in` | `93` | `10002503286825416621.wav` | 8.64 s |
| `hi-male.f32` | Hindi | male | `hi_in` | `695` | `1005638204747132541.wav` | 7.44 s |
| `ta-female.f32` | Tamil | female | `ta_in` | `878` | `10004352805657441150.wav` | 7.08 s |

Select with `pipeline.tts.voice` (`hi-female`, `hi-male`, `ta-female`).
Default is `hi-female`. An all-zero embedding is not shipped.

Codec: Aratako/MioCodec-25Hz-24kHz @ `3a737f0de2c6324cb2fe40c1fbd1056c7add423d`.
Extracted with `scripts/make-miocodec-istft-fixture.py`.
