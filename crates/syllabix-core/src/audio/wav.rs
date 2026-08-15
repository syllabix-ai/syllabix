//! PCM16 WAV read/write for fixtures. No compressed encodings.

use std::io::{Read, Write};

use crate::audio::convert::PcmFormat;
use crate::error::{Error, Result};

/// Decoded PCM16 WAV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavPcm {
    /// File sample rate and channel count.
    pub format: PcmFormat,
    /// Interleaved PCM16 samples.
    pub samples: Vec<i16>,
}

/// Write a PCM16 WAV (canonical 44-byte header + data).
pub fn write_wav<W: Write>(mut w: W, wav: &WavPcm) -> Result<()> {
    let channels = wav.format.channels;
    let rate = wav.format.sample_rate_hz;
    let data_bytes = wav.samples.len() * 2;
    let block_align = channels * 2;
    let byte_rate = rate * u32::from(block_align);
    let riff_size = 36 + data_bytes as u32;

    w.write_all(b"RIFF")?;
    w.write_all(&riff_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?; // PCM
    w.write_all(&channels.to_le_bytes())?;
    w.write_all(&rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&16u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&(data_bytes as u32).to_le_bytes())?;
    for s in &wav.samples {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// Read a PCM16 WAV. Other encodings are rejected with [`Error::InvalidAudio`].
pub fn read_wav<R: Read>(mut r: R) -> Result<WavPcm> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    if buf.len() < 44 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err(Error::InvalidAudio {
            message: "fixture is not a PCM WAV file".into(),
        });
    }
    let mut pos = 12usize;
    let mut format = None;
    let mut data = None;
    while pos + 8 <= buf.len() {
        let id = &buf[pos..pos + 4];
        let size = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + size > buf.len() {
            return Err(Error::InvalidAudio {
                message: "WAV chunk overruns file".into(),
            });
        }
        let chunk = &buf[pos..pos + size];
        if id == b"fmt " {
            format = Some(parse_fmt(chunk)?);
        } else if id == b"data" {
            data = Some(parse_pcm16(chunk)?);
        }
        pos += size + size % 2;
    }
    let (fmt, _) = format.ok_or_else(|| Error::InvalidAudio {
        message: "WAV missing fmt chunk".into(),
    })?;
    let samples = data.ok_or_else(|| Error::InvalidAudio {
        message: "WAV missing data chunk".into(),
    })?;
    Ok(WavPcm {
        format: fmt,
        samples,
    })
}

fn parse_fmt(chunk: &[u8]) -> Result<(PcmFormat, u16)> {
    if chunk.len() < 16 {
        return Err(Error::InvalidAudio {
            message: "WAV fmt chunk too short".into(),
        });
    }
    let audio_format = u16::from_le_bytes(chunk[0..2].try_into().unwrap());
    if audio_format != 1 {
        return Err(Error::InvalidAudio {
            message: format!("WAV audio format {audio_format} is not PCM"),
        });
    }
    let channels = u16::from_le_bytes(chunk[2..4].try_into().unwrap());
    let rate = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
    let bits = u16::from_le_bytes(chunk[14..16].try_into().unwrap());
    if bits != 16 {
        return Err(Error::InvalidAudio {
            message: format!("WAV bits-per-sample {bits} is not 16"),
        });
    }
    Ok((
        PcmFormat {
            sample_rate_hz: rate,
            channels,
        },
        bits,
    ))
}

fn parse_pcm16(data: &[u8]) -> Result<Vec<i16>> {
    if data.len() % 2 != 0 {
        return Err(Error::InvalidAudio {
            message: "WAV data chunk is not 16-bit aligned".into(),
        });
    }
    Ok(data
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn wav_roundtrip_preserves_pcm() {
        let wav = WavPcm {
            format: PcmFormat {
                sample_rate_hz: 16_000,
                channels: 1,
            },
            samples: vec![0, 1, -2, 32767, -32768],
        };
        let mut buf = Vec::new();
        write_wav(&mut buf, &wav).unwrap();
        let back = read_wav(Cursor::new(buf)).unwrap();
        assert_eq!(back, wav);
    }

    #[test]
    fn rejects_non_wav() {
        let err = read_wav(Cursor::new(b"not a wav")).unwrap_err();
        assert!(matches!(err, Error::InvalidAudio { .. }));
    }
}
