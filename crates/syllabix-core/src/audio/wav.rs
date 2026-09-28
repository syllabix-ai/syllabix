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
    if !data.len().is_multiple_of(2) {
        return Err(Error::InvalidAudio {
            message: "WAV data chunk is not 16-bit aligned".into(),
        });
    }
    Ok(data
        .as_chunks::<2>()
        .0
        .iter()
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

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    fn fmt_body(audio_format: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&audio_format.to_le_bytes());
        body.extend_from_slice(&channels.to_le_bytes());
        body.extend_from_slice(&rate.to_le_bytes());
        body.extend_from_slice(&(rate * u32::from(channels) * 2).to_le_bytes());
        body.extend_from_slice(&(channels * 2).to_le_bytes());
        body.extend_from_slice(&bits.to_le_bytes());
        body
    }

    fn riff(chunks: &[u8]) -> Vec<u8> {
        let mut body = chunks.to_vec();
        if body.len() < 32 {
            body.resize(32, 0);
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(4 + body.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn rejects_chunk_overrun() {
        let mut file = Vec::from(*b"data");
        file.extend_from_slice(&1000u32.to_le_bytes());
        file.extend_from_slice(&[1, 2, 3, 4]);
        let err = read_wav(Cursor::new(riff(&file))).unwrap_err();
        assert_eq!(err.to_string(), "invalid audio: WAV chunk overruns file");
    }

    #[test]
    fn rejects_missing_fmt_chunk() {
        let data = chunk(b"data", &[0, 0]);
        let err = read_wav(Cursor::new(riff(&data))).unwrap_err();
        assert_eq!(err.to_string(), "invalid audio: WAV missing fmt chunk");
    }

    #[test]
    fn rejects_missing_data_chunk() {
        let fmt = chunk(b"fmt ", &fmt_body(1, 1, 16_000, 16));
        let err = read_wav(Cursor::new(riff(&fmt))).unwrap_err();
        assert_eq!(err.to_string(), "invalid audio: WAV missing data chunk");
    }

    #[test]
    fn rejects_short_fmt_chunk() {
        let fmt = chunk(b"fmt ", &[1, 0]);
        let data = chunk(b"data", &[0, 0]);
        let mut chunks = fmt;
        chunks.extend_from_slice(&data);
        let file = riff(&chunks);
        let err = read_wav(Cursor::new(file)).unwrap_err();
        assert_eq!(err.to_string(), "invalid audio: WAV fmt chunk too short");
    }

    #[test]
    fn rejects_non_pcm_encoding() {
        let fmt = chunk(b"fmt ", &fmt_body(3, 1, 16_000, 16));
        let data = chunk(b"data", &[0, 0]);
        let mut chunks = fmt;
        chunks.extend_from_slice(&data);
        let file = riff(&chunks);
        let err = read_wav(Cursor::new(file)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid audio: WAV audio format 3 is not PCM"
        );
    }

    #[test]
    fn rejects_non_16_bit_samples() {
        let fmt = chunk(b"fmt ", &fmt_body(1, 1, 16_000, 8));
        let data = chunk(b"data", &[0]);
        let mut chunks = fmt;
        chunks.extend_from_slice(&data);
        let file = riff(&chunks);
        let err = read_wav(Cursor::new(file)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid audio: WAV bits-per-sample 8 is not 16"
        );
    }

    #[test]
    fn rejects_unaligned_data_chunk() {
        // `data` with an odd body forces the pad-skip path and the alignment
        // error together: size 3 pads to 4 bytes on disk.
        let fmt = chunk(b"fmt ", &fmt_body(1, 1, 16_000, 16));
        let mut chunks = fmt;
        chunks.extend_from_slice(b"data");
        chunks.extend_from_slice(&3u32.to_le_bytes());
        chunks.extend_from_slice(&[1, 2, 3, 0]);
        let file = riff(&chunks);
        let err = read_wav(Cursor::new(file)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid audio: WAV data chunk is not 16-bit aligned"
        );
    }

    #[test]
    fn skips_unknown_chunks() {
        let junk = chunk(b"JUNK", &[9, 9, 9, 9]);
        let fmt = chunk(b"fmt ", &fmt_body(1, 1, 16_000, 16));
        let data = chunk(b"data", &42i16.to_le_bytes());
        let mut chunks = junk;
        chunks.extend_from_slice(&fmt);
        chunks.extend_from_slice(&data);
        let file = riff(&chunks);
        let wav = read_wav(Cursor::new(file)).unwrap();
        assert_eq!(wav.samples, vec![42]);
    }
}
