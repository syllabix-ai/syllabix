//! 24 kHz mono Ogg/Opus pages for Moshi audio messages (`s2s-opus`).

use std::collections::VecDeque;
use std::io::Write;

use ogg::{PacketWriteEndInfo, PacketWriter};

use super::{framing_err, MOSHI_CHANNELS, MOSHI_SAMPLE_RATE_HZ};
use crate::error::{Error, Result};

/// Ogg stream serial used for Moshi audio (same constant as the Moshi CLI).
pub const OGG_SERIAL: u32 = 42;
/// 40 ms of 24 kHz mono (Opus frame size Moshi encodes).
pub const OPUS_FRAME_SAMPLES: usize = 960;

/// Encodes 24 kHz mono PCM16 into Ogg pages of Opus packets.
pub struct OggOpusEncoder {
    encoder: opus::Encoder,
    writer: PacketWriter<'static, Vec<u8>>,
    pending: VecDeque<i16>,
    total_samples: u64,
    encode_buf: Vec<u8>,
}

impl OggOpusEncoder {
    /// New stream with OpusHead and OpusTags pages queued.
    pub fn new() -> Result<Self> {
        let encoder = opus::Encoder::new(
            MOSHI_SAMPLE_RATE_HZ,
            opus::Channels::Mono,
            opus::Application::Voip,
        )
        .map_err(opus_err)?;
        let mut writer = PacketWriter::new(Vec::new());
        let mut head = Vec::new();
        write_opus_head(&mut head).map_err(ogg_err)?;
        writer
            .write_packet(head, OGG_SERIAL, PacketWriteEndInfo::EndPage, 0)
            .map_err(ogg_err)?;
        let mut tags = Vec::new();
        write_opus_tags(&mut tags).map_err(ogg_err)?;
        writer
            .write_packet(tags, OGG_SERIAL, PacketWriteEndInfo::EndPage, 0)
            .map_err(ogg_err)?;
        Ok(Self {
            encoder,
            writer,
            pending: VecDeque::new(),
            total_samples: 0,
            encode_buf: vec![0u8; 4096],
        })
    }

    /// Encode 24 kHz mono PCM16. Returns any complete Ogg pages, including
    /// identification headers on the first call.
    pub fn encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>> {
        self.pending.extend(pcm.iter().copied());
        while self.pending.len() >= OPUS_FRAME_SAMPLES {
            let frame: Vec<i16> = self.pending.drain(..OPUS_FRAME_SAMPLES).collect();
            self.total_samples += OPUS_FRAME_SAMPLES as u64;
            let size = self
                .encoder
                .encode(&frame, &mut self.encode_buf)
                .map_err(opus_err)?;
            if size > 0 {
                self.writer
                    .write_packet(
                        self.encode_buf[..size].to_vec(),
                        OGG_SERIAL,
                        PacketWriteEndInfo::EndPage,
                        self.total_samples,
                    )
                    .map_err(ogg_err)?;
            }
        }
        Ok(take_pages(&mut self.writer))
    }
}

/// Decodes Ogg/Opus pages (possibly split across websocket messages) to PCM16.
pub struct OggOpusDecoder {
    decoder: opus::Decoder,
    leftover: Vec<u8>,
    pending_packet: Vec<u8>,
}

impl OggOpusDecoder {
    /// 24 kHz mono decoder.
    pub fn new() -> Result<Self> {
        let decoder =
            opus::Decoder::new(MOSHI_SAMPLE_RATE_HZ, opus::Channels::Mono).map_err(opus_err)?;
        Ok(Self {
            decoder,
            leftover: Vec::new(),
            pending_packet: Vec::new(),
        })
    }

    /// Push Ogg bytes and return any decoded 24 kHz mono PCM16.
    pub fn decode(&mut self, ogg: &[u8]) -> Result<Vec<i16>> {
        self.leftover.extend_from_slice(ogg);
        let packets = drain_packets(&mut self.leftover, &mut self.pending_packet)?;
        let mut pcm = Vec::new();
        let mut buf = vec![0i16; MOSHI_SAMPLE_RATE_HZ as usize];
        for packet in packets {
            if packet.starts_with(b"OpusHead") || packet.starts_with(b"OpusTags") {
                continue;
            }
            let n = self
                .decoder
                .decode(&packet, &mut buf, false)
                .map_err(opus_err)?;
            pcm.extend_from_slice(&buf[..n]);
        }
        Ok(pcm)
    }
}

fn write_opus_head(w: &mut Vec<u8>) -> std::io::Result<()> {
    // https://wiki.xiph.org/OggOpus#ID_Header
    w.write_all(b"OpusHead")?;
    w.write_all(&[1u8])?; // version
    w.write_all(&[MOSHI_CHANNELS as u8])?;
    w.write_all(&3840u16.to_le_bytes())?; // pre-skip
    w.write_all(&48_000u32.to_le_bytes())?; // input sample rate hint
    w.write_all(&0i16.to_le_bytes())?; // output gain
    w.write_all(&[0u8])?; // RTP channel map
    Ok(())
}

fn write_opus_tags(w: &mut Vec<u8>) -> std::io::Result<()> {
    let vendor = b"Syllabix";
    w.write_all(b"OpusTags")?;
    w.write_all(&(vendor.len() as u32).to_le_bytes())?;
    w.write_all(vendor)?;
    w.write_all(&0u32.to_le_bytes())?;
    Ok(())
}

fn take_pages(writer: &mut PacketWriter<'static, Vec<u8>>) -> Vec<u8> {
    std::mem::take(writer.inner_mut())
}

struct OggPage {
    continued: bool,
    segments: Vec<u8>,
    body: Vec<u8>,
}

fn drain_packets(buf: &mut Vec<u8>, pending: &mut Vec<u8>) -> Result<Vec<Vec<u8>>> {
    let mut packets = Vec::new();
    loop {
        match next_page(buf)? {
            None => break,
            Some((page, consumed)) => {
                buf.drain(..consumed);
                assemble_packets(&mut packets, pending, page);
            }
        }
    }
    Ok(packets)
}

fn next_page(buf: &[u8]) -> Result<Option<(OggPage, usize)>> {
    if buf.len() < 4 {
        return Ok(None);
    }
    if &buf[..4] != b"OggS" {
        return Err(framing_err("Ogg page does not start with OggS"));
    }
    if buf.len() < 27 {
        return Ok(None);
    }
    if buf[4] != 0 {
        return Err(framing_err(format!("unsupported Ogg version {}", buf[4])));
    }
    let header_type = buf[5];
    let nsegs = buf[26] as usize;
    let table_end = 27 + nsegs;
    if buf.len() < table_end {
        return Ok(None);
    }
    let segments = buf[27..table_end].to_vec();
    let body_len: usize = segments.iter().map(|&s| s as usize).sum();
    let total = table_end + body_len;
    if buf.len() < total {
        return Ok(None);
    }
    Ok(Some((
        OggPage {
            continued: header_type & 0x01 != 0,
            segments,
            body: buf[table_end..total].to_vec(),
        },
        total,
    )))
}

fn assemble_packets(out: &mut Vec<Vec<u8>>, pending: &mut Vec<u8>, page: OggPage) {
    if !page.continued {
        pending.clear();
    }
    let mut offset = 0usize;
    for seg in page.segments {
        let n = seg as usize;
        pending.extend_from_slice(&page.body[offset..offset + n]);
        offset += n;
        if seg < 255 {
            out.push(std::mem::take(pending));
        }
    }
}

fn opus_err(err: opus::Error) -> Error {
    Error::Provider {
        provider: "opus",
        message: err.to_string(),
    }
}

fn ogg_err(err: std::io::Error) -> Error {
    Error::Provider {
        provider: "ogg",
        message: err.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s2s::codec::MoshiMessage;

    fn sine(samples: usize) -> Vec<i16> {
        (0..samples)
            .map(|i| {
                let t = i as f64 / f64::from(MOSHI_SAMPLE_RATE_HZ);
                (t * 440.0 * std::f64::consts::TAU).sin() * 16_000.0
            })
            .map(|v| v.round() as i16)
            .collect()
    }

    fn corr(a: &[i16], b: &[i16]) -> f64 {
        let n = a.len().min(b.len());
        let a = &a[..n];
        let b = &b[..n];
        let mean_a = a.iter().map(|s| f64::from(*s)).sum::<f64>() / n as f64;
        let mean_b = b.iter().map(|s| f64::from(*s)).sum::<f64>() / n as f64;
        let mut num = 0.0;
        let mut da = 0.0;
        let mut db = 0.0;
        for i in 0..n {
            let x = f64::from(a[i]) - mean_a;
            let y = f64::from(b[i]) - mean_b;
            num += x * y;
            da += x * x;
            db += y * y;
        }
        num / (da.sqrt() * db.sqrt())
    }

    #[test]
    fn headers_then_pcm_roundtrip() {
        let mut enc = OggOpusEncoder::new().expect("encoder");
        let headers = enc.encode(&[]).expect("headers");
        assert!(headers.starts_with(b"OggS"), "{headers:?}");
        assert!(headers.windows(8).any(|w| w == b"OpusHead"));
        assert!(headers.windows(8).any(|w| w == b"OpusTags"));
        assert!(headers.windows(8).any(|w| w == b"Syllabix"));

        let pcm = sine(OPUS_FRAME_SAMPLES * 2);
        let pages = enc.encode(&pcm).expect("encode");
        assert!(pages.starts_with(b"OggS"));

        let mut dec = OggOpusDecoder::new().expect("decoder");
        assert!(dec.decode(&headers).expect("skip headers").is_empty());
        let got = dec.decode(&pages).expect("decode");
        assert_eq!(got.len(), pcm.len());
        // VoIP Opus is lossy; allow a small encoder delay when scoring.
        let aligned = best_corr(&pcm, &got);
        assert!(aligned > 0.7, "correlation {aligned} too low");

        let mut enc = OggOpusEncoder::new().expect("encoder");
        let mut silent_pages = enc.encode(&[]).expect("headers");
        silent_pages.extend(
            enc.encode(&vec![0i16; OPUS_FRAME_SAMPLES * 2])
                .expect("silence"),
        );
        let mut dec = OggOpusDecoder::new().expect("decoder");
        let quiet = dec.decode(&silent_pages).expect("silence pcm");
        assert_eq!(quiet.len(), OPUS_FRAME_SAMPLES * 2);
        assert!(quiet.iter().all(|s| s.abs() < 64), "silence decoded loud");
    }

    fn best_corr(a: &[i16], b: &[i16]) -> f64 {
        let mut best = f64::NEG_INFINITY;
        for lag in 0..240 {
            if b.len() <= lag + OPUS_FRAME_SAMPLES {
                break;
            }
            best = best.max(corr(a, &b[lag..]));
        }
        best
    }

    #[test]
    fn partial_frame_is_buffered() {
        let mut enc = OggOpusEncoder::new().expect("encoder");
        let _ = enc.encode(&[]).expect("headers");
        let pages = enc.encode(&sine(OPUS_FRAME_SAMPLES - 1)).expect("partial");
        assert!(
            pages.is_empty(),
            "incomplete Opus frame must not emit a page"
        );
        let rest = sine(1);
        let pages = enc.encode(&rest).expect("complete");
        assert!(!pages.is_empty());
        let mut dec = OggOpusDecoder::new().expect("decoder");
        assert_eq!(dec.decode(&pages).expect("pcm").len(), OPUS_FRAME_SAMPLES);
    }

    #[test]
    fn pages_can_be_split_across_pushes() {
        let mut enc = OggOpusEncoder::new().expect("encoder");
        let mut all = enc.encode(&[]).expect("headers");
        all.extend(enc.encode(&sine(OPUS_FRAME_SAMPLES)).expect("audio"));

        let mut dec = OggOpusDecoder::new().expect("decoder");
        let mid = all.len() / 2;
        let first = dec.decode(&all[..mid]).expect("first half");
        let second = dec.decode(&all[mid..]).expect("second half");
        assert!(first.is_empty() || first.len() == OPUS_FRAME_SAMPLES);
        let pcm: Vec<i16> = first.into_iter().chain(second).collect();
        assert_eq!(pcm.len(), OPUS_FRAME_SAMPLES);
    }

    #[test]
    fn audio_message_roundtrip_wraps_ogg() {
        let mut enc = OggOpusEncoder::new().expect("encoder");
        let mut payload = enc.encode(&[]).expect("headers");
        payload.extend(enc.encode(&sine(OPUS_FRAME_SAMPLES)).expect("audio"));
        let msg = MoshiMessage::Audio(payload.clone());
        let decoded = MoshiMessage::decode(&msg.encode())
            .expect("framing")
            .expect("audio");
        let MoshiMessage::Audio(pages) = decoded else {
            panic!("expected audio");
        };
        let mut dec = OggOpusDecoder::new().expect("decoder");
        assert_eq!(dec.decode(&pages).expect("pcm").len(), OPUS_FRAME_SAMPLES);
    }

    #[test]
    fn garbage_ogg_is_an_error() {
        let mut dec = OggOpusDecoder::new().expect("decoder");
        let err = dec.decode(b"not-ogg").expect_err("garbage");
        assert!(err.to_string().contains("OggS"));
    }

    #[test]
    fn incomplete_page_waits_for_more_bytes() {
        let mut dec = OggOpusDecoder::new().expect("decoder");
        assert!(dec.decode(b"OggS").expect("short").is_empty());
        assert!(dec.decode(&[0u8; 20]).expect("still short").is_empty());
    }

    #[test]
    fn unsupported_ogg_version_is_an_error() {
        let mut page = vec![0u8; 27];
        page[..4].copy_from_slice(b"OggS");
        page[4] = 1;
        let mut dec = OggOpusDecoder::new().expect("decoder");
        let err = dec.decode(&page).expect_err("version");
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn constants_match_moshi() {
        assert_eq!(MOSHI_SAMPLE_RATE_HZ, 24_000);
        assert_eq!(MOSHI_CHANNELS, 1);
        assert_eq!(OPUS_FRAME_SAMPLES, 960);
        assert_eq!(OGG_SERIAL, 42);
        assert_eq!(
            OPUS_FRAME_SAMPLES * 1000 / MOSHI_SAMPLE_RATE_HZ as usize,
            40
        );
    }
}
