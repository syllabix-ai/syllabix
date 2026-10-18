//! Contributor-only VAD endpointing lab (`syllabix vad-lab`).
//!
//! The lab builds two-utterance streams from human-speech fixtures, joins them
//! with a configurable pause, optionally mixes stationary/ambient noise at a
//! target SNR, and counts how many turns the Silero turn policy closes. It
//! compares the production-shaped post-processing stages plus one candidate
//! change (a hysteresis rule on the silence counter).
//!
//! Accuracy is reported as `ok / total` over a noise × pause grid, with the
//! two failure modes that matter for a voice agent: `false_split` (a turn
//! closes too early — "cutting me off") and `missed` (a pause inside the
//! end-silence window swallows a turn). `end_silence_ms` is the dominant
//! lever; the lab exists so a turn-policy change can be re-measured instead
//! of guessed.
//!
//! This is hidden from `--help`, contributor-only, and the `syllabix` binary
//! is not published (the workspace sets `publish = false`), so it never
//! reaches crates.io. The AEC/NS stage here is a capture-only approximation of
//! `EchoController` — it has no render reference — so its numbers are
//! indicative, not a substitute for a live-mic A/B.
//!
//! Run: `cargo run -p syllabix -- vad-lab`.

use std::io::Cursor;
use std::ops::Range;
use std::time::Duration;

use sonora::config::{EchoCanceller, HighPassFilter, NoiseSuppression};
use sonora::{AudioProcessing, Config, StreamConfig};
use syllabix_core::audio::{i16_to_f32, read_wav, record_fixture_to_frames};
use syllabix_core::models::{HttpFetcher, StderrProgress};
use syllabix_core::types::{AudioFrame, DEFAULT_SAMPLE_RATE_HZ};
use syllabix_core::vad::{SileroVad, VadSettings, MIN_SPEECH_FRAMES, SPEECH_THRESHOLD};
use syllabix_core::{Cancel, Error, ModelCache, Result, Vad, VadEvent};

const SR: usize = DEFAULT_SAMPLE_RATE_HZ as usize;
/// Silero scores one 512-sample (32 ms) frame at a time.
const FRAME: usize = 512;
/// WebRTC audio processing (AEC3/NS) consumes 10 ms blocks.
const AEC_BLOCK: usize = SR / 100;
const RNG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// Pause after the second utterance so the closing hangover has room to fire.
const TRAIL_MS: usize = 3000;

/// End-silence settings to compare. 350 ms is the built-in default; 1000 ms
/// shows what the same policy does with a longer hangover.
const END_SILENCE_MS: [u64; 2] = [350, 1000];
const GAPS_MS: [usize; 5] = [250, 350, 500, 1000, 1500];
/// `(kind, snr_db)`; `snr_db > 100` means clean (no noise mixed).
const NOISES: [(&str, f32); 7] = [
    ("clean", 999.0),
    ("white", 10.0),
    ("white", 5.0),
    ("pink", 5.0),
    ("brown", 5.0),
    ("ambient", 10.0),
    ("ambient", 5.0),
];

/// Speech fixtures, embedded so the lab runs from any directory.
const POOL: [(&str, &[u8]); 4] = [
    (
        "librispeech-1089",
        include_bytes!("../../syllabix-core/tests/fixtures/stt/librispeech-1089-134686-0000.wav"),
    ),
    (
        "librispeech-121",
        include_bytes!("../../syllabix-core/tests/fixtures/stt/librispeech-121-127105-0009.wav"),
    ),
    (
        "librispeech-1995",
        include_bytes!("../../syllabix-core/tests/fixtures/stt/librispeech-1995-1837-0005.wav"),
    ),
    (
        "vad-speech",
        include_bytes!("../../syllabix-core/tests/fixtures/vad/speech.wav"),
    ),
];
const AMBIENT: &[u8] = include_bytes!("../../syllabix-core/tests/fixtures/vad/noise.wav");

const MODES: [&str; 5] = ["raw", "aec", "aecns", "hyst", "both"];

/// Decode an embedded 16 kHz mono WAV fixture to a flat PCM16 stream.
fn fixture(bytes: &[u8]) -> Result<Vec<i16>> {
    let wav = read_wav(Cursor::new(bytes))?;
    Ok(record_fixture_to_frames(&wav)?
        .into_iter()
        .flat_map(|frame| frame.samples)
        .collect())
}

/// xorshift64* — small, deterministic, no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        ((x >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn noise_sequence(kind: &str, len: usize, rng: &mut Rng) -> Result<Vec<f32>> {
    Ok(match kind {
        "white" => (0..len).map(|_| 0.3 * rng.next()).collect(),
        "pink" => {
            let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            (0..len)
                .map(|_| {
                    let w = rng.next();
                    b0 = 0.99886 * b0 + w * 0.055_517_9;
                    b1 = 0.99332 * b1 + w * 0.075_075_9;
                    b2 = 0.969 * b2 + w * 0.153_852;
                    b3 = 0.8665 * b3 + w * 0.310_485_6;
                    b4 = 0.55 * b4 + w * 0.532_952_2;
                    b5 = -0.7616 * b5 - w * 0.016_898;
                    let out = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
                    b6 = w * 0.115926;
                    out * 0.11
                })
                .collect()
        }
        "brown" => {
            let mut last = 0.0;
            (0..len)
                .map(|_| {
                    last = (last + 0.02 * rng.next()) / 1.02;
                    last * 3.5
                })
                .collect()
        }
        "ambient" => {
            let base: Vec<f32> = fixture(AMBIENT)?
                .iter()
                .map(|sample| f32::from(*sample) / 32768.0)
                .collect();
            (0..len).map(|i| base[i % base.len()]).collect()
        }
        other => {
            return Err(Error::InvalidAudio {
                message: format!("unknown noise kind {other:?}"),
            });
        }
    })
}

fn rms(samples: &[f32]) -> f32 {
    (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Mix `noise` into `clean` at `snr_db`, referenced to the clean signal RMS.
fn mix(clean: &[i16], noise: &[f32], snr_db: f32) -> Vec<i16> {
    let clean_f: Vec<f32> = clean
        .iter()
        .map(|sample| f32::from(*sample) / 32768.0)
        .collect();
    let gain = rms(&clean_f) / 10f32.powf(snr_db / 20.0) / rms(noise);
    clean_f
        .iter()
        .zip(noise)
        .map(|(s, n)| ((s + gain * n).clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect()
}

/// Pad into fixed 512-sample frames; the final frame is zero-filled.
fn frames(samples: &[i16]) -> Vec<AudioFrame> {
    samples
        .chunks(FRAME)
        .enumerate()
        .map(|(index, chunk)| {
            let mut block = chunk.to_vec();
            block.resize(FRAME, 0);
            AudioFrame::new(index as u64, SR as u32, 1, block).expect("frame layout")
        })
        .collect()
}

/// Mirror of `EchoController::new`'s processor, with an optional NS stage.
/// No render reference is fed, so this measures the capture-side stages only.
fn sonora_process(samples: &[i16], ns: bool) -> Result<Vec<i16>> {
    let stream = StreamConfig::new(SR as u32, 1);
    let config = Config {
        echo_canceller: Some(EchoCanceller::default()),
        high_pass_filter: Some(HighPassFilter::default()),
        noise_suppression: ns.then(NoiseSuppression::default),
        ..Default::default()
    };
    let mut ap = AudioProcessing::builder()
        .config(config)
        .capture_config(stream)
        .render_config(stream)
        .echo_detector(true)
        .build();
    let input = i16_to_f32(samples);
    let mut out = Vec::with_capacity(input.len());
    for chunk in input.chunks(AEC_BLOCK) {
        let mut block = chunk.to_vec();
        block.resize(AEC_BLOCK, 0.0);
        let mut clean = vec![0.0f32; AEC_BLOCK];
        ap.process_capture_f32(&[block.as_slice()], &mut [clean.as_mut_slice()])
            .map_err(|err| Error::Provider {
                provider: "sonora",
                message: err.to_string(),
            })?;
        out.extend(clean);
    }
    out.truncate(input.len());
    Ok(out
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect())
}

/// Turn count when `samples` is pushed through the real `SileroVad`.
fn real_turns(vad: &mut SileroVad, samples: &[i16]) -> Result<usize> {
    vad.reset();
    let mut ends = 0;
    for frame in frames(samples) {
        for event in vad.push_frame(frame)? {
            if matches!(event, VadEvent::SpeechEnd { .. }) {
                ends += 1;
            }
        }
    }
    Ok(ends)
}

/// Frame-stepped replica of `SileroVad::push_frame` plus the candidate
/// hysteresis rule: while a turn is open, a frame only counts toward the
/// silence hangover when `p < max(0.01, SPEECH_THRESHOLD - 0.15)`.
fn sim_turns(
    vad: &mut SileroVad,
    samples: &[i16],
    end_silence_ms: u64,
    hysteresis: bool,
) -> Result<usize> {
    vad.reset();
    let silence_needed = (end_silence_ms as usize).div_ceil(32);
    let hyst = (SPEECH_THRESHOLD - 0.15).max(0.01);
    let mut open = false;
    let mut speech = 0;
    let mut silence = 0;
    let mut turns = 0;
    for frame in frames(samples) {
        let p = vad.debug_probability(&frame)?;
        if !open {
            if p >= SPEECH_THRESHOLD {
                speech += 1;
                if speech >= MIN_SPEECH_FRAMES {
                    open = true;
                    silence = 0;
                }
            } else {
                speech = 0;
            }
        } else if p >= SPEECH_THRESHOLD || (hysteresis && p >= hyst) {
            silence = 0;
        } else {
            silence += 1;
            if silence >= silence_needed {
                turns += 1;
                open = false;
                speech = 0;
                silence = 0;
                vad.reset();
            }
        }
    }
    Ok(turns)
}

/// Longest contiguous ≥threshold run, tolerating up to three sub-threshold
/// frames, so a fixture yields one clean utterance.
fn speech_segment(vad: &mut SileroVad, samples: &[i16]) -> Result<Vec<i16>> {
    vad.reset();
    let probs: Vec<f32> = frames(samples)
        .iter()
        .map(|frame| vad.debug_probability(frame))
        .collect::<Result<_>>()?;
    let mut best = (0usize, 0usize);
    let mut i = 0;
    while i < probs.len() {
        if probs[i] < 0.5 {
            i += 1;
            continue;
        }
        let start = i;
        let mut end = i;
        let mut gap = 0;
        let mut j = i;
        while j < probs.len() {
            if probs[j] >= 0.5 {
                end = j;
                gap = 0;
            } else {
                gap += 1;
                if gap > 3 {
                    break;
                }
            }
            j += 1;
        }
        if end - start > best.1 - best.0 {
            best = (start, end);
        }
        i = j;
    }
    Ok(samples[best.0 * FRAME..(best.1 + 1) * FRAME].to_vec())
}

fn silence_ms(ms: usize) -> Vec<i16> {
    vec![0; SR * ms / 1000]
}

/// `A + pause + B + trailing silence`.
fn stream(a: &[i16], b: &[i16], gap_ms: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(a.len() + b.len() + SR * (gap_ms + TRAIL_MS) / 1000);
    out.extend_from_slice(a);
    out.extend(silence_ms(gap_ms));
    out.extend_from_slice(b);
    out.extend(silence_ms(TRAIL_MS));
    out
}

#[derive(Default, Clone, Copy)]
struct Score {
    ok: usize,
    false_split: usize,
    missed: usize,
    no_detect: usize,
    total: usize,
}

impl Score {
    fn add(&mut self, expected: usize, got: usize) {
        self.total += 1;
        match (expected, got) {
            (e, g) if g == e => self.ok += 1,
            (_, 0) => self.no_detect += 1,
            (e, g) if g > e => self.false_split += 1,
            _ => self.missed += 1,
        }
    }
}

struct Cell {
    kind: &'static str,
    snr_db: f32,
    gap_ms: usize,
    samples: Vec<i16>,
}

fn build_cells(a: &[i16], b: &[i16]) -> Result<Vec<Cell>> {
    let mut cells = Vec::with_capacity(NOISES.len() * GAPS_MS.len());
    for (kind, snr_db) in NOISES {
        for gap_ms in GAPS_MS {
            let clean = stream(a, b, gap_ms);
            let samples = if snr_db > 100.0 {
                clean
            } else {
                let mut rng = Rng(RNG_SEED);
                let noise = noise_sequence(kind, clean.len(), &mut rng)?;
                mix(&clean, &noise, snr_db)
            };
            cells.push(Cell {
                kind,
                snr_db,
                gap_ms,
                samples,
            });
        }
    }
    Ok(cells)
}

fn real_vad(cache: &ModelCache, end_silence_ms: u64) -> Result<SileroVad> {
    Ok(SileroVad::from_cache(
        cache,
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )?
    .with_settings(VadSettings {
        end_silence: Duration::from_millis(end_silence_ms),
        ..VadSettings::v0()
    }))
}

fn rms_range(samples: &[i16], range: Range<usize>) -> f32 {
    let slice: Vec<f32> = samples[range]
        .iter()
        .map(|sample| f32::from(*sample) / 32768.0)
        .collect();
    rms(&slice) * 1000.0
}

/// Run the lab and print the numbers.
pub fn run() -> Result<()> {
    let cache = ModelCache::v0();
    let mut seed = SileroVad::from_cache(
        &cache,
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )?;

    let mut segments: Vec<Vec<i16>> = POOL
        .iter()
        .map(|(_, bytes)| fixture(bytes).and_then(|s| speech_segment(&mut seed, &s)))
        .collect::<Result<_>>()?;
    segments.sort_by_key(|segment| std::cmp::Reverse(segment.len()));
    let (a, b) = (segments[0].clone(), segments[1].clone());

    println!("VAD endpointing lab");
    println!(
        "  fixtures: A={}ms B={}ms | noise x pause grid: {} x {} = {} cells",
        a.len() * 1000 / SR,
        b.len() * 1000 / SR,
        NOISES.len(),
        GAPS_MS.len(),
        NOISES.len() * GAPS_MS.len()
    );
    println!(
        "  note: aec/aecns use capture-only AEC3 (no render reference); \
         treat those columns as indicative, not a live-mic verdict\n"
    );

    let cells = build_cells(&a, &b)?;

    // The simulated state machine must agree with the real VAD baseline.
    let mut real_350 = real_vad(&cache, 350)?;
    let mut real_1000 = real_vad(&cache, 1000)?;
    let mut mismatch = 0;
    for cell in &cells {
        if real_turns(&mut real_350, &cell.samples)?
            != sim_turns(&mut seed, &cell.samples, 350, false)?
        {
            mismatch += 1;
        }
        if real_turns(&mut real_1000, &cell.samples)?
            != sim_turns(&mut seed, &cell.samples, 1000, false)?
        {
            mismatch += 1;
        }
    }
    println!("simulation baseline mismatches vs real VAD: {mismatch}\n");

    for end_silence_ms in END_SILENCE_MS {
        println!("=== end_silence_ms={end_silence_ms} ===");
        println!(
            "  {:>12} {:>5} {:>4} | {:>4} {:>4} {:>5} {:>4} {:>5}",
            "noise/snr", "pause", "exp", "raw", "aec", "aecns", "hyst", "both"
        );
        let mut scores = [Score::default(); MODES.len()];
        for cell in &cells {
            let expected = if (cell.gap_ms as u64) < end_silence_ms {
                1
            } else {
                2
            };
            let aec = sonora_process(&cell.samples, false)?;
            let aecns = sonora_process(&cell.samples, true)?;
            let got = [
                sim_turns(&mut seed, &cell.samples, end_silence_ms, false)?,
                sim_turns(&mut seed, &aec, end_silence_ms, false)?,
                sim_turns(&mut seed, &aecns, end_silence_ms, false)?,
                sim_turns(&mut seed, &cell.samples, end_silence_ms, true)?,
                sim_turns(&mut seed, &aecns, end_silence_ms, true)?,
            ];
            for (mode, turns) in got.iter().enumerate() {
                scores[mode].add(expected, *turns);
            }
            println!(
                "  {:>7}/{:<4} {:>5} {:>4} | {:>4} {:>4} {:>5} {:>4} {:>5}",
                cell.kind,
                cell.snr_db,
                cell.gap_ms,
                expected,
                got[0],
                got[1],
                got[2],
                got[3],
                got[4]
            );
        }
        for (mode, name) in MODES.iter().enumerate() {
            let score = scores[mode];
            println!(
                "    {name:<6} ok={:>2}/{} false_split={:>2} missed={:>2} no_detect={}",
                score.ok, score.total, score.false_split, score.missed, score.no_detect
            );
        }
        println!();
    }

    // Does NS attenuate gap noise, and does the AEC stage leave speech intact?
    println!("--- energy probe (white@10 dB, 1000 ms pause), RMS x1000 ---");
    let clean = stream(&a, &b, 1000);
    let mut rng = Rng(RNG_SEED);
    let noise = noise_sequence("white", clean.len(), &mut rng)?;
    let mixed = mix(&clean, &noise, 10.0);
    let aec = sonora_process(&mixed, false)?;
    let aecns = sonora_process(&mixed, true)?;
    let gap = a.len()..a.len() + SR;
    println!(
        "  pause  raw={:.2} aec={:.2} aecns={:.2}",
        rms_range(&mixed, gap.clone()),
        rms_range(&aec, gap.clone()),
        rms_range(&aecns, gap.clone())
    );
    println!(
        "  speech raw={:.2} aec={:.2} aecns={:.2}",
        rms_range(&mixed, 0..a.len()),
        rms_range(&aec, 0..a.len()),
        rms_range(&aecns, 0..a.len())
    );

    Ok(())
}
