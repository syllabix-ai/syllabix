//! PR 3 merge gate: fixture record/play on every OS, and a simulated 30-minute soak.
//!
//! The soak feeds 30 minutes of *audio time* through conversion + bounded queues.
//! It is not a 30-minute wall-clock wait.

#[cfg(target_os = "linux")]
use std::ffi::OsString;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::Duration;

#[cfg(target_os = "linux")]
use syllabix_core::audio::probe_device_names;
use syllabix_core::audio::{
    live_buffer_ceiling_bytes, record_and_play_fixture, select_input, select_output, sine_i16,
    write_wav, DeviceInfo, DeviceInventory, FixtureCapture, FrameSplitter, PcmConverter, PcmFormat,
    WavPcm, AUDIO_LIVE_BYTES_CEILING, FRAME_SPLITTER_MAX,
};
use syllabix_core::{bounded, AudioCapture, Cancel, QueueCaps, FRAME_SAMPLES};

fn fixture_wav() -> WavPcm {
    WavPcm {
        format: PcmFormat {
            sample_rate_hz: 48_000,
            channels: 2,
        },
        samples: sine_i16(48_000, 2, 440.0, Duration::from_millis(250), 0.5),
    }
}

#[cfg(target_os = "linux")]
struct AlsaNullDevice {
    _lock: MutexGuard<'static, ()>,
    prior_config: Option<OsString>,
}

#[cfg(target_os = "linux")]
fn alsa_config_lock() -> &'static Mutex<()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(target_os = "linux")]
impl AlsaNullDevice {
    fn install() -> Self {
        let lock = alsa_config_lock().lock().expect("ALSA config lock");
        let prior_config = std::env::var_os("ALSA_CONFIG_PATH");
        std::env::set_var(
            "ALSA_CONFIG_PATH",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/alsa-null.conf"),
        );
        Self {
            _lock: lock,
            prior_config,
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for AlsaNullDevice {
    fn drop(&mut self) {
        match &self.prior_config {
            Some(value) => std::env::set_var("ALSA_CONFIG_PATH", value),
            None => std::env::remove_var("ALSA_CONFIG_PATH"),
        }
    }
}

#[test]
fn record_and_play_wav_fixture() {
    let wav = fixture_wav();
    let mut encoded = Vec::new();
    write_wav(&mut encoded, &wav).expect("encode wav");
    let decoded = syllabix_core::audio::read_wav(Cursor::new(&encoded)).expect("decode wav");
    assert_eq!(decoded.format.sample_rate_hz, 48_000);
    assert_eq!(decoded.format.channels, 2);

    let (frames, played) = record_and_play_fixture(&decoded).expect("fixture io");
    assert!(!frames.is_empty(), "capture must emit v0 frames");
    for frame in &frames {
        frame.validate().expect("v0 frame");
        assert_eq!(frame.samples.len(), FRAME_SAMPLES);
    }
    // 250 ms at 16 kHz ≈ 4000 samples ≈ 7 frames of 512, plus flush padding.
    assert!(frames.len() >= 7, "got {} frames", frames.len());
    assert!(
        frames.iter().any(|f| f.has_energy()),
        "recorded fixture should not be silence"
    );

    // Playback is 48 kHz stereo: ~250 ms → ~24000 frames * 2 samples.
    assert!(played.len() > 8_000, "played {} samples", played.len());
    let energy: f32 = played.iter().map(|s| s.abs()).sum();
    assert!(energy > 1.0, "playback PCM was silent (energy {energy})");

    // Stereo duplicate: L and R of the first frame should match after mono roundtrip.
    assert_eq!(played.len() % 2, 0);
    let l = played[0];
    let r = played[1];
    assert!(
        (l - r).abs() < 1e-5,
        "mono TTS/capture path duplicates channels"
    );
}

#[test]
fn fixture_capture_trait_yields_ordered_frames() {
    let wav = fixture_wav();
    let mut cap = FixtureCapture::from_wav(&wav).unwrap();
    assert_eq!(cap.name(), "fixture");
    let cancel = Cancel::new();
    let mut seq = 0u64;
    let mut n = 0;
    while let Some(frame) = cap.next_frame(&cancel).unwrap() {
        assert_eq!(frame.seq, seq);
        seq += 1;
        n += 1;
    }
    assert!(n >= 7);
}

#[test]
fn device_selection_uses_last_resort_devices() {
    struct NoDefaults {
        inputs: Vec<DeviceInfo>,
        outputs: Vec<DeviceInfo>,
    }

    impl DeviceInventory for NoDefaults {
        fn default_input(&self) -> Option<DeviceInfo> {
            None
        }

        fn default_output(&self) -> Option<DeviceInfo> {
            None
        }

        fn inputs(&self) -> Vec<DeviceInfo> {
            self.inputs.clone()
        }

        fn outputs(&self) -> Vec<DeviceInfo> {
            self.outputs.clone()
        }
    }

    let inventory = NoDefaults {
        inputs: vec![DeviceInfo {
            name: "Stereo Mix".into(),
            sample_rate_hz: 48_000,
            channels: 2,
        }],
        outputs: vec![DeviceInfo {
            name: "Fallback speakers".into(),
            sample_rate_hz: 48_000,
            channels: 2,
        }],
    };
    assert_eq!(select_input(&inventory).unwrap().reason, "only-available");
    assert_eq!(select_output(&inventory).unwrap().reason, "first-available");
}

/// 30 minutes of audio time, processed faster than real time.
#[test]
fn simulated_thirty_minute_loop_stays_in_bounds() {
    const AUDIO_SECONDS: u32 = 30 * 60;
    const DEVICE_RATE: u32 = 48_000;
    const DEVICE_CH: u16 = 2;
    // 20 ms of device PCM per chunk: 48000 * 0.02 * 2 = 1920 samples.
    const CHUNK_FRAMES: usize = 960;
    let caps = QueueCaps::v0();
    let (frame_tx, frame_rx, frame_stats) = bounded("frames", caps.frames);
    let (play_tx, play_rx, play_stats) = bounded("audio", caps.audio);
    let frame_stats_prod = std::sync::Arc::clone(&frame_stats);
    let frame_stats_play = std::sync::Arc::clone(&frame_stats);
    let play_stats_play = std::sync::Arc::clone(&play_stats);

    let live_max = Arc::new(AtomicUsize::new(0));
    let live_after_warmup = Arc::new(AtomicUsize::new(0));
    let live_end = Arc::new(AtomicUsize::new(0));

    let producer = {
        let live_max = Arc::clone(&live_max);
        thread::spawn(move || {
            let mut conv = PcmConverter::new(
                PcmFormat {
                    sample_rate_hz: DEVICE_RATE,
                    channels: DEVICE_CH,
                },
                PcmFormat::v0(),
            )
            .unwrap();
            let mut split = FrameSplitter::new();
            let total_device_frames = DEVICE_RATE as usize * AUDIO_SECONDS as usize;
            let mut produced = 0usize;
            let mut phase = 0.0f32;
            let step = 440.0 * 2.0 * std::f32::consts::PI / DEVICE_RATE as f32;
            while produced < total_device_frames {
                let n = CHUNK_FRAMES.min(total_device_frames - produced);
                let mut chunk = Vec::with_capacity(n * DEVICE_CH as usize);
                for _ in 0..n {
                    let s = phase.sin() * 0.2;
                    phase += step;
                    chunk.push(s);
                    chunk.push(s);
                }
                produced += n;
                let converted = conv.push(&chunk);
                let i16s = syllabix_core::audio::f32_to_i16(&converted);
                for frame in split.push(&i16s).unwrap() {
                    frame_tx.send(frame).unwrap();
                }
                let leftover = conv.leftover_bytes()
                    + split.leftover_samples() * 2
                    + frame_stats_prod.snapshot().current * FRAME_SAMPLES * 2;
                live_max.fetch_max(leftover, Ordering::SeqCst);
                assert!(
                    split.leftover_samples() <= FRAME_SPLITTER_MAX,
                    "frame splitter grew"
                );
                assert!(
                    leftover <= AUDIO_LIVE_BYTES_CEILING,
                    "capture live bytes {leftover} over ceiling"
                );
            }
            for frame in split.flush().unwrap() {
                frame_tx.send(frame).unwrap();
            }
            let _ = conv.flush();
        })
    };

    let player = {
        let live_max = Arc::clone(&live_max);
        let live_after_warmup = Arc::clone(&live_after_warmup);
        let live_end = Arc::clone(&live_end);
        thread::spawn(move || {
            let mut conv = PcmConverter::new(
                PcmFormat::v0(),
                PcmFormat {
                    sample_rate_hz: DEVICE_RATE,
                    channels: DEVICE_CH,
                },
            )
            .unwrap();
            let mut frames_out = 0usize;
            let warmup_frames = (2 * 16_000) / FRAME_SAMPLES;
            while let Ok(frame) = frame_rx.recv() {
                let f32s = syllabix_core::audio::i16_to_f32(&frame.samples);
                let device_pcm = conv.push(&f32s);
                // Bound the playback queue with a compact token, not the PCM itself.
                play_tx.send(device_pcm.len()).unwrap();
                frames_out += 1;
                let leftover = conv.leftover_bytes()
                    + play_stats_play.snapshot().current * std::mem::size_of::<usize>()
                    + frame_stats_play.snapshot().current * FRAME_SAMPLES * 2;
                live_max.fetch_max(leftover, Ordering::SeqCst);
                if frames_out == warmup_frames {
                    live_after_warmup.store(leftover, Ordering::SeqCst);
                }
                assert!(leftover <= AUDIO_LIVE_BYTES_CEILING);
            }
            let _ = conv.flush();
            live_end.store(conv.leftover_bytes(), Ordering::SeqCst);
        })
    };

    let speaker = thread::spawn(move || {
        while let Ok(_len) = play_rx.recv() {
            // Drop immediately: speakers do not accumulate PCM.
        }
    });

    producer.join().expect("producer");
    player.join().expect("player");
    speaker.join().expect("speaker");

    let frames = frame_stats.snapshot();
    let audio = play_stats.snapshot();
    assert!(frames.within_capacity(), "{frames:?}");
    assert!(audio.within_capacity(), "{audio:?}");
    assert_eq!(frames.current, 0);
    assert_eq!(audio.current, 0);
    assert!(frames.sent > 0);
    assert_eq!(frames.sent, frames.received);
    assert!(audio.sent > 0);
    assert_eq!(audio.sent, audio.received);
    assert!(frames.high_water <= caps.frames);
    assert!(audio.high_water <= caps.audio);

    let warmup = live_after_warmup.load(Ordering::SeqCst);
    let end = live_end.load(Ordering::SeqCst);
    let peak = live_max.load(Ordering::SeqCst);
    assert!(
        peak <= live_buffer_ceiling_bytes(),
        "peak live bytes {peak} exceeds {}",
        live_buffer_ceiling_bytes()
    );
    // After warm-up, leftover state must not trend toward unbounded growth.
    assert!(
        end <= warmup.max(4096),
        "live bytes grew after warm-up: warmup={warmup} end={end}"
    );
}

#[test]
fn native_open_fails_with_actionable_error_when_no_device() {
    #[cfg(target_os = "linux")]
    let _alsa_lock = alsa_config_lock().lock().expect("ALSA config lock");

    match syllabix_core::audio::NativeCapture::open() {
        Ok(cap) => {
            drop(cap);
        }
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("microphone")
                    || msg.contains("Microphone")
                    || msg.contains("device")
                    || msg.contains("permission")
                    || msg.contains("Speakers")
                    || msg.contains("speakers"),
                "unhelpful device error: {msg}"
            );
        }
    }
}

/// Linux CI has no real audio hardware. The ALSA null plugin is a deterministic
/// device that still drives cpal's stream callbacks and conversion path.
#[cfg(target_os = "linux")]
#[test]
fn native_streams_work_with_alsa_null_device() {
    let _alsa = AlsaNullDevice::install();
    use syllabix_core::audio::{NativeCapture, NativePlayback};
    use syllabix_core::{AudioSink, GenerationId, SynthesizedAudio, TurnId};

    let (input, output) = probe_device_names().expect("null devices");
    assert_eq!(input.reason, "os-default");
    assert_eq!(output.reason, "os-default");

    let (mut playback, reference) = NativePlayback::open_with_echo().expect("null speakers");
    let mut capture = NativeCapture::open_with_echo(reference).expect("null microphone");
    let cancel = Cancel::new();
    let tone = sine_i16(16_000, 1, 440.0, Duration::from_millis(200), 0.3);
    playback
        .play(
            SynthesizedAudio {
                turn: TurnId(0),
                generation: GenerationId(0),
                index: 0,
                samples: tone,
                is_last: true,
            },
            &cancel,
        )
        .expect("play tone");
    let frame = capture
        .next_frame(&cancel)
        .expect("read null microphone")
        .expect("null microphone must emit a frame");
    frame.validate().expect("v0 microphone frame");
}

/// Human + laptop: play a short tone through AEC and pull mic frames. Not run in CI.
#[test]
#[ignore]
fn hardware_record_and_play_if_devices_exist() {
    use syllabix_core::audio::{NativeCapture, NativePlayback};
    use syllabix_core::{AudioSink, GenerationId, SynthesizedAudio, TurnId};

    let (mut playback, reference) = NativePlayback::open_with_echo().expect("speakers");
    let mut capture = NativeCapture::open_with_echo(reference).expect("microphone");
    let cancel = Cancel::new();
    let tone = sine_i16(16_000, 1, 440.0, Duration::from_millis(200), 0.3);
    playback
        .play(
            SynthesizedAudio {
                turn: TurnId(0),
                generation: GenerationId(0),
                index: 0,
                samples: tone,
                is_last: true,
            },
            &cancel,
        )
        .expect("play tone");
    thread::sleep(Duration::from_millis(250));
    let frame = capture
        .next_frame(&cancel)
        .expect("mic frame")
        .expect("expected at least one frame");
    frame.validate().expect("v0 mic frame");
}

/// PR 17 hardware gate: a quiet laptop must not turn its own speaker into a user turn.
///
/// Run in a quiet room with the normal laptop microphone and speakers selected.
/// Do not wear headphones and do not speak during the measurement.
#[test]
#[ignore]
fn hardware_aec_1_minute_playback_has_zero_false_turns() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    use syllabix_core::audio::{read_wav, record_fixture_to_frames, NativeCapture, NativePlayback};
    use syllabix_core::{
        AudioSink, GenerationId, HttpFetcher, ModelCache, SileroVad, StderrProgress,
        SynthesizedAudio, TurnId, Vad, VadEvent,
    };

    const CALIBRATION: Duration = Duration::from_secs(10);
    const MEASUREMENT: Duration = Duration::from_secs(60);

    let mut vad = SileroVad::from_cache(
        &ModelCache::v0(),
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )
    .expect("load Silero for hardware AEC gate");
    let fixture_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vad/speech.wav");
    let wav = read_wav(
        std::fs::File::open(&fixture_path)
            .unwrap_or_else(|err| panic!("open {}: {err}", fixture_path.display())),
    )
    .expect("read speaker fixture");
    let playback_samples = record_fixture_to_frames(&wav)
        .expect("convert speaker fixture")
        .into_iter()
        .flat_map(|frame| frame.samples)
        .collect::<Vec<_>>();

    let (mut playback, reference) = NativePlayback::open_with_echo().expect("speakers with AEC");
    let mut capture = NativeCapture::open_with_echo(reference).expect("microphone with AEC");
    let cancel = Cancel::new();
    let playing = Arc::new(AtomicBool::new(true));
    let playing_worker = Arc::clone(&playing);
    let playback_cancel = cancel.clone();
    let playback_worker = thread::spawn(move || {
        let mut index = 0_u32;
        while playing_worker.load(Ordering::SeqCst) {
            let result = playback.play(
                SynthesizedAudio {
                    turn: TurnId(0),
                    generation: GenerationId(0),
                    index,
                    samples: playback_samples.clone(),
                    is_last: true,
                },
                &playback_cancel,
            );
            if let Err(err) = result {
                if playback_cancel.is_shutdown() {
                    break;
                }
                panic!("continuous agent playback: {err}");
            }
            index = index.wrapping_add(1);
        }
    });

    let started = Instant::now();
    let measurement_start = started + CALIBRATION;
    let deadline = measurement_start + MEASUREMENT;
    let mut measuring = false;
    let mut false_starts = 0_u64;
    while Instant::now() < deadline {
        let Some(frame) = capture.next_frame(&cancel).expect("AEC microphone frame") else {
            panic!("microphone ended during hardware AEC gate");
        };
        if !measuring && Instant::now() >= measurement_start {
            vad.reset();
            measuring = true;
            eprintln!("AEC calibrated; starting 1-minute false-turn measurement");
        }
        if measuring {
            false_starts += vad
                .push_frame(frame)
                .expect("Silero on AEC capture")
                .iter()
                .filter(|event| matches!(event, VadEvent::SpeechStart { .. }))
                .count() as u64;
        }
    }

    playing.store(false, Ordering::SeqCst);
    cancel.shutdown();
    playback_worker.join().expect("playback worker");
    assert_eq!(
        false_starts, 0,
        "agent playback caused {false_starts} false Silero turns"
    );
}
