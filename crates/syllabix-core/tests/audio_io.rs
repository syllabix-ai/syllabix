//! PR 3 merge gate: fixture record/play on every OS, and a simulated 30-minute soak.
//!
//! The soak feeds 30 minutes of *audio time* through conversion + bounded queues.
//! It is not a 30-minute wall-clock wait.

#[cfg(target_os = "linux")]
use std::ffi::OsString;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(target_os = "linux")]
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::Duration;

#[cfg(target_os = "linux")]
use syllabix_core::audio::probe_device_names;
use syllabix_core::audio::{
    f32_to_i16, i16_to_f32, live_buffer_ceiling_bytes, record_and_play_fixture, select_input,
    select_output, sine_i16, write_wav, DeviceInfo, DeviceInventory, EchoController, EchoRecording,
    EchoReference, FixtureCapture, FrameSplitter, PcmConverter, PcmFormat, SampleRing, WavPcm,
    AEC_FRAME_SAMPLES, AUDIO_LIVE_BYTES_CEILING, FRAME_SPLITTER_MAX,
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
    capture
        .echo_mut()
        .expect("AEC controller")
        .enable_recording();
    eprintln!(
        "AEC hardware gate: input={} ({} Hz {} ch) output={} ({} Hz {} ch)",
        capture.device_name,
        capture.device_format.sample_rate_hz,
        capture.device_format.channels,
        playback.device_name,
        playback.device_format.sample_rate_hz,
        playback.device_format.channels
    );
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
    let mut speech_starts: Vec<String> = Vec::new();
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
            let hits = vad
                .push_frame(frame)
                .expect("Silero on AEC capture")
                .iter()
                .filter(|event| matches!(event, VadEvent::SpeechStart { .. }))
                .count() as u64;
            if hits > 0 {
                let at = started.elapsed().as_secs_f32();
                speech_starts.push(format!("{at:.3}s"));
                false_starts += hits;
            }
        }
    }

    playing.store(false, Ordering::SeqCst);
    cancel.shutdown();
    playback_worker.join().expect("playback worker");
    let rec = capture.echo_mut().expect("AEC controller").recording();
    let dump_dir = write_aec_debug_dump(&rec, &speech_starts, false_starts);
    eprintln!(
        "AEC dump {} calibration={:?} estimated_delay_ms={:?} aec_delay_ms={:?} far_end_blocks={} holds={} false_starts={false_starts} starts={speech_starts:?}",
        dump_dir.display(),
        rec.calibration,
        rec.estimated_delay_ms,
        rec.aec_delay_ms,
        rec.far_end_blocks,
        rec.render_starved_holds,
    );
    assert_eq!(
        false_starts,
        0,
        "agent playback caused {false_starts} false Silero turns; dumps in {}",
        dump_dir.display()
    );
}

fn aec_debug_dir() -> PathBuf {
    std::env::var("SYLLABIX_AEC_DEBUG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/aec-debug")
        })
}

fn write_mono16k_wav(path: &Path, samples: &[f32]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create AEC dump dir");
    }
    let wav = WavPcm {
        format: PcmFormat {
            sample_rate_hz: 16_000,
            channels: 1,
        },
        samples: f32_to_i16(samples),
    };
    let file = std::fs::File::create(path)
        .unwrap_or_else(|err| panic!("create {}: {err}", path.display()));
    write_wav(file, &wav).expect("write AEC wav");
}

fn write_aec_debug_dump(
    rec: &EchoRecording,
    speech_starts: &[String],
    false_starts: u64,
) -> PathBuf {
    let dir = aec_debug_dir();
    std::fs::create_dir_all(&dir).expect("create AEC dump dir");
    write_mono16k_wav(&dir.join("render.wav"), &rec.render);
    write_mono16k_wav(&dir.join("capture.wav"), &rec.capture);
    write_mono16k_wav(&dir.join("clean.wav"), &rec.clean);
    let sidecar = format!(
        "{{\n  \"calibration\": \"{:?}\",\n  \"estimated_delay_ms\": {},\n  \"aec_delay_ms\": {},\n  \"far_end_blocks\": {},\n  \"render_starved_holds\": {},\n  \"false_starts\": {false_starts},\n  \"speech_starts\": [{}],\n  \"render_samples\": {},\n  \"capture_samples\": {},\n  \"clean_samples\": {}\n}}\n",
        rec.calibration,
        rec.estimated_delay_ms
            .map(|ms| ms.to_string())
            .unwrap_or_else(|| "null".into()),
        rec.aec_delay_ms
            .map(|ms| ms.to_string())
            .unwrap_or_else(|| "null".into()),
        rec.far_end_blocks,
        rec.render_starved_holds,
        speech_starts
            .iter()
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", "),
        rec.render.len(),
        rec.capture.len(),
        rec.clean.len(),
    );
    std::fs::write(dir.join("sidecar.json"), sidecar).expect("write AEC sidecar");
    dir
}

fn load_vad_speech_f32() -> Vec<f32> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vad/speech.wav");
    let wav = syllabix_core::audio::read_wav(
        std::fs::File::open(&path).unwrap_or_else(|err| panic!("open {}: {err}", path.display())),
    )
    .expect("read speech.wav");
    let frames = syllabix_core::audio::record_fixture_to_frames(&wav).expect("speech frames");
    let samples: Vec<i16> = frames.into_iter().flat_map(|frame| frame.samples).collect();
    i16_to_f32(&samples)
}

/// Scripted laptop: speech fixture as far-end, 80 ms delayed leak into the mic.
/// Silero must not treat the cancelled echo as a user turn.
#[test]
fn delayed_speech_fixture_is_cancelled_before_silero() {
    const DELAY: usize = AEC_FRAME_SAMPLES * 8;
    const ADAPT_LOOPS: usize = 8;
    const MEASURE_LOOPS: usize = 2;

    let speech = load_vad_speech_f32();
    assert!(
        speech.len() > AEC_FRAME_SAMPLES * 20,
        "speech.wav too short"
    );
    let ring = Arc::new(SampleRing::new(AEC_FRAME_SAMPLES * 8));
    let mut echo =
        EchoController::new(EchoReference::new(Arc::clone(&ring), PcmFormat::v0())).expect("echo");
    let mut history = std::collections::VecDeque::from(vec![0.0f32; DELAY]);
    let mut measure_clean = Vec::new();
    for loop_idx in 0..(ADAPT_LOOPS + MEASURE_LOOPS) {
        let mut offset = 0usize;
        while offset + AEC_FRAME_SAMPLES <= speech.len() {
            let render = &speech[offset..offset + AEC_FRAME_SAMPLES];
            assert_eq!(ring.try_push_slice(render), render.len());
            for sample in render {
                history.push_back(sample * 0.55);
            }
            let capture: Vec<f32> = history.drain(..AEC_FRAME_SAMPLES).collect();
            let clean = echo
                .process_capture(&capture)
                .expect("cancel delayed speech");
            if loop_idx >= ADAPT_LOOPS {
                measure_clean.extend(clean);
            }
            offset += AEC_FRAME_SAMPLES;
        }
    }

    let starts = silero_speech_starts(&measure_clean);
    assert_eq!(
        starts, 0,
        "AEC left delayed speech.wav loud enough for {starts} Silero turns"
    );
}

fn load_mono16k_f32(path: &Path) -> Vec<f32> {
    let wav = syllabix_core::audio::read_wav(
        std::fs::File::open(path).unwrap_or_else(|err| panic!("open {}: {err}", path.display())),
    )
    .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    assert_eq!(wav.format.sample_rate_hz, 16_000);
    assert_eq!(wav.format.channels, 1);
    i16_to_f32(&wav.samples)
}

fn silero_speech_starts(samples: &[f32]) -> u64 {
    use syllabix_core::{HttpFetcher, ModelCache, SileroVad, StderrProgress, Vad, VadEvent};

    // Both delayed-fixture tests run in parallel and use the versioned shared
    // cache. Resolve atomically per test, not concurrently, so one test cannot
    // observe the other's in-progress `.part` replacement as a corrupt model.
    static SILERO_CACHE_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> =
        std::sync::OnceLock::new();
    let _guard = SILERO_CACHE_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .expect("Silero cache lock");
    let mut vad = SileroVad::from_cache(
        &ModelCache::v0(),
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )
    .expect("load Silero");
    let mut splitter = FrameSplitter::new();
    let mut starts = 0_u64;
    let mut frames = splitter.push(&f32_to_i16(samples)).expect("frames");
    frames.extend(splitter.flush().expect("flush"));
    for frame in frames {
        starts += vad
            .push_frame(frame)
            .expect("silero")
            .iter()
            .filter(|event| matches!(event, VadEvent::SpeechStart { .. }))
            .count() as u64;
    }
    starts
}

/// Kokoro far-end, 80 ms delayed leak, no microphone.
#[test]
fn delayed_kokoro_fixture_is_cancelled_before_silero() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/aec/kokoro_farend.wav");
    let speech = load_mono16k_f32(&path);
    assert!(
        speech.len() > AEC_FRAME_SAMPLES * 20,
        "kokoro fixture too short"
    );

    const DELAY: usize = AEC_FRAME_SAMPLES * 8;
    const ADAPT_LOOPS: usize = 6;
    const MEASURE_LOOPS: usize = 2;

    let ring = Arc::new(SampleRing::new(AEC_FRAME_SAMPLES * 8));
    let mut echo =
        EchoController::new(EchoReference::new(Arc::clone(&ring), PcmFormat::v0())).expect("echo");
    let mut history = std::collections::VecDeque::from(vec![0.0f32; DELAY]);
    let mut measure_clean = Vec::new();
    for loop_idx in 0..(ADAPT_LOOPS + MEASURE_LOOPS) {
        let mut offset = 0usize;
        while offset + AEC_FRAME_SAMPLES <= speech.len() {
            let render = &speech[offset..offset + AEC_FRAME_SAMPLES];
            assert_eq!(ring.try_push_slice(render), render.len());
            for sample in render {
                history.push_back(sample * 0.55);
            }
            let capture: Vec<f32> = history.drain(..AEC_FRAME_SAMPLES).collect();
            let clean = echo.process_capture(&capture).expect("cancel kokoro echo");
            if loop_idx >= ADAPT_LOOPS {
                measure_clean.extend(clean);
            }
            offset += AEC_FRAME_SAMPLES;
        }
    }
    let starts = silero_speech_starts(&measure_clean);
    assert_eq!(
        starts, 0,
        "AEC left delayed Kokoro loud enough for {starts} Silero turns"
    );
}

/// One-shot: synthesize the versioned Kokoro far-end fixture (loads ONNX; not CI).
#[test]
#[ignore]
fn generate_kokoro_aec_farend_fixture() {
    use syllabix_core::{
        GenerationId, HttpFetcher, KokoroTts, ModelCache, StderrProgress, TokenChunk, Tts, TurnId,
    };

    let mut tts = KokoroTts::from_cache(
        &ModelCache::v0(),
        &HttpFetcher,
        &mut StderrProgress::new(),
        &Cancel::new(),
    )
    .expect("load Kokoro");
    let token = TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index: 0,
        text: "The children played outside in the garden after lunch.".into(),
        is_last: true,
    };
    let chunks = tts
        .synthesize_chunk(&token, &Cancel::new())
        .expect("synthesize Kokoro far-end");
    let mut samples = Vec::new();
    for chunk in chunks {
        samples.extend(i16_to_f32(&chunk.samples));
    }
    assert!(
        samples.len() > 16_000,
        "Kokoro far-end must be longer than one second"
    );
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/aec/kokoro_farend.wav");
    write_mono16k_wav(&path, &samples);
    eprintln!("wrote {} ({} samples)", path.display(), samples.len());
}
