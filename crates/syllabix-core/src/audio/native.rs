//! cpal mic capture and speaker playback.
//!
//! `cpal::Stream` is not `Send` on Linux. The stream lives on a dedicated
//! thread; capture/playback types only hold the sample ring and converters.

use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
#[cfg(not(coverage))]
use std::thread;
use std::thread::JoinHandle;
#[cfg(not(coverage))]
use std::time::{Duration, Instant};

#[cfg(not(coverage))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(not(coverage))]
use cpal::{SampleFormat, Stream, StreamConfig, SupportedStreamConfig, SupportedStreamConfigRange};

use crate::audio::convert::{f32_to_i16, i16_to_f32, FrameSplitter, PcmConverter, PcmFormat};
use crate::audio::devices::DeviceChoice;
#[cfg(not(coverage))]
use crate::audio::devices::{select_input, select_output, DeviceInfo, DeviceInventory};
use crate::audio::echo::{EchoCalibration, EchoController, EchoReference};
#[cfg(not(coverage))]
use crate::audio::ring::device_ring_capacity_samples;
use crate::audio::ring::SampleRing;
use crate::cancel::Cancel;
use crate::error::{Error, Result};
use crate::pipeline::LoopEvent;
use crate::providers::{AudioCapture, AudioSink};
use crate::turn_debug::PlaybackWatch;
use crate::types::{AudioFrame, GenerationId, SynthesizedAudio, TurnId};

/// Tracks the callback after a turn's final samples have entered the device
/// ring. It is deliberately separate from diagnostics: default `run` needs
/// the same drain boundary even when timeline recording is off.
#[cfg(not(coverage))]
#[derive(Clone)]
struct PlaybackDrain {
    inner: Arc<(Mutex<PlaybackDrainState>, Condvar)>,
}

#[cfg(not(coverage))]
#[derive(Default)]
struct PlaybackDrainState {
    final_turn: Option<TurnId>,
    generation: Option<GenerationId>,
    final_enqueued: bool,
    final_enqueued_after_callback: u64,
    callback_count: u64,
    drained_turn: Option<TurnId>,
}

#[cfg(not(coverage))]
impl PlaybackDrain {
    fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(PlaybackDrainState::default()), Condvar::new())),
        }
    }

    fn begin_final(&self, turn: TurnId, generation: GenerationId) {
        let mut state = self.inner.0.lock().expect("playback drain");
        state.final_turn = Some(turn);
        state.generation = Some(generation);
        state.final_enqueued = false;
        state.final_enqueued_after_callback = state.callback_count;
        state.drained_turn = None;
    }

    fn final_enqueued(&self, turn: TurnId) {
        let mut state = self.inner.0.lock().expect("playback drain");
        if state.final_turn == Some(turn) {
            state.final_enqueued = true;
            // A callback can pop silence just before the producer writes the
            // final PCM. Require a later callback so that stale observation
            // cannot release VAD before the new samples are consumed.
            state.final_enqueued_after_callback = state.callback_count;
        }
    }

    fn on_callback(&self, rendered: usize) {
        let mut state = self.inner.0.lock().expect("playback drain");
        state.callback_count += 1;
        if rendered == 0
            && state.final_enqueued
            && state.callback_count > state.final_enqueued_after_callback
        {
            if let Some(turn) = state.final_turn.take() {
                state.final_enqueued = false;
                state.drained_turn = Some(turn);
                self.inner.1.notify_all();
            }
        }
    }

    fn wait_for(&self, turn: TurnId, cancel: &Cancel) -> Result<()> {
        let mut state = self.inner.0.lock().expect("playback drain");
        loop {
            if state.drained_turn == Some(turn) || state.final_turn != Some(turn) {
                return Ok(());
            }
            if cancel.is_shutdown()
                || state
                    .generation
                    .is_some_and(|generation| cancel.is_stale(generation))
            {
                return Err(Error::Cancelled);
            }
            let (next, _) = self
                .inner
                .1
                .wait_timeout(state, Duration::from_millis(5))
                .expect("playback drain");
            state = next;
        }
    }

    fn clear(&self) {
        let mut state = self.inner.0.lock().expect("playback drain");
        state.final_turn = None;
        state.generation = None;
        state.final_enqueued = false;
        state.final_enqueued_after_callback = state.callback_count;
        self.inner.1.notify_all();
    }
}

#[cfg(coverage)]
#[derive(Clone, Default)]
struct PlaybackDrain;

#[cfg(coverage)]
impl PlaybackDrain {
    #[cfg(test)]
    fn new() -> Self {
        Self
    }

    fn begin_final(&self, _turn: TurnId, _generation: GenerationId) {}

    fn final_enqueued(&self, _turn: TurnId) {}

    fn wait_for(&self, _turn: TurnId, _cancel: &Cancel) -> Result<()> {
        Ok(())
    }

    fn clear(&self) {}
}

/// Backend identifier used in logs.
pub fn backend_name() -> &'static str {
    "cpal"
}

/// One selected cpal device and the stream config we will open.
#[cfg(not(coverage))]
pub struct SelectedCpalDevice {
    /// User-facing name.
    pub name: String,
    /// Why this device was chosen.
    pub reason: &'static str,
    /// Pipeline-facing format after device conversion.
    pub device_format: PcmFormat,
    sample_format: SampleFormat,
    device: cpal::Device,
    config: StreamConfig,
}

#[cfg(coverage)]
pub struct SelectedCpalDevice;

#[cfg(not(coverage))]
struct CpalInventory {
    host: cpal::Host,
}

#[cfg(not(coverage))]
impl CpalInventory {
    fn new() -> Self {
        Self {
            host: cpal::default_host(),
        }
    }

    fn info_from_device(
        device: &cpal::Device,
        default: Option<SupportedStreamConfig>,
    ) -> Option<DeviceInfo> {
        let name = device.name().ok()?;
        let cfg = default?;
        Some(DeviceInfo {
            name,
            sample_rate_hz: cfg.sample_rate().0,
            channels: cfg.channels(),
        })
    }
}

#[cfg(not(coverage))]
impl DeviceInventory for CpalInventory {
    fn default_input(&self) -> Option<DeviceInfo> {
        let device = self.host.default_input_device()?;
        let cfg = device.default_input_config().ok();
        Self::info_from_device(&device, cfg)
    }

    fn default_output(&self) -> Option<DeviceInfo> {
        let device = self.host.default_output_device()?;
        let cfg = device.default_output_config().ok();
        Self::info_from_device(&device, cfg)
    }

    fn inputs(&self) -> Vec<DeviceInfo> {
        let Ok(devices) = self.host.input_devices() else {
            return Vec::new();
        };
        devices
            .filter_map(|d| {
                let cfg = d.default_input_config().ok();
                Self::info_from_device(&d, cfg)
            })
            .collect()
    }

    fn outputs(&self) -> Vec<DeviceInfo> {
        let Ok(devices) = self.host.output_devices() else {
            return Vec::new();
        };
        devices
            .filter_map(|d| {
                let cfg = d.default_output_config().ok();
                Self::info_from_device(&d, cfg)
            })
            .collect()
    }
}

#[cfg(not(coverage))]
const CPAL_RECOVERY_HINT: &str =
    "Close other apps using the device, then retry. If this is a laptop, check the OS sound panel and microphone permission.";

#[cfg(not(coverage))]
fn map_cpal(err: impl std::fmt::Display, what: &str) -> Error {
    Error::AudioDevice {
        message: format!(
            "{what}: {}. {CPAL_RECOVERY_HINT}",
            err.to_string().trim_end_matches('.')
        ),
    }
}

/// Backend text from a [`map_cpal`] message, without the recovery hint.
#[cfg(not(coverage))]
fn audio_device_backend_text(message: &str) -> Option<&str> {
    let head = message.split(". Close other apps").next()?;
    head.split_once(": ").map(|(_, backend)| backend)
}

/// True when the cpal/backend message is permission, access, or device-in-use.
#[cfg(not(coverage))]
fn cpal_backend_text_is_fatal(backend: &str) -> bool {
    let lower = backend.to_ascii_lowercase();
    lower.contains("permission denied")
        || lower.contains("denied permission")
        || lower.contains("access denied")
        || lower.contains("access is denied")
        || lower.contains("device busy")
        || lower.contains("device is busy")
        || lower.contains("already in use")
        || lower.contains(" in use")
        || lower.ends_with("in use")
}

/// Permission, access, and device-in-use failures are not retried.
#[cfg(not(coverage))]
fn open_failure_is_fatal(err: &Error) -> bool {
    let Error::AudioDevice { message } = err else {
        return true;
    };
    let Some(backend) = audio_device_backend_text(message) else {
        return false;
    };
    cpal_backend_text_is_fatal(backend)
}

/// How long a device open keeps retrying transient failures before it reports failure.
#[cfg(not(coverage))]
const OPEN_RETRY_WINDOW: Duration = Duration::from_secs(3);

/// Pause between device open attempts.
#[cfg(not(coverage))]
const OPEN_RETRY_DELAY: Duration = Duration::from_millis(100);

/// Run `attempt` until it succeeds or `window` elapses, then return its error.
///
/// Only retryable failures sleep between attempts, so a healthy device opens on
/// the first try with no added latency.
#[cfg(not(coverage))]
fn open_with_retries<T>(
    what: &str,
    window: Duration,
    delay: Duration,
    mut attempt: impl FnMut() -> Result<T>,
) -> Result<T> {
    let started = Instant::now();
    let mut retries = 0u32;
    loop {
        match attempt() {
            Ok(opened) => {
                if retries > 0 {
                    eprintln!("syllabix: {what} opened after {retries} retries");
                }
                return Ok(opened);
            }
            Err(err) => {
                if open_failure_is_fatal(&err) || started.elapsed() >= window {
                    return Err(err);
                }
                retries += 1;
                eprintln!("syllabix: {what} did not open yet (attempt {retries}); retrying");
                thread::sleep(delay);
            }
        }
    }
}

/// Sample rate to request from a supported range: standard rates first.
#[cfg(not(coverage))]
fn preferred_sample_rate(range: &SupportedStreamConfigRange) -> cpal::SampleRate {
    let min = range.min_sample_rate().0;
    let max = range.max_sample_rate().0;
    let rate = if (min..=max).contains(&48_000) {
        48_000
    } else if (min..=max).contains(&44_100) {
        44_100
    } else {
        max
    };
    cpal::SampleRate(rate)
}

/// Sort key: float samples first, 16-bit integer last, then rate and channels.
#[cfg(not(coverage))]
fn config_rank(cfg: &SupportedStreamConfig) -> (u8, u32, u8) {
    let format = match cfg.sample_format() {
        SampleFormat::F32 => 0,
        SampleFormat::I16 => 2,
        _ => 1,
    };
    let rate = match cfg.sample_rate().0 {
        48_000 => 0,
        44_100 => 1,
        rate => rate.abs_diff(48_000),
    };
    let channels = match cfg.channels() {
        2 => 0,
        1 => 1,
        _ => 2,
    };
    (format, rate, channels)
}

/// The device default first, then every other format it advertises.
#[cfg(not(coverage))]
fn config_candidates(
    default: Option<SupportedStreamConfig>,
    ranges: impl IntoIterator<Item = SupportedStreamConfigRange>,
) -> Vec<SupportedStreamConfig> {
    let mut extras: Vec<SupportedStreamConfig> = ranges
        .into_iter()
        .map(|range| range.with_sample_rate(preferred_sample_rate(&range)))
        .collect();
    extras.sort_by_key(config_rank);
    let mut configs = Vec::new();
    if let Some(default) = default {
        configs.push(default);
    }
    for cfg in extras {
        if !configs.contains(&cfg) {
            configs.push(cfg);
        }
    }
    configs
}

/// Build the open candidates for one device, default config first.
#[cfg(not(coverage))]
fn open_candidates(
    device: cpal::Device,
    choice: &DeviceChoice,
    default: Option<SupportedStreamConfig>,
    ranges: impl IntoIterator<Item = SupportedStreamConfigRange>,
    what: &str,
) -> Result<Vec<SelectedCpalDevice>> {
    let configs = config_candidates(default, ranges);
    if configs.is_empty() {
        return Err(Error::AudioDevice {
            message: format!(
                "{what} '{}' reported no usable stream format. Reconnect it and retry.",
                choice.device.name
            ),
        });
    }
    Ok(configs
        .into_iter()
        .map(|cfg| {
            let (device_format, sample_format, config) = supported_to_config(cfg);
            SelectedCpalDevice {
                name: choice.device.name.clone(),
                reason: choice.reason,
                device_format,
                sample_format,
                device: device.clone(),
                config,
            }
        })
        .collect())
}

#[cfg(not(coverage))]
fn supported_to_config(cfg: SupportedStreamConfig) -> (PcmFormat, SampleFormat, StreamConfig) {
    let sample_format = cfg.sample_format();
    let device_format = PcmFormat {
        sample_rate_hz: cfg.sample_rate().0,
        channels: cfg.channels(),
    };
    let config: StreamConfig = cfg.into();
    (device_format, sample_format, config)
}

#[cfg(not(coverage))]
fn open_input(choice: &DeviceChoice, host: &cpal::Host) -> Result<Vec<SelectedCpalDevice>> {
    let device = find_input(host, &choice.device.name)?;
    let supported = device.supported_input_configs().ok();
    open_candidates(
        device.clone(),
        choice,
        device.default_input_config().ok(),
        supported.into_iter().flatten(),
        "Microphone",
    )
}

#[cfg(not(coverage))]
fn open_output(choice: &DeviceChoice, host: &cpal::Host) -> Result<Vec<SelectedCpalDevice>> {
    let device = find_output(host, &choice.device.name)?;
    let supported = device.supported_output_configs().ok();
    open_candidates(
        device.clone(),
        choice,
        device.default_output_config().ok(),
        supported.into_iter().flatten(),
        "Speakers",
    )
}

#[cfg(not(coverage))]
fn find_input(host: &cpal::Host, name: &str) -> Result<cpal::Device> {
    if let Some(dev) = host.default_input_device() {
        if dev.name().ok().as_deref() == Some(name) {
            return Ok(dev);
        }
    }
    let devices = host
        .input_devices()
        .map_err(|e| map_cpal(e, "Could not list microphones"))?;
    for dev in devices {
        if dev.name().ok().as_deref() == Some(name) {
            return Ok(dev);
        }
    }
    Err(Error::AudioDevice {
        message: format!(
            "Microphone '{name}' disappeared before it could be opened. Reconnect it and retry."
        ),
    })
}

#[cfg(not(coverage))]
fn find_output(host: &cpal::Host, name: &str) -> Result<cpal::Device> {
    if let Some(dev) = host.default_output_device() {
        if dev.name().ok().as_deref() == Some(name) {
            return Ok(dev);
        }
    }
    let devices = host
        .output_devices()
        .map_err(|e| map_cpal(e, "Could not list speakers"))?;
    for dev in devices {
        if dev.name().ok().as_deref() == Some(name) {
            return Ok(dev);
        }
    }
    Err(Error::AudioDevice {
        message: format!(
            "Speakers '{name}' disappeared before they could be opened. Reconnect them and retry."
        ),
    })
}

/// One successfully opened capture stream and its ring.
#[cfg(not(coverage))]
struct OpenedInputStream {
    stream: Stream,
    ring: Arc<SampleRing>,
    name: String,
    device_format: PcmFormat,
}

/// One successfully opened playback stream, its ring, and the AEC tap ring.
#[cfg(not(coverage))]
struct OpenedOutputStream {
    stream: Stream,
    ring: Arc<SampleRing>,
    echo_ring: Option<Arc<SampleRing>>,
    name: String,
    device_format: PcmFormat,
}

/// Try every candidate config once, returning the first stream that starts.
#[cfg(not(coverage))]
fn open_input_stream(candidates: &[SelectedCpalDevice]) -> Result<OpenedInputStream> {
    let mut last_error = None;
    for selected in candidates {
        let cap = device_ring_capacity_samples(
            selected.device_format.sample_rate_hz,
            selected.device_format.channels,
        )
        .max(256);
        let ring = Arc::new(SampleRing::new(cap));
        let stream = match build_input_stream(selected, Arc::clone(&ring)) {
            Ok(stream) => stream,
            Err(err) => {
                last_error = Some(err);
                continue;
            }
        };
        if let Err(err) = stream.play() {
            last_error = Some(map_cpal(err, "Could not start the microphone"));
            continue;
        }
        return Ok(OpenedInputStream {
            stream,
            ring,
            name: selected.name.clone(),
            device_format: selected.device_format,
        });
    }
    Err(last_error.unwrap_or_else(|| no_open_attempts("microphone")))
}

/// [`open_input_stream`], for the speaker side, with the AEC render tap.
#[cfg(not(coverage))]
fn open_output_stream(
    candidates: &[SelectedCpalDevice],
    tap_render: bool,
    watch: Option<PlaybackWatch>,
    drain: PlaybackDrain,
) -> Result<OpenedOutputStream> {
    let mut last_error = None;
    for selected in candidates {
        let cap = device_ring_capacity_samples(
            selected.device_format.sample_rate_hz,
            selected.device_format.channels,
        )
        .max(256);
        let ring = Arc::new(SampleRing::new(cap));
        let echo_ring = tap_render.then(|| Arc::new(SampleRing::new(cap)));
        let stream = match build_output_stream(
            selected,
            Arc::clone(&ring),
            echo_ring.as_ref().map(Arc::clone),
            watch.clone(),
            drain.clone(),
        ) {
            Ok(stream) => stream,
            Err(err) => {
                last_error = Some(err);
                continue;
            }
        };
        if let Err(err) = stream.play() {
            last_error = Some(map_cpal(err, "Could not start the speakers"));
            continue;
        }
        return Ok(OpenedOutputStream {
            stream,
            ring,
            echo_ring,
            name: selected.name.clone(),
            device_format: selected.device_format,
        });
    }
    Err(last_error.unwrap_or_else(|| no_open_attempts("speakers")))
}

/// Retry `preferred` until `window` elapses; on transient failure with a fallback, run it once.
#[cfg(not(coverage))]
fn open_preferred_then_fallback<T>(
    what: &str,
    window: Duration,
    delay: Duration,
    has_fallback: bool,
    mut preferred: impl FnMut() -> Result<T>,
    fallback: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match open_with_retries(what, window, delay, &mut preferred) {
        Ok(opened) => Ok(opened),
        Err(retry_err) => {
            if open_failure_is_fatal(&retry_err) || !has_fallback {
                return Err(retry_err);
            }
            fallback()
        }
    }
}

/// Retry the preferred config, then try other formats once after a transient timeout.
#[cfg(not(coverage))]
fn open_input_preferred_then_fallback(
    candidates: &[SelectedCpalDevice],
) -> Result<OpenedInputStream> {
    open_preferred_then_fallback(
        "microphone",
        OPEN_RETRY_WINDOW,
        OPEN_RETRY_DELAY,
        candidates.len() > 1,
        || open_input_stream(&candidates[0..1]),
        || open_input_stream(&candidates[1..]),
    )
}

/// [`open_input_preferred_then_fallback`], for speakers.
#[cfg(not(coverage))]
fn open_output_preferred_then_fallback(
    candidates: &[SelectedCpalDevice],
    tap_render: bool,
    watch: Option<PlaybackWatch>,
    drain: PlaybackDrain,
) -> Result<OpenedOutputStream> {
    open_preferred_then_fallback(
        "speakers",
        OPEN_RETRY_WINDOW,
        OPEN_RETRY_DELAY,
        candidates.len() > 1,
        || open_output_stream(&candidates[0..1], tap_render, watch.clone(), drain.clone()),
        || open_output_stream(&candidates[1..], tap_render, watch.clone(), drain.clone()),
    )
}

#[cfg(not(coverage))]
fn no_open_attempts(what: &str) -> Error {
    Error::AudioDevice {
        message: format!("Could not open the {what}: no stream format was tried."),
    }
}

/// Owns a cpal stream on a thread so [`NativeCapture`] / [`NativePlayback`] stay `Send`.
struct StreamWorker {
    thread: Option<JoinHandle<()>>,
    stop: Arc<(Mutex<bool>, Condvar)>,
}

impl Drop for StreamWorker {
    fn drop(&mut self) {
        if let Ok(mut done) = self.stop.0.lock() {
            *done = true;
            self.stop.1.notify_all();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(not(coverage))]
struct OpenedStream {
    worker: StreamWorker,
    name: String,
    device_format: PcmFormat,
    ring: Arc<SampleRing>,
    echo_ring: Option<Arc<SampleRing>>,
}

#[cfg(not(coverage))]
type OpenedStreamMetadata = (String, PcmFormat, Arc<SampleRing>, Option<Arc<SampleRing>>);

#[cfg(not(coverage))]
fn spawn_input(choice: DeviceChoice) -> Result<OpenedStream> {
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let stop = Arc::new((Mutex::new(false), Condvar::new()));
    let stop_thread = Arc::clone(&stop);
    let thread = thread::Builder::new()
        .name("syllabix-mic".into())
        .spawn(move || {
            let host = cpal::default_host();
            let selected = match open_input(&choice, &host) {
                Ok(s) => s,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let opened = match open_input_preferred_then_fallback(&selected) {
                Ok(opened) => opened,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let meta = (
                opened.name.clone(),
                opened.device_format,
                Arc::clone(&opened.ring),
                None,
            );
            let _ = ready_tx.send(Ok(meta));
            let (lock, cv) = &*stop_thread;
            let mut done = lock.lock().expect("stream stop");
            while !*done {
                done = cv.wait(done).expect("stream stop");
            }
            drop(opened.stream);
        })
        .map_err(|err| Error::AudioDevice {
            message: format!("Could not start the microphone thread: {err}"),
        })?;
    let (name, device_format, ring, echo_ring) = recv_open(ready_rx)?;
    Ok(OpenedStream {
        worker: StreamWorker {
            thread: Some(thread),
            stop,
        },
        name,
        device_format,
        ring,
        echo_ring,
    })
}

#[cfg(not(coverage))]
fn spawn_output(
    choice: DeviceChoice,
    tap_render: bool,
    watch: Option<PlaybackWatch>,
    drain: PlaybackDrain,
) -> Result<OpenedStream> {
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let stop = Arc::new((Mutex::new(false), Condvar::new()));
    let stop_thread = Arc::clone(&stop);
    let thread = thread::Builder::new()
        .name("syllabix-spk".into())
        .spawn(move || {
            let host = cpal::default_host();
            let selected = match open_output(&choice, &host) {
                Ok(s) => s,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let opened = match open_output_preferred_then_fallback(
                &selected,
                tap_render,
                watch.clone(),
                drain.clone(),
            ) {
                Ok(opened) => opened,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let meta = (
                opened.name.clone(),
                opened.device_format,
                Arc::clone(&opened.ring),
                opened.echo_ring.as_ref().map(Arc::clone),
            );
            let _ = ready_tx.send(Ok(meta));
            let (lock, cv) = &*stop_thread;
            let mut done = lock.lock().expect("stream stop");
            while !*done {
                done = cv.wait(done).expect("stream stop");
            }
            drop(opened.stream);
        })
        .map_err(|err| Error::AudioDevice {
            message: format!("Could not start the speaker thread: {err}"),
        })?;
    let (name, device_format, ring, echo_ring) = recv_open(ready_rx)?;
    Ok(OpenedStream {
        worker: StreamWorker {
            thread: Some(thread),
            stop,
        },
        name,
        device_format,
        ring,
        echo_ring,
    })
}

#[cfg(not(coverage))]
fn recv_open(rx: mpsc::Receiver<Result<OpenedStreamMetadata>>) -> Result<OpenedStreamMetadata> {
    rx.recv().map_err(|_| Error::AudioDevice {
        message: "Audio thread exited before the device finished opening.".into(),
    })?
}

/// Convert live microphone samples into pipeline [`AudioFrame`]s.
pub struct NativeCapture {
    _worker: StreamWorker,
    ring: Arc<SampleRing>,
    conv: PcmConverter,
    echo: Option<EchoController>,
    echo_status: EchoCalibration,
    aec_events: Option<mpsc::Sender<LoopEvent>>,
    split: FrameSplitter,
    pending: Vec<AudioFrame>,
    pcm_tap: bool,
    capture_leftover: Vec<i16>,
    /// Selected device name.
    pub device_name: String,
    /// Device PCM layout before conversion.
    pub device_format: PcmFormat,
}

impl NativeCapture {
    /// Open the selected default (or first usable) input device.
    #[cfg(not(coverage))]
    pub fn open() -> Result<Self> {
        Self::open_inner(None, None)
    }

    /// Open the microphone with full-duplex AEC fed by the speaker callback.
    #[cfg(not(coverage))]
    pub fn open_with_echo(reference: EchoReference) -> Result<Self> {
        Self::open_inner(Some(reference), None)
    }

    /// Open with AEC and report live calibration changes to the TUI.
    #[cfg(not(coverage))]
    pub fn open_with_echo_and_events(
        reference: EchoReference,
        events: Option<mpsc::Sender<LoopEvent>>,
    ) -> Result<Self> {
        Self::open_inner(Some(reference), events)
    }

    #[cfg(not(coverage))]
    fn open_inner(
        reference: Option<EchoReference>,
        aec_events: Option<mpsc::Sender<LoopEvent>>,
    ) -> Result<Self> {
        let inv = CpalInventory::new();
        let choice = select_input(&inv)?;
        let opened = spawn_input(choice)?;
        let echo = match reference {
            Some(reference) => {
                let mut echo = EchoController::new(reference)?;
                echo.start_render_pump();
                Some(echo)
            }
            None => None,
        };
        Ok(Self {
            conv: PcmConverter::new(opened.device_format, PcmFormat::v0())?,
            echo,
            echo_status: EchoCalibration::WaitingForPlayback,
            aec_events,
            split: FrameSplitter::new(),
            pending: Vec::new(),
            pcm_tap: false,
            capture_leftover: Vec::new(),
            device_name: opened.name,
            device_format: opened.device_format,
            ring: opened.ring,
            _worker: opened.worker,
        })
    }

    /// Live AEC controller, if this capture was opened with a speaker tap.
    pub fn echo_mut(&mut self) -> Option<&mut EchoController> {
        self.echo.as_mut()
    }

    /// Drop anything the device captured before the pipeline starts reading.
    ///
    /// Capture is opened before the model load, so the microphone keeps recording
    /// while weights page in. Those seconds are stale and would otherwise open a
    /// turn on speech from before the app was ready.
    pub fn discard_buffered_audio(&mut self) {
        AudioCapture::discard_buffered_audio(self);
    }

    /// Align pre-AEC PCM with each clean frame for the diagnostics WAVs.
    pub fn enable_pcm_tap(&mut self) {
        AudioCapture::enable_pcm_tap(self);
    }

    fn attach_capture_pcm(
        &mut self,
        mut frames: Vec<AudioFrame>,
        capture: &[i16],
    ) -> Vec<AudioFrame> {
        if !self.pcm_tap {
            return frames;
        }
        self.capture_leftover.extend_from_slice(capture);
        for frame in &mut frames {
            let n = frame.samples.len();
            if self.capture_leftover.len() >= n {
                frame.capture_pcm = Some(self.capture_leftover.drain(..n).collect());
            } else {
                frame.capture_pcm = Some(frame.samples.clone());
            }
        }
        frames
    }
}

impl Drop for NativeCapture {
    fn drop(&mut self) {
        self.ring.close();
    }
}

impl AudioCapture for NativeCapture {
    fn name(&self) -> &'static str {
        backend_name()
    }

    fn discard_buffered_audio(&mut self) {
        self.pending.clear();
        self.capture_leftover.clear();
        self.ring.clear();
    }

    fn enable_pcm_tap(&mut self) {
        self.pcm_tap = true;
        if let Some(echo) = &mut self.echo {
            echo.enable_debug_tap();
        }
    }

    fn next_frame(&mut self, cancel: &Cancel) -> Result<Option<AudioFrame>> {
        loop {
            if cancel.is_shutdown() {
                return Err(Error::Cancelled);
            }
            if !self.pending.is_empty() {
                return Ok(Some(self.pending.remove(0)));
            }
            let mut buf = [0.0f32; 2048];
            let n = self.ring.pop_slice_cancellable(&mut buf, cancel)?;
            if n == 0 {
                return Ok(None);
            }
            let converted = self.conv.push(&buf[..n]);
            let (clean, capture_i16) = if let Some(echo) = &mut self.echo {
                let clean = echo.process_capture(&converted)?;
                let status = echo.calibration();
                if status != self.echo_status {
                    if let Some(events) = &self.aec_events {
                        let (active, restart_required) = match status {
                            EchoCalibration::WaitingForPlayback | EchoCalibration::Calibrating => {
                                (false, false)
                            }
                            EchoCalibration::Active => (true, false),
                            EchoCalibration::Degraded => (false, true),
                        };
                        let _ = events.send(LoopEvent::Aec {
                            active,
                            restart_required,
                        });
                    }
                    self.echo_status = status;
                }
                let capture_i16 = if self.pcm_tap {
                    let (pre, post) = echo.take_debug_tap();
                    if pre.len() == post.len() && post.len() == clean.len() {
                        f32_to_i16(&pre)
                    } else {
                        f32_to_i16(&clean)
                    }
                } else {
                    Vec::new()
                };
                (clean, capture_i16)
            } else {
                let clean = converted;
                let capture_i16 = if self.pcm_tap {
                    f32_to_i16(&clean)
                } else {
                    Vec::new()
                };
                (clean, capture_i16)
            };
            let i16s = f32_to_i16(&clean);
            let frames = self.split.push(&i16s)?;
            self.pending = self.attach_capture_pcm(frames, &capture_i16);
        }
    }
}

/// Live speakers. Implements [`AudioSink`].
pub struct NativePlayback {
    _worker: StreamWorker,
    ring: Arc<SampleRing>,
    echo_ring: Option<Arc<SampleRing>>,
    conv: PcmConverter,
    watch: Option<PlaybackWatch>,
    drain: PlaybackDrain,
    /// Selected device name.
    pub device_name: String,
    /// Device PCM layout after conversion.
    pub device_format: PcmFormat,
}

impl NativePlayback {
    /// Open the selected default (or first usable) output device.
    #[cfg(not(coverage))]
    pub fn open() -> Result<Self> {
        Self::open_inner(false, None).map(|(playback, _)| playback)
    }

    /// Open speakers and tap the exact rendered PCM for full-duplex AEC.
    #[cfg(not(coverage))]
    pub fn open_with_echo() -> Result<(Self, EchoReference)> {
        Self::open_with_echo_and_watch(None)
    }

    /// [`Self::open_with_echo`] plus timeline anchors for the speaker callback
    /// (`playback_first` / `playback_done` in the diagnostics sidecar).
    #[cfg(not(coverage))]
    pub fn open_with_echo_and_watch(watch: Option<PlaybackWatch>) -> Result<(Self, EchoReference)> {
        let (playback, reference) = Self::open_inner(true, watch)?;
        Ok((
            playback,
            reference.expect("render tap requested but not constructed"),
        ))
    }

    #[cfg(not(coverage))]
    fn open_inner(
        tap_render: bool,
        watch: Option<PlaybackWatch>,
    ) -> Result<(Self, Option<EchoReference>)> {
        let inv = CpalInventory::new();
        let choice = select_output(&inv)?;
        let drain = PlaybackDrain::new();
        let opened = spawn_output(choice, tap_render, watch.clone(), drain.clone())?;
        let reference = opened
            .echo_ring
            .as_ref()
            .map(|ring| EchoReference::new(Arc::clone(ring), opened.device_format));
        Ok((
            Self {
                conv: PcmConverter::new(PcmFormat::v0(), opened.device_format)?,
                device_name: opened.name,
                device_format: opened.device_format,
                ring: opened.ring,
                echo_ring: opened.echo_ring,
                watch,
                drain,
                _worker: opened.worker,
            },
            reference,
        ))
    }
}

impl Drop for NativePlayback {
    fn drop(&mut self) {
        self.ring.close();
        if let Some(ring) = &self.echo_ring {
            ring.close();
        }
    }
}

impl AudioSink for NativePlayback {
    fn play(&mut self, audio: SynthesizedAudio, cancel: &Cancel) -> Result<()> {
        if cancel.is_shutdown() {
            return Err(Error::Cancelled);
        }
        if cancel.is_stale(audio.generation) {
            self.interrupt();
            return Ok(());
        }
        if let Some(watch) = &self.watch {
            watch.begin_turn(audio.turn);
        }
        if audio.is_last {
            self.drain.begin_final(audio.turn, audio.generation);
        }
        let f32s = i16_to_f32(&audio.samples);
        let mut device_pcm = self.conv.push(&f32s);
        if audio.is_last {
            device_pcm.extend(self.conv.flush());
        }
        let result = match self
            .ring
            .push_slice_cancellable(&device_pcm, cancel, audio.generation)
        {
            Ok(()) => Ok(()),
            Err(Error::Cancelled) if !cancel.is_shutdown() => {
                self.interrupt();
                Ok(())
            }
            Err(err) => Err(err),
        };
        if result.is_ok() && audio.is_last {
            self.drain.final_enqueued(audio.turn);
        }
        result
    }

    fn finish_turn(&mut self, turn: TurnId, cancel: &Cancel) -> Result<()> {
        self.drain.wait_for(turn, cancel)
    }

    fn interrupt(&mut self) {
        self.ring.clear();
        self.conv.reset();
        self.drain.clear();
    }
}

#[cfg(not(coverage))]
fn build_input_stream(selected: &SelectedCpalDevice, ring: Arc<SampleRing>) -> Result<Stream> {
    let err_fn = |err| {
        eprintln!("syllabix microphone error: {err}");
    };
    match selected.sample_format {
        SampleFormat::F32 => selected
            .device
            .build_input_stream(
                &selected.config,
                move |data: &[f32], _| {
                    let _ = ring.try_push_slice(data);
                },
                err_fn,
                None,
            )
            .map_err(|e| map_cpal(e, "Could not open the microphone")),
        SampleFormat::I16 => selected
            .device
            .build_input_stream(
                &selected.config,
                move |data: &[i16], _| {
                    let converted = i16_to_f32(data);
                    let _ = ring.try_push_slice(&converted);
                },
                err_fn,
                None,
            )
            .map_err(|e| map_cpal(e, "Could not open the microphone")),
        SampleFormat::U16 => selected
            .device
            .build_input_stream(
                &selected.config,
                move |data: &[u16], _| {
                    let converted: Vec<f32> = data
                        .iter()
                        .map(|s| (f32::from(*s) / f32::from(u16::MAX)) * 2.0 - 1.0)
                        .collect();
                    let _ = ring.try_push_slice(&converted);
                },
                err_fn,
                None,
            )
            .map_err(|e| map_cpal(e, "Could not open the microphone")),
        other => Err(Error::AudioDevice {
            message: format!(
                "Microphone sample format {other} is not supported. Connect a standard input device."
            ),
        }),
    }
}

#[cfg(not(coverage))]
fn build_output_stream(
    selected: &SelectedCpalDevice,
    ring: Arc<SampleRing>,
    echo_ring: Option<Arc<SampleRing>>,
    watch: Option<PlaybackWatch>,
    drain: PlaybackDrain,
) -> Result<Stream> {
    let err_fn = |err| {
        eprintln!("syllabix speaker error: {err}");
    };
    match selected.sample_format {
        SampleFormat::F32 => selected
            .device
            .build_output_stream(
                &selected.config,
                move |data: &mut [f32], _| {
                    let rendered = ring.try_pop_slice(data);
                    if let Some(watch) = &watch {
                        watch.on_callback(rendered);
                    }
                    drain.on_callback(rendered);
                    if let Some(reference) = &echo_ring {
                        // The device renders the whole buffer (zeros included);
                        // the AEC reference must see exactly that.
                        let _ = reference.try_push_slice(data);
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| map_cpal(e, "Could not open the speakers")),
        SampleFormat::I16 => selected
            .device
            .build_output_stream(
                &selected.config,
                move |data: &mut [i16], _| {
                    let mut f = vec![0.0f32; data.len()];
                    let rendered = ring.try_pop_slice(&mut f);
                    if let Some(watch) = &watch {
                        watch.on_callback(rendered);
                    }
                    drain.on_callback(rendered);
                    let i = f32_to_i16(&f);
                    data.copy_from_slice(&i);
                    if let Some(reference) = &echo_ring {
                        let _ = reference.try_push_slice(&f);
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| map_cpal(e, "Could not open the speakers")),
        SampleFormat::U16 => selected
            .device
            .build_output_stream(
                &selected.config,
                move |data: &mut [u16], _| {
                    let mut f = vec![0.0f32; data.len()];
                    let rendered = ring.try_pop_slice(&mut f);
                    if let Some(watch) = &watch {
                        watch.on_callback(rendered);
                    }
                    drain.on_callback(rendered);
                    for (slot, sample) in data.iter_mut().zip(f.iter()) {
                        let scaled = (sample.clamp(-1.0, 1.0) + 1.0) * 0.5 * f32::from(u16::MAX);
                        *slot = scaled.round() as u16;
                    }
                    if let Some(reference) = &echo_ring {
                        let _ = reference.try_push_slice(&f);
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| map_cpal(e, "Could not open the speakers")),
        other => Err(Error::AudioDevice {
            message: format!(
                "Speaker sample format {other} is not supported. Connect a standard output device."
            ),
        }),
    }
}

/// Probe whether the host can name a default input and output (no stream).
#[cfg(not(coverage))]
pub fn probe_device_names() -> Result<(DeviceChoice, DeviceChoice)> {
    let inv = CpalInventory::new();
    Ok((select_input(&inv)?, select_output(&inv)?))
}

#[cfg(coverage)]
fn coverage_audio_unavailable() -> Error {
    Error::AudioDevice {
        message: "Native audio devices are unavailable in coverage tests.".into(),
    }
}

#[cfg(coverage)]
impl NativeCapture {
    pub fn open() -> Result<Self> {
        Err(coverage_audio_unavailable())
    }

    pub fn open_with_echo(_reference: EchoReference) -> Result<Self> {
        Err(coverage_audio_unavailable())
    }

    pub fn open_with_echo_and_events(
        _reference: EchoReference,
        _events: Option<mpsc::Sender<LoopEvent>>,
    ) -> Result<Self> {
        Err(coverage_audio_unavailable())
    }
}

#[cfg(coverage)]
impl NativePlayback {
    pub fn open() -> Result<Self> {
        Err(coverage_audio_unavailable())
    }

    pub fn open_with_echo() -> Result<(Self, EchoReference)> {
        Err(coverage_audio_unavailable())
    }

    pub fn open_with_echo_and_watch(
        _watch: Option<PlaybackWatch>,
    ) -> Result<(Self, EchoReference)> {
        Err(coverage_audio_unavailable())
    }
}

#[cfg(coverage)]
pub fn probe_device_names() -> Result<(DeviceChoice, DeviceChoice)> {
    Err(coverage_audio_unavailable())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn idle_worker() -> StreamWorker {
        StreamWorker {
            thread: None,
            stop: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    fn test_capture() -> NativeCapture {
        let device_format = PcmFormat::v0();
        NativeCapture {
            _worker: idle_worker(),
            ring: Arc::new(SampleRing::new(4_096)),
            conv: PcmConverter::new(device_format, PcmFormat::v0()).expect("v0 converter"),
            echo: None,
            echo_status: EchoCalibration::WaitingForPlayback,
            aec_events: None,
            split: FrameSplitter::new(),
            pending: Vec::new(),
            pcm_tap: false,
            capture_leftover: Vec::new(),
            device_name: "test microphone".into(),
            device_format,
        }
    }

    fn test_playback() -> NativePlayback {
        let device_format = PcmFormat::v0();
        NativePlayback {
            _worker: idle_worker(),
            ring: Arc::new(SampleRing::new(4_096)),
            echo_ring: None,
            conv: PcmConverter::new(PcmFormat::v0(), device_format).expect("v0 converter"),
            watch: None,
            drain: PlaybackDrain::new(),
            device_name: "test speakers".into(),
            device_format,
        }
    }

    #[test]
    fn backend_is_cpal() {
        assert_eq!(backend_name(), "cpal");
    }

    #[test]
    #[cfg(not(coverage))]
    fn cpal_errors_include_recovery_guidance() {
        let error = map_cpal("device busy", "Could not open microphone");
        let Error::AudioDevice { message } = error else {
            panic!("expected an audio device error");
        };
        assert!(message.contains("device busy"));
        assert!(message.contains("Close other apps"));
        assert!(message.contains("microphone permission"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn cpal_errors_do_not_double_the_full_stop() {
        let error = map_cpal(
            "The requested stream configuration is not supported by the device.",
            "Could not open the speakers",
        );
        let Error::AudioDevice { message } = error else {
            panic!("expected an audio device error");
        };
        assert!(
            message.contains("by the device. Close other apps"),
            "{message}"
        );
        assert!(!message.contains("device.."), "{message}");
    }

    #[test]
    #[cfg(not(coverage))]
    fn open_with_retries_succeeds_immediately_without_sleeping() {
        let started = Instant::now();
        let opened =
            open_with_retries("test device", OPEN_RETRY_WINDOW, OPEN_RETRY_DELAY, || Ok(7))
                .expect("first attempt succeeds");
        assert_eq!(opened, 7);
        assert!(
            started.elapsed() < Duration::from_millis(5),
            "elapsed {:?}",
            started.elapsed()
        );
    }

    #[test]
    #[cfg(not(coverage))]
    fn open_with_retries_does_not_retry_fatal_errors() {
        let mut attempts = 0;
        let error = open_with_retries(
            "test microphone",
            Duration::from_secs(1),
            Duration::from_millis(1),
            || -> Result<()> {
                attempts += 1;
                Err(map_cpal("device busy", "Could not open the microphone"))
            },
        )
        .expect_err("device busy is fatal");
        assert_eq!(attempts, 1, "fatal open errors fail fast");
        let Error::AudioDevice { message } = error else {
            panic!("expected audio device error");
        };
        let backend = audio_device_backend_text(&message).expect("mapped cpal message");
        assert!(cpal_backend_text_is_fatal(backend), "{backend}");
        assert!(backend.contains("device busy"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn mapped_stream_config_not_supported_is_retryable() {
        let err = map_cpal(
            "The requested stream configuration is not supported by the device.",
            "Could not open the speakers",
        );
        assert!(
            !open_failure_is_fatal(&err),
            "recovery hint must not make stream-config races fatal"
        );
    }

    #[test]
    #[cfg(not(coverage))]
    fn open_with_retries_on_stream_config_not_supported_race() {
        let mut attempts = 0;
        let opened = open_with_retries(
            "test speakers",
            Duration::from_millis(500),
            Duration::from_millis(1),
            || {
                attempts += 1;
                if attempts < 2 {
                    Err(map_cpal(
                        "The requested stream configuration is not supported by the device.",
                        "Could not open the speakers",
                    ))
                } else {
                    Ok(attempts)
                }
            },
        )
        .expect("second attempt succeeds after transient stream-config error");
        assert_eq!(opened, 2);
    }

    #[test]
    #[cfg(not(coverage))]
    fn fatal_preferred_open_never_runs_fallback() {
        let mut fallback_calls = 0;
        let error = open_preferred_then_fallback(
            "test microphone",
            Duration::from_millis(50),
            Duration::from_millis(1),
            true,
            || Err(map_cpal("device busy", "Could not open the microphone")),
            || {
                fallback_calls += 1;
                Ok(())
            },
        )
        .expect_err("fatal preferred error");
        assert_eq!(fallback_calls, 0);
        let Error::AudioDevice { message } = error else {
            panic!("expected audio device error");
        };
        let backend = audio_device_backend_text(&message).expect("mapped cpal message");
        assert!(backend.contains("device busy"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn transient_preferred_timeout_returns_fallback_error() {
        let error = open_preferred_then_fallback(
            "test speakers",
            Duration::from_millis(25),
            Duration::from_millis(5),
            true,
            || -> Result<()> {
                Err(map_cpal(
                    "The requested stream configuration is not supported by the device.",
                    "Could not open the speakers",
                ))
            },
            || {
                Err(Error::AudioDevice {
                    message: "fallback format failed".into(),
                })
            },
        )
        .expect_err("fallback error wins after transient retry window");
        assert_eq!(error.to_string(), "fallback format failed");
        assert!(
            !error.to_string().contains("not supported"),
            "must not return the retry error"
        );
    }

    #[test]
    #[cfg(not(coverage))]
    fn discard_buffered_audio_clears_pending_leftover_and_ring() {
        let mut capture = test_capture();
        capture
            .pending
            .push(AudioFrame::new(0, 16_000, 1, vec![1; 160]).expect("frame"));
        capture.capture_leftover.extend_from_slice(&[2, 3, 4]);
        capture.ring.try_push_slice(&[0.5; 64]);
        assert!(!capture.pending.is_empty());
        assert!(!capture.capture_leftover.is_empty());
        assert!(capture.ring.occupancy() > 0);

        capture.discard_buffered_audio();

        assert!(capture.pending.is_empty());
        assert!(capture.capture_leftover.is_empty());
        assert_eq!(capture.ring.occupancy(), 0);
    }

    #[test]
    #[cfg(not(coverage))]
    fn open_retries_transient_failures() {
        let mut attempts = 0;
        let opened = open_with_retries(
            "test speakers",
            Duration::from_millis(500),
            Duration::from_millis(1),
            || {
                attempts += 1;
                if attempts < 3 {
                    Err(Error::AudioDevice {
                        message: "transient".into(),
                    })
                } else {
                    Ok(attempts)
                }
            },
        )
        .expect("third attempt succeeds");
        assert_eq!(opened, 3);
    }

    #[test]
    #[cfg(not(coverage))]
    fn open_stops_retrying_once_the_window_closes() {
        let mut attempts = 0;
        let error = open_with_retries(
            "test speakers",
            Duration::from_millis(25),
            Duration::from_millis(5),
            || -> Result<()> {
                attempts += 1;
                Err(Error::AudioDevice {
                    message: "still busy".into(),
                })
            },
        )
        .expect_err("a permanently busy device fails");
        assert!(attempts > 1, "retried {attempts} times");
        assert!(error.to_string().contains("still busy"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn default_config_comes_first_and_repeats_are_dropped() {
        let default = cpal::SupportedStreamConfig::new(
            2,
            cpal::SampleRate(44_100),
            cpal::SupportedBufferSize::Unknown,
            SampleFormat::F32,
        );
        let ranges = [
            cpal::SupportedStreamConfigRange::new(
                2,
                cpal::SampleRate(44_100),
                cpal::SampleRate(44_100),
                cpal::SupportedBufferSize::Unknown,
                SampleFormat::F32,
            ),
            cpal::SupportedStreamConfigRange::new(
                1,
                cpal::SampleRate(44_100),
                cpal::SampleRate(96_000),
                cpal::SupportedBufferSize::Unknown,
                SampleFormat::F32,
            ),
        ];
        let configs = config_candidates(Some(default.clone()), ranges);

        assert_eq!(configs.first(), Some(&default));
        assert_eq!(configs.len(), 2, "{configs:?}");
        assert_eq!(
            configs[1],
            cpal::SupportedStreamConfig::new(
                1,
                cpal::SampleRate(48_000),
                cpal::SupportedBufferSize::Unknown,
                SampleFormat::F32,
            )
        );
    }

    #[test]
    #[cfg(not(coverage))]
    fn float_default_stays_ahead_of_integer_fallbacks() {
        let default = cpal::SupportedStreamConfig::new(
            2,
            cpal::SampleRate(48_000),
            cpal::SupportedBufferSize::Unknown,
            SampleFormat::F32,
        );
        let ranges = [cpal::SupportedStreamConfigRange::new(
            2,
            cpal::SampleRate(48_000),
            cpal::SampleRate(48_000),
            cpal::SupportedBufferSize::Unknown,
            SampleFormat::I16,
        )];
        let configs = config_candidates(Some(default.clone()), ranges);

        assert_eq!(configs.first(), Some(&default));
        assert_eq!(configs.len(), 2);
        assert_eq!(configs[1].sample_format(), SampleFormat::I16);
    }

    #[test]
    #[cfg(not(coverage))]
    fn supported_rate_prefers_the_standard_rates() {
        let range = |min: u32, max: u32| {
            cpal::SupportedStreamConfigRange::new(
                2,
                cpal::SampleRate(min),
                cpal::SampleRate(max),
                cpal::SupportedBufferSize::Unknown,
                SampleFormat::F32,
            )
        };
        assert_eq!(preferred_sample_rate(&range(44_100, 96_000)).0, 48_000);
        assert_eq!(preferred_sample_rate(&range(44_100, 44_100)).0, 44_100);
        assert_eq!(preferred_sample_rate(&range(96_000, 192_000)).0, 192_000);
    }

    #[test]
    fn stream_worker_drop_signals_stop_without_a_thread() {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let worker = StreamWorker {
            thread: None,
            stop: Arc::clone(&stop),
        };
        drop(worker);
        assert!(*stop.0.lock().expect("stop state"));
    }

    #[test]
    fn stream_worker_drop_joins_a_finished_thread() {
        let stop = Arc::new((Mutex::new(false), Condvar::new()));
        let worker = StreamWorker {
            thread: Some(thread::spawn(|| {})),
            stop: Arc::clone(&stop),
        };
        drop(worker);
        assert!(*stop.0.lock().expect("stop state"));
    }

    #[cfg(coverage)]
    #[test]
    fn coverage_native_openers_report_unavailable() {
        let error_message = |result: Result<()>| {
            let Err(Error::AudioDevice { message }) = result else {
                panic!("coverage native opener should be unavailable");
            };
            assert!(message.contains("coverage tests"));
        };

        error_message(NativeCapture::open().map(|_| ()));
        error_message(
            NativeCapture::open_with_echo(EchoReference::new(
                Arc::new(SampleRing::new(8)),
                PcmFormat::v0(),
            ))
            .map(|_| ()),
        );
        error_message(
            NativeCapture::open_with_echo_and_events(
                EchoReference::new(Arc::new(SampleRing::new(8)), PcmFormat::v0()),
                None,
            )
            .map(|_| ()),
        );
        error_message(NativePlayback::open().map(|_| ()));
        error_message(NativePlayback::open_with_echo().map(|_| ()));
        error_message(NativePlayback::open_with_echo_and_watch(None).map(|_| ()));
        error_message(probe_device_names().map(|_| ()));

        let drain = PlaybackDrain::new();
        let cancel = Cancel::new();
        drain.begin_final(crate::types::TurnId(0), cancel.generation());
        drain.final_enqueued(crate::types::TurnId(0));
        drain
            .wait_for(crate::types::TurnId(0), &cancel)
            .expect("coverage drain stub");
        drain.clear();
    }

    #[test]
    #[cfg(not(coverage))]
    fn closed_stream_open_channel_is_actionable() {
        let (tx, rx) = mpsc::sync_channel(1);
        drop(tx);
        let err = match recv_open(rx) {
            Err(err) => err,
            Ok(_) => panic!("closed channel must fail"),
        };
        assert!(err
            .to_string()
            .contains("Audio thread exited before the device finished opening"));
    }

    #[test]
    #[cfg(not(coverage))]
    fn cpal_inventory_queries_are_safe_without_devices() {
        let inventory = CpalInventory::new();
        let _ = inventory.default_input();
        let _ = inventory.default_output();
        let _ = inventory.inputs();
        let _ = inventory.outputs();
    }

    #[test]
    fn capture_converts_ring_samples_and_respects_shutdown() {
        let mut capture = test_capture();
        let cancel = Cancel::new();
        assert_eq!(capture.name(), "cpal");

        let samples = vec![0.25; 1_024];
        assert_eq!(capture.ring.try_push_slice(&samples), samples.len());
        let first = capture
            .next_frame(&cancel)
            .expect("read test microphone")
            .expect("frame");
        first.validate().expect("v0 frame");
        assert!(first.has_energy());
        assert!(first.capture_pcm.is_none());

        capture.ring.close();
        let pending = capture
            .next_frame(&cancel)
            .expect("read pending test microphone")
            .expect("pending frame");
        pending.validate().expect("v0 pending frame");
        assert_eq!(capture.next_frame(&cancel).expect("closed capture"), None);

        cancel.shutdown();
        assert!(matches!(capture.next_frame(&cancel), Err(Error::Cancelled)));
    }

    #[test]
    fn pcm_tap_duplicates_clean_as_capture_without_aec() {
        let mut capture = test_capture();
        capture.enable_pcm_tap();
        let cancel = Cancel::new();
        let samples = vec![0.5; 1_024];
        assert_eq!(capture.ring.try_push_slice(&samples), samples.len());
        let first = capture.next_frame(&cancel).expect("read").expect("frame");
        assert_eq!(first.capture_pcm.as_deref(), Some(first.samples.as_slice()));
    }

    #[test]
    fn playback_drops_stale_audio_and_writes_live_audio() {
        let mut playback = test_playback();
        let cancel = Cancel::new();
        let audio = SynthesizedAudio {
            turn: crate::types::TurnId(0),
            generation: cancel.generation(),
            index: 0,
            samples: vec![1_024, -1_024],
            is_last: true,
        };
        playback.play(audio.clone(), &cancel).expect("play audio");
        let mut rendered = [0.0; 2];
        assert_eq!(playback.ring.try_pop_slice(&mut rendered), 2);
        assert!(rendered[0] > 0.0);
        assert!(rendered[1] < 0.0);
        let mut flush_tail = [0.0; 8];
        assert!(playback.ring.try_pop_slice(&mut flush_tail) > 0);

        cancel.cancel_generation();
        playback.play(audio, &cancel).expect("drop stale audio");
        assert_eq!(playback.ring.occupancy(), 0);

        cancel.shutdown();
        let stopped = SynthesizedAudio {
            turn: crate::types::TurnId(0),
            generation: cancel.generation(),
            index: 1,
            samples: vec![0],
            is_last: true,
        };
        assert!(matches!(
            playback.play(stopped, &cancel),
            Err(Error::Cancelled)
        ));
    }

    #[cfg(not(coverage))]
    #[test]
    fn final_turn_waits_for_the_silent_callback_after_its_pcm() {
        let drain = PlaybackDrain::new();
        let cancel = Cancel::new();
        let turn = crate::types::TurnId(7);
        let generation = cancel.generation();
        drain.begin_final(turn, generation);
        drain.final_enqueued(turn);

        let (done_tx, done_rx) = mpsc::channel();
        let wait_drain = drain.clone();
        let wait_cancel = cancel.clone();
        let waiter = std::thread::spawn(move || {
            wait_drain.wait_for(turn, &wait_cancel).expect("drain wait");
            done_tx.send(()).expect("notify drain");
        });

        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(20))
                .is_err(),
            "a queued final chunk must not release VAD before device consumption"
        );
        drain.on_callback(128);
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(20))
                .is_err(),
            "the callback carrying final PCM is not yet a drain"
        );
        drain.on_callback(0);
        done_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("silent callback releases waiter");
        waiter.join().expect("drain waiter");
    }

    #[cfg(not(coverage))]
    #[test]
    fn final_turn_wait_exits_when_barge_in_cancels_generation() {
        let drain = PlaybackDrain::new();
        let cancel = Cancel::new();
        let turn = crate::types::TurnId(7);
        let generation = cancel.generation();
        drain.begin_final(turn, generation);
        drain.final_enqueued(turn);
        let wait_drain = drain.clone();
        let wait_cancel = cancel.clone();
        let waiter = std::thread::spawn(move || wait_drain.wait_for(turn, &wait_cancel));

        std::thread::sleep(std::time::Duration::from_millis(10));
        cancel.cancel_generation();
        assert!(matches!(
            waiter.join().expect("drain waiter"),
            Err(Error::Cancelled)
        ));
    }

    #[test]
    fn interrupt_clears_queued_playback() {
        let mut playback = test_playback();
        let cancel = Cancel::new();
        playback
            .play(
                SynthesizedAudio {
                    turn: crate::types::TurnId(0),
                    generation: cancel.generation(),
                    index: 0,
                    samples: vec![1_024; 16],
                    is_last: false,
                },
                &cancel,
            )
            .expect("queue audio");
        assert!(playback.ring.occupancy() > 0);
        playback.interrupt();
        assert_eq!(playback.ring.occupancy(), 0);
    }

    #[test]
    fn play_attributes_callback_consumption_to_the_active_turn() {
        let dir = std::env::temp_dir().join(format!(
            "syllabix-native-watch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let recorder = crate::turn_debug::TurnDebug::open(&dir).expect("recorder");
        let watch = crate::turn_debug::PlaybackWatch::new(recorder.clone());
        // Seed the VAD epoch the pipeline always records for a real turn.
        recorder.note_anchor(
            crate::types::TurnId(3),
            crate::turn_debug::TimelineAnchor::SpeechStart,
            std::time::Instant::now(),
        );
        let device_format = PcmFormat::v0();
        let mut playback = NativePlayback {
            _worker: idle_worker(),
            ring: Arc::new(SampleRing::new(4_096)),
            echo_ring: None,
            conv: PcmConverter::new(PcmFormat::v0(), device_format).expect("v0 converter"),
            watch: Some(watch.clone()),
            drain: PlaybackDrain::new(),
            device_name: "test speakers".into(),
            device_format,
        };
        let cancel = Cancel::new();
        let live_generation = cancel.generation();
        playback
            .play(
                SynthesizedAudio {
                    turn: crate::types::TurnId(3),
                    generation: live_generation,
                    index: 0,
                    samples: vec![256; 64],
                    is_last: false,
                },
                &cancel,
            )
            .expect("queue chunk");

        // Simulate the device callback consuming what play() queued.
        let mut out = [0.0f32; 64];
        let rendered = playback.ring.try_pop_slice(&mut out);
        assert!(rendered > 0);
        watch.on_callback(rendered);

        // Stale generations never touch the watch.
        cancel.cancel_generation();
        playback
            .play(
                SynthesizedAudio {
                    turn: crate::types::TurnId(4),
                    generation: live_generation, // pre-bump ⇒ now stale
                    index: 1,
                    samples: vec![256; 64],
                    is_last: false,
                },
                &cancel,
            )
            .expect("stale play returns Ok");
        watch.on_callback(0); // drain attributes to the still-active turn 3

        let anchored = recorder.anchored(crate::types::TurnId(3));
        assert_eq!(
            anchored.iter().map(|(a, _)| *a).collect::<Vec<_>>(),
            vec![
                crate::turn_debug::TimelineAnchor::SpeechStart,
                crate::turn_debug::TimelineAnchor::PlaybackFirst,
                crate::turn_debug::TimelineAnchor::PlaybackDone
            ]
        );
        assert!(
            recorder.anchored(crate::types::TurnId(4)).is_empty(),
            "a stale chunk must not retarget the watch"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn play_stops_feeding_the_ring_when_generation_goes_stale() {
        let mut playback = NativePlayback {
            _worker: idle_worker(),
            ring: Arc::new(SampleRing::new(32)),
            echo_ring: None,
            conv: PcmConverter::new(PcmFormat::v0(), PcmFormat::v0()).expect("v0 converter"),
            watch: None,
            drain: PlaybackDrain::new(),
            device_name: "test speakers".into(),
            device_format: PcmFormat::v0(),
        };
        let cancel = Cancel::new();
        let generation = cancel.generation();
        let watcher = cancel.clone();
        thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(30));
            watcher.cancel_generation();
        });
        let started = std::time::Instant::now();
        playback
            .play(
                SynthesizedAudio {
                    turn: crate::types::TurnId(0),
                    generation,
                    index: 0,
                    samples: vec![1_024; 16_000],
                    is_last: true,
                },
                &cancel,
            )
            .expect("stale play returns");
        assert!(
            started.elapsed() < std::time::Duration::from_millis(500),
            "barge-in must abort a blocked speaker write, took {:?}",
            started.elapsed()
        );
        assert_eq!(playback.ring.occupancy(), 0);
    }

    // Windows hosted runners can access-violate inside the audio driver while
    // opening a device. Keep this real-device smoke test on Unix; Windows
    // still runs inventory and all deterministic CPAL tests above.
    #[cfg(not(target_os = "windows"))]
    #[test]
    fn missing_hardware_error_is_actionable() {
        // CI has no guaranteed mic. Opening may fail; the message must tell a human what to do.
        match NativeCapture::open() {
            Ok(_) => {}
            Err(Error::AudioDevice { message }) => {
                assert!(
                    message.contains("microphone")
                        || message.contains("Microphone")
                        || message.contains("permission")
                        || message.contains("device"),
                    "{message}"
                );
            }
            Err(other) => panic!("unexpected error: {other}"),
        }
    }
}
