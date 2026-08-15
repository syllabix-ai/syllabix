//! Device selection rules. No CLI flags — OS default, then first usable device.

use crate::error::{Error, Result};

/// One capture or playback device the host advertised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// OS name (`MacBook Pro Microphone`, `hw:0,0`, …).
    pub name: String,
    /// Default stream sample rate.
    pub sample_rate_hz: u32,
    /// Default stream channel count.
    pub channels: u16,
}

impl DeviceInfo {
    /// True when the name looks like a speaker tap rather than a microphone.
    pub fn looks_like_monitor(&self) -> bool {
        name_looks_like_monitor(&self.name)
    }
}

/// Chosen device plus why it won.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceChoice {
    /// Selected device.
    pub device: DeviceInfo,
    /// Short reason for logs (`os-default`, `first-non-monitor`, `only-available`).
    pub reason: &'static str,
}

/// Inventory the native host (or a test fake) can answer.
pub trait DeviceInventory {
    /// OS default input, if any.
    fn default_input(&self) -> Option<DeviceInfo>;
    /// OS default output, if any.
    fn default_output(&self) -> Option<DeviceInfo>;
    /// All input devices, default first when the host lists it that way (order is not required).
    fn inputs(&self) -> Vec<DeviceInfo>;
    /// All output devices.
    fn outputs(&self) -> Vec<DeviceInfo>;
}

/// True when a device name looks like a speaker tap rather than a microphone.
pub fn name_looks_like_monitor(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("monitor")
        || n.contains("loopback")
        || n.contains("what u hear")
        || n.contains("stereo mix")
        || n.contains("wave out mix")
}

/// OS-specific sentence appended to missing-mic errors.
pub fn input_privacy_hint() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "Allow Microphone access in System Settings → Privacy & Security, then retry."
    }
    #[cfg(target_os = "windows")]
    {
        "Allow microphone access in Settings → Privacy → Microphone, then retry."
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        "Check that a microphone is plugged in, unmuted, and not blocked by PipeWire/PulseAudio."
    }
}

fn output_hint() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "Check System Settings → Sound → Output and that the volume is not muted."
    }
    #[cfg(target_os = "windows")]
    {
        "Check Settings → System → Sound and that an output device is enabled."
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        "Check that speakers or headphones are connected and not muted."
    }
}

/// Pick a microphone. Prefer the OS default unless it is a monitor and a real mic exists.
pub fn select_input<I: DeviceInventory>(inv: &I) -> Result<DeviceChoice> {
    let inputs = inv.inputs();
    if inputs.is_empty() && inv.default_input().is_none() {
        return Err(Error::AudioDevice {
            message: format!("No microphone found. {}", input_privacy_hint()),
        });
    }
    if let Some(default) = inv.default_input() {
        if default.looks_like_monitor() {
            if let Some(mic) = inputs.iter().find(|d| !d.looks_like_monitor()) {
                return Ok(DeviceChoice {
                    device: mic.clone(),
                    reason: "skipped-monitor-default",
                });
            }
        }
        return Ok(DeviceChoice {
            device: default,
            reason: "os-default",
        });
    }
    if let Some(mic) = inputs.iter().find(|d| !d.looks_like_monitor()) {
        return Ok(DeviceChoice {
            device: mic.clone(),
            reason: "first-non-monitor",
        });
    }
    Ok(DeviceChoice {
        device: inputs.into_iter().next().expect("non-empty checked"),
        reason: "only-available",
    })
}

/// Pick speakers. Prefer the OS default output.
pub fn select_output<I: DeviceInventory>(inv: &I) -> Result<DeviceChoice> {
    if let Some(default) = inv.default_output() {
        return Ok(DeviceChoice {
            device: default,
            reason: "os-default",
        });
    }
    let outputs = inv.outputs();
    if let Some(dev) = outputs.into_iter().next() {
        return Ok(DeviceChoice {
            device: dev,
            reason: "first-available",
        });
    }
    Err(Error::AudioDevice {
        message: format!("No speakers found. {}", output_hint()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeInv {
        default_in: Option<DeviceInfo>,
        default_out: Option<DeviceInfo>,
        inputs: Vec<DeviceInfo>,
        outputs: Vec<DeviceInfo>,
    }

    impl DeviceInventory for FakeInv {
        fn default_input(&self) -> Option<DeviceInfo> {
            self.default_in.clone()
        }
        fn default_output(&self) -> Option<DeviceInfo> {
            self.default_out.clone()
        }
        fn inputs(&self) -> Vec<DeviceInfo> {
            self.inputs.clone()
        }
        fn outputs(&self) -> Vec<DeviceInfo> {
            self.outputs.clone()
        }
    }

    fn mic(name: &str) -> DeviceInfo {
        DeviceInfo {
            name: name.into(),
            sample_rate_hz: 48_000,
            channels: 1,
        }
    }

    #[test]
    fn prefers_os_default_input() {
        let inv = FakeInv {
            default_in: Some(mic("Built-in Mic")),
            default_out: None,
            inputs: vec![mic("USB"), mic("Built-in Mic")],
            outputs: vec![],
        };
        let choice = select_input(&inv).unwrap();
        assert_eq!(choice.device.name, "Built-in Mic");
        assert_eq!(choice.reason, "os-default");
    }

    #[test]
    fn skips_monitor_default_when_a_real_mic_exists() {
        let inv = FakeInv {
            default_in: Some(mic("Monitor of Built-in Audio")),
            default_out: None,
            inputs: vec![mic("Monitor of Built-in Audio"), mic("Webcam Mic")],
            outputs: vec![],
        };
        let choice = select_input(&inv).unwrap();
        assert_eq!(choice.device.name, "Webcam Mic");
        assert_eq!(choice.reason, "skipped-monitor-default");
    }

    #[test]
    fn empty_inputs_are_actionable() {
        let inv = FakeInv {
            default_in: None,
            default_out: None,
            inputs: vec![],
            outputs: vec![],
        };
        let err = select_input(&inv).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("No microphone found"), "{msg}");
        assert!(!msg.is_empty());
    }

    #[test]
    fn empty_outputs_are_actionable() {
        let inv = FakeInv {
            default_in: None,
            default_out: None,
            inputs: vec![],
            outputs: vec![],
        };
        let err = select_output(&inv).unwrap_err();
        assert!(err.to_string().contains("No speakers found"));
    }

    #[test]
    fn first_non_monitor_when_no_default() {
        let inv = FakeInv {
            default_in: None,
            default_out: None,
            inputs: vec![mic("Stereo Mix"), mic("Headset")],
            outputs: vec![],
        };
        let choice = select_input(&inv).unwrap();
        assert_eq!(choice.device.name, "Headset");
        assert_eq!(choice.reason, "first-non-monitor");
    }

    #[test]
    fn output_uses_os_default() {
        let inv = FakeInv {
            default_in: None,
            default_out: Some(mic("Speakers")),
            inputs: vec![],
            outputs: vec![mic("HDMI"), mic("Speakers")],
        };
        let choice = select_output(&inv).unwrap();
        assert_eq!(choice.device.name, "Speakers");
        assert_eq!(choice.reason, "os-default");
    }
}
