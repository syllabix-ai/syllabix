//! Moshi websocket binary framing, and 24 kHz Ogg/Opus pages with `s2s-opus`.
//!
//! Protocol: <https://github.com/kyutai-labs/moshi/blob/main/rust/protocol.md>
//! Each websocket binary message starts with a 1-byte type. Unknown types
//! are discarded. Audio payloads are Ogg pages of Opus at 24 kHz mono.

use crate::error::{Error, Result};

#[cfg(feature = "s2s-opus")]
mod opus;
#[cfg(feature = "s2s-opus")]
pub use opus::{OggOpusDecoder, OggOpusEncoder, OGG_SERIAL, OPUS_FRAME_SAMPLES};

/// Moshi in/out PCM rate (Hz).
pub const MOSHI_SAMPLE_RATE_HZ: u32 = 24_000;
/// Moshi channel layout.
pub const MOSHI_CHANNELS: u16 = 1;
/// Protocol version in handshake payloads (always 0 today).
pub const PROTOCOL_VERSION: u32 = 0;

/// First byte of a Moshi websocket binary message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Handshake: protocol version + model version (`u32` LE each).
    Handshake = 0,
    /// Audio: Ogg/Opus pages (24 kHz mono).
    Audio = 1,
    /// UTF-8 text.
    Text = 2,
    /// One control byte (not used in full streaming mode).
    Control = 3,
    /// UTF-8 JSON metadata.
    Metadata = 4,
    /// UTF-8 error description.
    Error = 5,
    /// Ping; no payload, currently unused.
    Ping = 6,
}

impl MessageType {
    /// Parse a known type byte. Unknown values are `None` (discard).
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Handshake),
            1 => Some(Self::Audio),
            2 => Some(Self::Text),
            3 => Some(Self::Control),
            4 => Some(Self::Metadata),
            5 => Some(Self::Error),
            6 => Some(Self::Ping),
            _ => None,
        }
    }
}

/// Control payload byte (`MT=3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ControlKind {
    /// Start.
    Start = 0,
    /// End of turn.
    EndTurn = 1,
    /// Pause.
    Pause = 2,
    /// Restart.
    Restart = 3,
}

impl ControlKind {
    /// Known control bytes. Other values stay as the raw `u8` on [`MoshiMessage::Control`].
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Start),
            1 => Some(Self::EndTurn),
            2 => Some(Self::Pause),
            3 => Some(Self::Restart),
            _ => None,
        }
    }
}

/// One decoded Moshi websocket message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoshiMessage {
    /// `MT=0`.
    Handshake {
        /// Protocol version (`u32` LE).
        protocol_version: u32,
        /// Model version (`u32` LE).
        model_version: u32,
    },
    /// `MT=1` Ogg/Opus bytes.
    Audio(Vec<u8>),
    /// `MT=2`.
    Text(String),
    /// `MT=3` control byte (known or raw).
    Control(u8),
    /// `MT=4`.
    Metadata(String),
    /// `MT=5`.
    Error(String),
    /// `MT=6`.
    Ping,
}

impl MoshiMessage {
    /// Encode as a websocket binary payload (type byte + body).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::Handshake {
                protocol_version,
                model_version,
            } => {
                let mut out = Vec::with_capacity(9);
                out.push(MessageType::Handshake as u8);
                out.extend_from_slice(&protocol_version.to_le_bytes());
                out.extend_from_slice(&model_version.to_le_bytes());
                out
            }
            Self::Audio(pages) => {
                let mut out = Vec::with_capacity(1 + pages.len());
                out.push(MessageType::Audio as u8);
                out.extend_from_slice(pages);
                out
            }
            Self::Text(text) => encode_utf8(MessageType::Text, text),
            Self::Control(kind) => vec![MessageType::Control as u8, *kind],
            Self::Metadata(json) => encode_utf8(MessageType::Metadata, json),
            Self::Error(message) => encode_utf8(MessageType::Error, message),
            Self::Ping => vec![MessageType::Ping as u8],
        }
    }

    /// Decode a websocket binary payload.
    ///
    /// Empty input is an error. Unknown type bytes return `Ok(None)` so the
    /// caller can discard them. Known types with a truncated or invalid
    /// payload return `Err`.
    pub fn decode(bytes: &[u8]) -> Result<Option<Self>> {
        let Some((&ty, payload)) = bytes.split_first() else {
            return Err(framing_err("empty websocket message"));
        };
        let Some(kind) = MessageType::from_u8(ty) else {
            return Ok(None);
        };
        match kind {
            MessageType::Handshake => {
                if payload.len() != 8 {
                    return Err(framing_err(format!(
                        "handshake payload is {} bytes, expected 8",
                        payload.len()
                    )));
                }
                let protocol_version = u32::from_le_bytes(payload[0..4].try_into().expect("4"));
                let model_version = u32::from_le_bytes(payload[4..8].try_into().expect("4"));
                Ok(Some(Self::Handshake {
                    protocol_version,
                    model_version,
                }))
            }
            MessageType::Audio => Ok(Some(Self::Audio(payload.to_vec()))),
            MessageType::Text => Ok(Some(Self::Text(utf8("text", payload)?))),
            MessageType::Control => {
                if payload.len() != 1 {
                    return Err(framing_err(format!(
                        "control payload is {} bytes, expected 1",
                        payload.len()
                    )));
                }
                Ok(Some(Self::Control(payload[0])))
            }
            MessageType::Metadata => Ok(Some(Self::Metadata(utf8("metadata", payload)?))),
            MessageType::Error => Ok(Some(Self::Error(utf8("error", payload)?))),
            MessageType::Ping => Ok(Some(Self::Ping)),
        }
    }
}

fn encode_utf8(ty: MessageType, text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + text.len());
    out.push(ty as u8);
    out.extend_from_slice(text.as_bytes());
    out
}

fn utf8(field: &str, payload: &[u8]) -> Result<String> {
    String::from_utf8(payload.to_vec())
        .map_err(|_| framing_err(format!("{field} payload is not valid UTF-8")))
}

fn framing_err(message: impl Into<String>) -> Error {
    Error::Provider {
        provider: "moshi",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(msg: MoshiMessage) -> MoshiMessage {
        MoshiMessage::decode(&msg.encode())
            .expect("decode")
            .expect("known type")
    }

    #[test]
    fn message_type_bytes_match_protocol() {
        assert_eq!(MessageType::Handshake as u8, 0);
        assert_eq!(MessageType::Audio as u8, 1);
        assert_eq!(MessageType::Text as u8, 2);
        assert_eq!(MessageType::Control as u8, 3);
        assert_eq!(MessageType::Metadata as u8, 4);
        assert_eq!(MessageType::Error as u8, 5);
        assert_eq!(MessageType::Ping as u8, 6);
        assert_eq!(MessageType::from_u8(0), Some(MessageType::Handshake));
        assert_eq!(MessageType::from_u8(7), None);
        assert_eq!(MessageType::from_u8(255), None);
        assert_eq!(ControlKind::from_u8(0), Some(ControlKind::Start));
        assert_eq!(ControlKind::from_u8(1), Some(ControlKind::EndTurn));
        assert_eq!(ControlKind::from_u8(2), Some(ControlKind::Pause));
        assert_eq!(ControlKind::from_u8(3), Some(ControlKind::Restart));
        assert_eq!(ControlKind::from_u8(4), None);
    }

    #[test]
    fn handshake_roundtrip_is_little_endian() {
        let msg = MoshiMessage::Handshake {
            protocol_version: PROTOCOL_VERSION,
            model_version: 0x0102_0304,
        };
        let bytes = msg.encode();
        assert_eq!(bytes[0], 0);
        assert_eq!(&bytes[1..5], &0u32.to_le_bytes());
        assert_eq!(&bytes[5..9], &0x0102_0304u32.to_le_bytes());
        assert_eq!(roundtrip(msg.clone()), msg);
    }

    #[test]
    fn audio_text_metadata_error_ping_roundtrip() {
        let audio = MoshiMessage::Audio(vec![1, 2, 3]);
        assert_eq!(roundtrip(audio.clone()), audio);
        assert_eq!(audio.encode()[0], 1);

        let text = MoshiMessage::Text(String::from("hello"));
        assert_eq!(roundtrip(text.clone()), text);
        assert_eq!(text.encode(), b"\x02hello");

        let empty_text = MoshiMessage::Text(String::new());
        assert_eq!(roundtrip(empty_text.clone()), empty_text);

        let meta = MoshiMessage::Metadata(String::from("{\"ok\":true}"));
        assert_eq!(roundtrip(meta.clone()), meta);
        assert_eq!(meta.encode()[0], 4);

        let err = MoshiMessage::Error(String::from("boom"));
        assert_eq!(roundtrip(err.clone()), err);
        assert_eq!(err.encode()[0], 5);

        assert_eq!(roundtrip(MoshiMessage::Ping), MoshiMessage::Ping);
        assert_eq!(MoshiMessage::Ping.encode(), vec![6]);
    }

    #[test]
    fn control_roundtrip_keeps_unknown_bytes() {
        for kind in [
            ControlKind::Start,
            ControlKind::EndTurn,
            ControlKind::Pause,
            ControlKind::Restart,
        ] {
            let msg = MoshiMessage::Control(kind as u8);
            assert_eq!(roundtrip(msg.clone()), msg);
            assert_eq!(msg.encode(), vec![3, kind as u8]);
        }
        let raw = MoshiMessage::Control(99);
        assert_eq!(roundtrip(raw.clone()), raw);
    }

    #[test]
    fn unknown_type_is_discarded() {
        assert_eq!(MoshiMessage::decode(&[7, 1, 2]).expect("ok"), None);
        assert_eq!(MoshiMessage::decode(&[99]).expect("ok"), None);
    }

    #[test]
    fn empty_message_is_an_error() {
        let err = MoshiMessage::decode(&[]).expect_err("empty");
        assert!(matches!(
            err,
            Error::Provider {
                provider: "moshi",
                ..
            }
        ));
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn truncated_handshake_and_control_are_errors() {
        let short = MoshiMessage::decode(&[0, 1, 2, 3]).expect_err("short handshake");
        assert!(short.to_string().contains("handshake"));
        let long = MoshiMessage::decode(&[0; 10]).expect_err("long handshake");
        assert!(long.to_string().contains("handshake"));
        let empty_ctrl = MoshiMessage::decode(&[3]).expect_err("empty control");
        assert!(empty_ctrl.to_string().contains("control"));
        let long_ctrl = MoshiMessage::decode(&[3, 0, 1]).expect_err("long control");
        assert!(long_ctrl.to_string().contains("control"));
    }

    #[test]
    fn invalid_utf8_text_and_error_are_rejected() {
        let bad = [2, 0xff, 0xfe];
        let err = MoshiMessage::decode(&bad).expect_err("text");
        assert!(err.to_string().contains("UTF-8"));
        let err = MoshiMessage::decode(&[5, 0x80]).expect_err("error");
        assert!(err.to_string().contains("UTF-8"));
        let err = MoshiMessage::decode(&[4, 0xff]).expect_err("metadata");
        assert!(err.to_string().contains("UTF-8"));
    }

    #[test]
    fn ping_ignores_trailing_payload() {
        assert_eq!(
            MoshiMessage::decode(&[6, 1, 2]).expect("ok"),
            Some(MoshiMessage::Ping)
        );
    }

    #[test]
    fn debug_and_eq_cover_variants() {
        let a = MoshiMessage::Audio(vec![9]);
        let t = MoshiMessage::Text(String::from("x"));
        assert_ne!(a, t);
        assert!(format!("{a:?}").contains("Audio"));
        assert!(format!("{t:?}").contains('x'));
        assert_eq!(MOSHI_SAMPLE_RATE_HZ, 24_000);
        assert_eq!(MOSHI_CHANNELS, 1);
    }
}
