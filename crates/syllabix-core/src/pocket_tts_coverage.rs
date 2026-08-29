//! Coverage-only API surface. Native inference tests are excluded under
//! `cfg(coverage)` because they load large external model artifacts.

pub const POCKET_TTS_BUNDLE_ASSET: &str = "pocket-tts-bundle";
pub const POCKET_TTS_BOS_ASSET: &str = "pocket-tts-bos";
pub const POCKET_TTS_TOKENIZER_ASSET: &str = "pocket-tts-tokenizer";
pub const POCKET_TTS_TEXT_CONDITIONER_ASSET: &str = "pocket-tts-text-conditioner";
pub const POCKET_TTS_FLOW_MAIN_ASSET: &str = "pocket-tts-flow-main";
pub const POCKET_TTS_FLOW_ASSET: &str = "pocket-tts-flow";
pub const POCKET_TTS_MIMI_DECODER_ASSET: &str = "pocket-tts-mimi-decoder";
pub const POCKET_TTS_VOICE_ASSET: &str = "pocket-tts-voice-alba";

pub struct PocketTts;
