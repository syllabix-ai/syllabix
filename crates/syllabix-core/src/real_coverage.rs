use crate::cancel::Cancel;
use crate::config::AgentConfig;
use crate::error::{Error, Result};
use crate::models::{Fetcher, ModelCache, Progress};
use crate::providers::{Llm, Stt, Tts};
use crate::types::{HistoryTurn, LlmDebugMeta, TokenChunk, Transcript, Utterance};
use crate::vad::SileroVad;

pub struct LiveLlm;
pub struct LiveTts;
pub struct LiveStt;

fn unavailable(provider: &'static str) -> Error { Error::Provider { provider, message: "inference is not loaded in coverage tests".into() } }

impl Llm for LiveLlm {
    fn name(&self) -> &'static str { "local" }
    fn debug_meta(&self) -> Option<LlmDebugMeta> { None }
    fn take_tool_events(&mut self) -> Vec<crate::types::ToolTurnEvent> { Vec::new() }
    fn generate(&mut self, _: &[HistoryTurn], _: &Transcript, _: &Cancel, _: &mut dyn FnMut(TokenChunk) -> Result<()>) -> Result<()> { Err(unavailable("llm")) }
}

impl Tts for LiveTts {
    fn name(&self) -> &'static str { "local" }
    fn model_id(&self) -> Option<&str> { None }
    fn synthesize_chunk(&mut self, _: &TokenChunk, _: &Cancel) -> Result<Vec<crate::types::SynthesizedAudio>> { Err(unavailable("tts")) }
    fn synthesize_chunk_into(&mut self, _: &TokenChunk, _: &Cancel, _: &mut dyn FnMut(crate::types::SynthesizedAudio) -> Result<()>) -> Result<()> { Err(unavailable("tts")) }
}

impl Stt for LiveStt {
    fn name(&self) -> &'static str { "whisper" }
    fn transcribe(&mut self, _: &Utterance, _: &Cancel) -> Result<Transcript> { Err(unavailable("stt")) }
    fn supports_partials(&self) -> bool { false }
    fn start_turn(&mut self, _: crate::TurnId, _: &Cancel) -> Result<()> { Err(unavailable("stt")) }
    fn push_frame(&mut self, _: &crate::AudioFrame, _: &Cancel) -> Result<Option<String>> { Err(unavailable("stt")) }
    fn cancel_turn(&mut self, _: crate::TurnId) {}
}

impl LiveStt { pub fn with_language(self, _: &str) -> Result<Self> { Ok(self) } }
pub fn build_stt(_: &ModelCache, _: &dyn Fetcher, _: &mut dyn Progress, _: &Cancel, _: &AgentConfig) -> Result<LiveStt> { Err(unavailable("stt")) }
pub fn build_tts(_: &ModelCache, _: &dyn Fetcher, _: &mut dyn Progress, _: &Cancel, _: &AgentConfig) -> Result<LiveTts> { Err(unavailable("tts")) }
pub fn build_llm(_: &ModelCache, _: &dyn Fetcher, _: &mut dyn Progress, _: &Cancel, _: &AgentConfig, _: Option<&zeroize::Zeroizing<String>>) -> Result<LiveLlm> { Err(unavailable("llm")) }
pub fn load_real_providers(_: &ModelCache, _: &dyn Fetcher, _: &mut dyn Progress, _: &Cancel, _: &AgentConfig, _: Option<&zeroize::Zeroizing<String>>) -> Result<(SileroVad, LiveStt, LiveLlm, LiveTts)> { Err(unavailable("providers")) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{BlockedFetcher, Manifest, NoProgress};

    #[test]
    fn coverage_provider_seams_are_callable() {
        let cancel = Cancel::new();
        let mut llm = LiveLlm;
        assert_eq!(llm.name(), "local");
        assert!(llm.debug_meta().is_none());
        assert!(llm.take_tool_events().is_empty());
        let transcript = Transcript { turn: crate::TurnId(0), text: String::new(), language: String::new() };
        assert!(llm.generate(&[], &transcript, &cancel, &mut |_| Ok(())).is_err());
        let mut tts = LiveTts;
        assert_eq!(tts.name(), "local");
        assert!(tts.model_id().is_none());
        let token = TokenChunk { turn: crate::TurnId(0), generation: crate::GenerationId(0), index: 0, text: String::new(), is_last: true };
        assert!(tts.synthesize_chunk(&token, &cancel).is_err());
        assert!(tts.synthesize_chunk_into(&token, &cancel, &mut |_| Ok(())).is_err());
        let mut stt = LiveStt;
        assert_eq!(stt.name(), "whisper");
        assert!(!stt.supports_partials());
        let utterance = Utterance { turn: crate::TurnId(0), frames: Vec::new() };
        assert!(stt.transcribe(&utterance, &cancel).is_err());
        assert!(stt.start_turn(crate::TurnId(0), &cancel).is_err());
        let frame = crate::AudioFrame { seq: 0, sample_rate_hz: 16_000, channels: 1, samples: vec![0], capture_pcm: None };
        assert!(stt.push_frame(&frame, &cancel).is_err());
        stt.cancel_turn(crate::TurnId(0));
        assert!(stt.with_language("en").is_ok());
        let cache = ModelCache::new(
            std::env::temp_dir().join(format!("syllabix-real-coverage-{}", std::process::id())),
            Manifest { version: 1, assets: vec![] },
        );
        let fetcher = BlockedFetcher::default();
        let mut progress = NoProgress;
        assert!(build_stt(&cache, &fetcher, &mut progress, &cancel, &AgentConfig::v0()).is_err());
        assert!(build_tts(&cache, &fetcher, &mut progress, &cancel, &AgentConfig::v0()).is_err());
        assert!(build_llm(&cache, &fetcher, &mut progress, &cancel, &AgentConfig::v0(), None).is_err());
        assert!(load_real_providers(&cache, &fetcher, &mut progress, &cancel, &AgentConfig::v0(), None).is_err());
    }
}
