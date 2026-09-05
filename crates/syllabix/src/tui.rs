//! Inline transcript and status renderer for `syllabix run`.

use syllabix_core::{LoopEvent, ThinkFilter, TurnId};

/// Rolling transcript shown in the TUI.
#[derive(Debug, Default)]
#[cfg_attr(coverage, allow(dead_code))]
pub struct TranscriptUi {
    lines: Vec<String>,
    partial_user: Option<(TurnId, String)>,
    current_agent: Option<(TurnId, String)>,
    think: ThinkFilter,
    latency: String,
    spoken_reply_turn: Option<TurnId>,
    thinking: bool,
    using_tools: bool,
    display_queue: Vec<String>,
    aec_active: bool,
    aec_restart_required: bool,
}

#[cfg_attr(coverage, allow(dead_code))]
impl TranscriptUi {
    /// Apply one pipeline event.
    pub fn apply(&mut self, event: LoopEvent) {
        match event {
            LoopEvent::Ready | LoopEvent::Playback { .. } => {}
            LoopEvent::Thinking { .. } => {
                self.flush_agent();
                self.thinking = true;
                self.using_tools = false;
                self.spoken_reply_turn = None;
            }
            LoopEvent::Aec {
                active,
                restart_required,
            } => {
                self.aec_active = active;
                self.aec_restart_required = restart_required;
            }
            LoopEvent::Partial { turn, text } => {
                self.partial_user = Some((turn, text));
            }
            LoopEvent::User {
                turn,
                text,
                language,
            } => {
                self.flush_agent();
                if matches!(&self.partial_user, Some((id, _)) if *id == turn) {
                    self.partial_user = None;
                }
                self.thinking = true;
                self.using_tools = false;
                self.spoken_reply_turn = None;
                // Non-English turns carry a visible language tag (fixed or
                // auto-detected); plain English stays untagged.
                let line = if language.is_empty() || language == "en" {
                    format!("You: {text}")
                } else {
                    format!("You [{language}]: {text}")
                };
                self.lines.push(line.clone());
                self.display_queue.push(line);
            }
            LoopEvent::Assistant {
                turn,
                text,
                is_last,
            } => {
                if !matches!(&self.current_agent, Some((id, _)) if *id == turn) {
                    self.flush_agent();
                }
                let spoken = self.think.push(&text, is_last);
                if !spoken.is_empty() {
                    self.thinking = false;
                    self.using_tools = false;
                    self.spoken_reply_turn = Some(turn);
                }
                match &mut self.current_agent {
                    Some((id, buf)) if *id == turn => buf.push_str(&spoken),
                    _ => self.current_agent = Some((turn, spoken)),
                }
                if is_last {
                    self.flush_agent();
                    self.thinking = false;
                }
            }
            LoopEvent::Tool { turn, event } => {
                if self.spoken_reply_turn == Some(turn) {
                    return;
                }
                match event.kind.as_str() {
                    "call" => {
                        self.using_tools = true;
                        self.thinking = false;
                    }
                    "result" => {
                        self.using_tools = false;
                        self.thinking = true;
                    }
                    _ => {}
                }
            }
            LoopEvent::Timings { timings, .. } => {
                self.latency = timings.format_line();
                self.thinking = false;
                self.using_tools = false;
            }
        }
    }

    pub fn is_thinking(&self) -> bool {
        self.thinking
    }
    pub fn is_using_tools(&self) -> bool {
        self.using_tools
    }
    pub fn take_display(&mut self) -> Vec<String> {
        std::mem::take(&mut self.display_queue)
    }
    pub fn agent_stream(&self) -> Option<(TurnId, String)> {
        self.current_agent.clone()
    }

    fn aec_pill(&self) -> &'static str {
        if self.aec_active {
            "AEC ● on"
        } else if self.aec_restart_required {
            "AEC ○ off — restart to fix"
        } else {
            "AEC ○ off"
        }
    }

    fn flush_agent(&mut self) {
        if let Some((_, text)) = self.current_agent.take() {
            if !text.is_empty() {
                let line = format!("Agent: {text}");
                self.lines.push(line.clone());
                self.display_queue.push(line);
            }
        }
        self.think = ThinkFilter::default();
    }

    /// Visible transcript, including an in-flight assistant line.
    pub fn transcript_text(&self) -> String {
        let mut lines = self.lines.clone();
        if let Some((_, text)) = &self.partial_user {
            if !text.is_empty() {
                lines.push(format!("You: {text}…"));
            }
        }
        if let Some((_, text)) = &self.current_agent {
            if !text.is_empty() {
                lines.push(format!("Agent: {text}"));
            }
        }
        if lines.is_empty() {
            "Listening… speak to start. q or Ctrl+C to quit.".into()
        } else {
            lines.join("\n")
        }
    }

    /// Latency footer.
    pub fn latency_line(&self) -> &str {
        if self.latency.is_empty() {
            "STT —  TTFT —  TTFB —  total —"
        } else {
            &self.latency
        }
    }
}

#[cfg(not(coverage))]
mod live_terminal {
    use super::TranscriptUi;
    use std::io::{stdout, Write};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    use crossterm::{
        cursor, execute,
        terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType},
    };
    use syllabix_core::{
        run_live_with_controls, AgentConfig, Cancel, Error, LoopEvent, Result, RuntimeControls,
    };

    struct RawTerminalGuard;

    impl Drop for RawTerminalGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
        }
    }

    struct InlineRenderer {
        ui: TranscriptUi,
        ready: bool,
        playing: bool,
        footer_drawn: bool,
        partial_visible: bool,
        last_mic_muted: bool,
        footer_text: Option<(String, String)>,
        hint_shown: bool,
        seen_agent: String,
        pending_agent: String,
        agent_open: bool,
    }

    impl InlineRenderer {
        fn new() -> Self {
            Self {
                ui: TranscriptUi::default(),
                ready: false,
                playing: false,
                footer_drawn: false,
                partial_visible: false,
                last_mic_muted: false,
                footer_text: None,
                hint_shown: false,
                seen_agent: String::new(),
                pending_agent: String::new(),
                agent_open: false,
            }
        }

        /// Remove the one provisional user line so the next partial can take
        /// its place, or the final transcript can replace it exactly once.
        fn clear_partial(&mut self, out: &mut impl Write) -> std::io::Result<()> {
            if self.partial_visible {
                execute!(
                    out,
                    cursor::MoveUp(1),
                    cursor::MoveToColumn(0),
                    Clear(ClearType::CurrentLine)
                )?;
                self.partial_visible = false;
            }
            Ok(())
        }

        fn clear_footer(&mut self, out: &mut impl Write) -> std::io::Result<()> {
            if self.footer_drawn {
                execute!(
                    out,
                    cursor::MoveToColumn(0),
                    Clear(ClearType::CurrentLine),
                    cursor::MoveUp(1),
                    cursor::MoveToColumn(0),
                    Clear(ClearType::CurrentLine)
                )?;
                self.footer_drawn = false;
            }
            Ok(())
        }

        fn commit_live_reply(&mut self) {
            self.seen_agent.clear();
            self.pending_agent.clear();
            self.agent_open = false;
        }

        fn queue_agent_text(&mut self, text: &str) {
            if text.len() < self.seen_agent.len() || !text.starts_with(&self.seen_agent) {
                self.seen_agent.clear();
                self.pending_agent.clear();
                self.pending_agent.push_str(text);
                self.seen_agent.push_str(text);
                return;
            }
            let suffix = &text[self.seen_agent.len()..];
            self.pending_agent.push_str(suffix);
            self.seen_agent.push_str(suffix);
        }

        fn flush_paragraphs(
            &mut self,
            out: &mut impl Write,
            final_fragment: bool,
            controls: &RuntimeControls,
        ) -> std::io::Result<()> {
            let ready_at = if final_fragment {
                self.pending_agent.len()
            } else {
                self.pending_agent.rfind('\n').map(|i| i + 1).unwrap_or(0)
            };
            if ready_at == 0 {
                return Ok(());
            }
            let remainder = self.pending_agent.split_off(ready_at);
            let paragraph = std::mem::replace(&mut self.pending_agent, remainder);
            self.clear_footer(out)?;
            if !self.agent_open {
                write!(out, "Agent: ")?;
                self.agent_open = true;
            }
            write_terminal_text(out, &paragraph)?;
            if !paragraph.ends_with('\n') {
                write!(out, "\r\n")?;
            }
            self.draw_footer_at_cursor(out, controls)
        }

        fn footer(&self, controls: &RuntimeControls) -> (String, String) {
            let (mic, agent_pill) = if controls.mic_muted() {
                ("mic ◌ muted", "agent ○ idle (press u to listen)")
            } else if controls.agent_muted() {
                ("mic ● live", "agent ○ idle (agent muted)")
            } else if controls.speaker_muted() {
                ("mic ● live", "agent ○ idle")
            } else if self.ui.is_using_tools() {
                ("mic ◌ muted", "agent ⚙ using tools")
            } else if self.ui.is_thinking() {
                ("mic ◌ muted", "agent … thinking")
            } else if self.playing && controls.barge_in() {
                ("mic ● live", "agent 🔊 speaking — talk to interrupt")
            } else if self.playing {
                ("mic ◌ muted", "agent 🔊 speaking — press b for barge-in")
            } else {
                ("mic ● live", "agent ○ idle")
            };
            let status = format!("{mic}  ·  {agent_pill}  ·  {}", self.ui.aec_pill());
            let speaker = if controls.speaker_muted() {
                "unmute"
            } else {
                "mute"
            };
            let agent = if controls.agent_muted() {
                "agent-on"
            } else {
                "agent-off"
            };
            let barge = if controls.barge_in() {
                "barge-off"
            } else {
                "barge-on"
            };
            (
                status,
                format!("q:quit  m:{speaker}  a:{agent}  b:{barge}  u:mic"),
            )
        }

        fn draw_footer_at_cursor(
            &mut self,
            out: &mut impl Write,
            controls: &RuntimeControls,
        ) -> std::io::Result<()> {
            if !self.ready {
                return Ok(());
            }
            self.draw_footer_text(out, self.footer(controls))
        }

        fn draw_footer_text(
            &mut self,
            out: &mut impl Write,
            text: (String, String),
        ) -> std::io::Result<()> {
            write!(out, "{}\r\n{}", text.0, text.1)?;
            out.flush()?;
            self.footer_drawn = true;
            self.footer_text = Some(text);
            Ok(())
        }

        fn refresh_footer(
            &mut self,
            out: &mut impl Write,
            controls: &RuntimeControls,
        ) -> std::io::Result<()> {
            if !self.ready {
                return Ok(());
            }
            let next = self.footer(controls);
            if self.footer_drawn && self.footer_text.as_ref() == Some(&next) {
                return Ok(());
            }
            self.clear_footer(out)?;
            self.draw_footer_text(out, next)
        }

        fn apply(
            &mut self,
            event: LoopEvent,
            out: &mut impl Write,
            controls: &RuntimeControls,
        ) -> std::io::Result<()> {
            if matches!(event, LoopEvent::Partial { .. }) {
                self.clear_footer(out)?;
                self.clear_partial(out)?;
                self.ui.apply(event);
                if let Some(line) = self.ui.transcript_text().lines().last() {
                    write!(out, "{line}\r\n")?;
                    self.partial_visible = true;
                }
                return self.refresh_footer(out, controls);
            }
            let playback_ended = matches!(&event, LoopEvent::Playback { playing: false });
            if matches!(&event, LoopEvent::Ready) {
                self.ready = true;
            }
            if let LoopEvent::Playback { playing } = &event {
                self.playing = *playing;
            }
            self.ui.apply(event);

            let mut final_agent = None;
            for line in self.ui.take_display() {
                if let Some(text) = line.strip_prefix("Agent: ") {
                    final_agent = Some(text.to_owned());
                    continue;
                }
                self.commit_live_reply();
                self.clear_footer(out)?;
                self.clear_partial(out)?;
                write!(out, "{line}\r\n")?;
            }
            if self.ready && !self.hint_shown {
                write!(out, "Listening… speak to start. q or Ctrl+C to quit.\r\n")?;
                self.hint_shown = true;
            }
            if let Some((_, text)) = self.ui.agent_stream() {
                self.queue_agent_text(&text);
            }
            let final_fragment = final_agent.is_some();
            if let Some(text) = final_agent {
                self.queue_agent_text(&text);
            }
            self.flush_paragraphs(out, final_fragment, controls)?;
            if playback_ended {
                self.commit_live_reply();
            }
            self.refresh_footer(out, controls)
        }

        fn finish(&mut self, out: &mut impl Write, show_stats: bool) -> std::io::Result<()> {
            self.commit_live_reply();
            self.clear_footer(out)?;
            if show_stats && !self.ui.latency_line().contains('—') {
                write!(out, "stats: {}\r\n", compact_stats(self.ui.latency_line()))?;
            }
            write!(out, "\r\n")?;
            out.flush()
        }
    }

    fn compact_stats(line: &str) -> String {
        line.replace("STT", "stt")
            .replace("TTFT", "llm")
            .replace("TTFB", "tts")
    }

    fn write_terminal_text(out: &mut impl Write, text: &str) -> std::io::Result<()> {
        for piece in text.split_inclusive('\n') {
            if let Some(line) = piece.strip_suffix('\n') {
                write!(out, "{}\r\n", line.strip_suffix('\r').unwrap_or(line))?;
            } else {
                write!(out, "{piece}")?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    #[test]
    fn final_stats_are_compact() {
        assert_eq!(
            compact_stats("STT 375ms  TTFT 255ms  TTFB 1.67s  total 7.64s"),
            "stt 375ms  llm 255ms  tts 1.67s  total 7.64s"
        );
    }

    #[cfg(test)]
    #[test]
    fn model_newlines_become_terminal_crlf() {
        let mut out = Vec::new();
        write_terminal_text(&mut out, "one\ntwo\n\nthree").expect("write model text");
        assert_eq!(String::from_utf8(out).unwrap(), "one\r\ntwo\r\n\r\nthree");
    }

    pub fn run_conversation_tui(config: AgentConfig, cancel: Cancel, barge_in: bool) -> Result<()> {
        let (event_tx, event_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let controls = RuntimeControls::with_auto_timeout(
            barge_in,
            config.auto_timeout_mic_mute_ms,
            config.auto_timeout_exit_ms,
        );
        let loop_cancel = cancel.clone();
        let loop_controls = controls.clone();
        thread::spawn(move || {
            let result =
                run_live_with_controls(&config, loop_cancel, Some(event_tx), loop_controls);
            let _ = done_tx.send(result);
        });

        let _guard = RawTerminalGuard;
        let mut out = stdout();
        let mut renderer = InlineRenderer::new();
        let mut raw_mode = false;

        let mut quit_requested = false;
        let outcome = loop {
            while let Ok(event) = event_rx.try_recv() {
                if matches!(&event, LoopEvent::Ready) && !raw_mode {
                    enable_raw_mode().map_err(Error::from)?;
                    raw_mode = true;
                }
                renderer
                    .apply(event, &mut out, &controls)
                    .map_err(Error::from)?;
            }
            if let Ok(done) = done_rx.try_recv() {
                break done;
            }
            if raw_mode && controls.mic_muted() != renderer.last_mic_muted {
                renderer.last_mic_muted = controls.mic_muted();
                renderer
                    .refresh_footer(&mut out, &controls)
                    .map_err(Error::from)?;
            }
            if raw_mode && event::poll(Duration::from_millis(50)).map_err(Error::from)? {
                if let Event::Key(key) = event::read().map_err(Error::from)? {
                    if key.kind == KeyEventKind::Press
                        && (key.code == KeyCode::Char('q')
                            || key.code == KeyCode::Esc
                            || (key.code == KeyCode::Char('c')
                                && key.modifiers.contains(KeyModifiers::CONTROL)))
                    {
                        quit_requested = key.code == KeyCode::Char('q');
                        controls.touch_idle();
                        cancel.shutdown();
                    } else if key.kind == KeyEventKind::Press {
                        controls.touch_idle();
                        match key.code {
                            KeyCode::Char('b') => {
                                controls.toggle_barge_in();
                            }
                            KeyCode::Char('m') => {
                                controls.toggle_speaker_muted();
                                renderer.playing = false;
                            }
                            KeyCode::Char('a') => {
                                controls.toggle_agent_muted();
                            }
                            KeyCode::Char('u') => {
                                controls.unmute_mic();
                            }
                            _ => {
                                // Any other key still resets the idle clock.
                            }
                        }
                        renderer
                            .refresh_footer(&mut out, &controls)
                            .map_err(Error::from)?;
                    }
                }
            }
        };

        renderer
            .finish(&mut out, quit_requested)
            .map_err(Error::from)?;
        match outcome {
            Ok(_) | Err(Error::Cancelled) => Ok(()),
            Err(err) => Err(err),
        }
    }
}

#[cfg(not(coverage))]
pub use live_terminal::run_conversation_tui;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use syllabix_core::{TurnId, TurnTimings};

    #[test]
    fn apply_builds_transcript_and_latency() {
        let mut ui = TranscriptUi::default();
        ui.apply(LoopEvent::User {
            turn: TurnId(0),
            text: "hello".into(),
            language: "en".into(),
        });
        ui.apply(LoopEvent::Assistant {
            turn: TurnId(0),
            text: "hi".into(),
            is_last: false,
        });
        ui.apply(LoopEvent::Assistant {
            turn: TurnId(0),
            text: " there".into(),
            is_last: true,
        });
        ui.apply(LoopEvent::Timings {
            turn: TurnId(0),
            timings: TurnTimings {
                stt: Duration::from_millis(10),
                ttft: Duration::from_millis(20),
                ttfb: Duration::from_millis(30),
                total: Duration::from_millis(40),
            },
        });
        assert_eq!(ui.transcript_text(), "You: hello\nAgent: hi there");
        assert_eq!(
            ui.latency_line(),
            "STT 10ms  TTFT 20ms  TTFB 30ms  total 40ms"
        );
    }

    #[test]
    fn partial_user_text_is_replaced_by_the_final_transcript() {
        let mut ui = TranscriptUi::default();
        ui.apply(LoopEvent::Partial {
            turn: TurnId(4),
            text: "hello wor".into(),
        });
        assert_eq!(ui.transcript_text(), "You: hello wor…");
        ui.apply(LoopEvent::Partial {
            turn: TurnId(4),
            text: "hello world".into(),
        });
        assert_eq!(ui.transcript_text(), "You: hello world…");
        ui.apply(LoopEvent::User {
            turn: TurnId(4),
            text: "hello world".into(),
            language: "en".into(),
        });
        assert_eq!(ui.transcript_text(), "You: hello world");
    }

    #[test]
    fn detected_language_is_tagged_on_the_user_line() {
        let mut ui = TranscriptUi::default();
        ui.apply(LoopEvent::User {
            turn: TurnId(0),
            text: "bonjour".into(),
            language: "fr".into(),
        });
        assert!(ui.transcript_text().starts_with("You [fr]: bonjour"));
        ui.apply(LoopEvent::User {
            turn: TurnId(1),
            text: "hello".into(),
            language: "en".into(),
        });
        assert!(ui.transcript_text().contains("\nYou: hello"));
    }

    #[test]
    fn tool_events_stay_out_of_the_transcript() {
        let mut ui = TranscriptUi::default();
        // User turn so flush_agent has something to flush.
        ui.apply(LoopEvent::User {
            turn: TurnId(0),
            text: "hey".into(),
            language: "en".into(),
        });
        // Two call events + their result events for the same turn.
        for kind in ["call", "result", "call", "result"] {
            ui.apply(LoopEvent::Tool {
                turn: TurnId(0),
                event: syllabix_core::ToolTurnEvent {
                    kind: kind.into(),
                    name: "shell".into(),
                    call_id: "c1".into(),
                    arguments: "{\"argv\":[\"df\"]}".into(),
                    content: "exit: 0\nstdout:\n…".into(),
                },
            });
        }
        // Final assistant reply flushes the turn.
        ui.apply(LoopEvent::Assistant {
            turn: TurnId(0),
            text: "Done.".into(),
            is_last: true,
        });
        let text = ui.transcript_text();
        assert!(text.contains("Agent: Done."), "got: {text}");
        assert!(!text.contains("tools-called:"), "got: {text}");
        assert!(!text.contains("exit:"), "got: {text}");
        assert!(!text.contains("[tool"), "got: {text}");
    }

    #[test]
    fn think_tokens_are_hidden_until_the_spoken_reply() {
        let mut ui = TranscriptUi::default();
        ui.apply(LoopEvent::User {
            turn: TurnId(1),
            text: "hey".into(),
            language: "en".into(),
        });
        ui.apply(LoopEvent::Assistant {
            turn: TurnId(1),
            text: "<think>plan".into(),
            is_last: false,
        });
        assert_eq!(ui.transcript_text(), "You: hey");
        ui.apply(LoopEvent::Assistant {
            turn: TurnId(1),
            text: "</think> I'm well.".into(),
            is_last: true,
        });
        assert_eq!(ui.transcript_text(), "You: hey\nAgent:  I'm well.");
    }

    #[test]
    fn empty_ui_shows_listen_hint() {
        let ui = TranscriptUi::default();
        assert!(ui.transcript_text().contains("Listening"));
        assert!(ui.latency_line().contains("STT"));
    }

    #[test]
    fn aec_pill_tracks_native_state() {
        let mut ui = TranscriptUi::default();
        assert_eq!(ui.aec_pill(), "AEC ○ off");
        ui.apply(LoopEvent::Aec {
            active: true,
            restart_required: false,
        });
        assert_eq!(ui.aec_pill(), "AEC ● on");
        ui.apply(LoopEvent::Aec {
            active: false,
            restart_required: true,
        });
        assert_eq!(ui.aec_pill(), "AEC ○ off — restart to fix");
    }
}
