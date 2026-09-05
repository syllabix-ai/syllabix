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
    tool_call_count: usize,
}

#[cfg_attr(coverage, allow(dead_code))]
impl TranscriptUi {
    /// Apply one pipeline event.
    pub fn apply(&mut self, event: LoopEvent) {
        match event {
            LoopEvent::Ready | LoopEvent::Playback { .. } => {}
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
                // Non-English turns carry a visible language tag (fixed or
                // auto-detected); plain English stays untagged.
                if language.is_empty() || language == "en" {
                    self.lines.push(format!("You: {text}"));
                } else {
                    self.lines.push(format!("You [{language}]: {text}"));
                }
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
                match &mut self.current_agent {
                    Some((id, buf)) if *id == turn => buf.push_str(&spoken),
                    _ => self.current_agent = Some((turn, spoken)),
                }
                if is_last {
                    self.flush_agent();
                }
            }
            LoopEvent::Tool { event, .. } => {
                if event.kind == "call" {
                    self.tool_call_count += 1;
                }
            }
            LoopEvent::Timings { timings, .. } => {
                self.latency = timings.format_line();
            }
        }
    }

    fn flush_agent(&mut self) {
        if let Some((_, text)) = self.current_agent.take() {
            if self.tool_call_count > 0 {
                self.lines
                    .push(format!("tools-called:{}", self.tool_call_count));
                self.tool_call_count = 0;
            }
            if !text.is_empty() {
                self.lines.push(format!("Agent: {text}"));
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
            } else if self.think.in_think() {
                lines.push("Agent: …".into());
            }
        } else if self.think.in_think() {
            lines.push("Agent: …".into());
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
        printed_lines: usize,
        ready: bool,
        playing: bool,
        footer_drawn: bool,
        partial_visible: bool,
        last_mic_muted: bool,
    }

    impl InlineRenderer {
        fn new() -> Self {
            Self {
                ui: TranscriptUi::default(),
                printed_lines: 0,
                ready: false,
                playing: false,
                footer_drawn: false,
                partial_visible: false,
                last_mic_muted: false,
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

        fn footer(&self, controls: &RuntimeControls) -> (String, String) {
            let status = if controls.mic_muted() {
                "(mic muted — press u to listen)".to_string()
            } else if controls.agent_muted() {
                "(agent muted, listening...)".to_string()
            } else if controls.speaker_muted() {
                "(speaker muted, agent responding)".to_string()
            } else if self.playing && controls.barge_in() {
                "(audio playing, listening)".to_string()
            } else if self.playing {
                "(audio playing, enable barge-in to listen and interrupt)".to_string()
            } else {
                "(listening...)".to_string()
            };
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
                format!("q:quit  m:{speaker}  a:{agent}  b:{barge}  u:mic  c:config"),
            )
        }

        fn draw_footer(
            &mut self,
            out: &mut impl Write,
            controls: &RuntimeControls,
        ) -> std::io::Result<()> {
            if !self.ready {
                return Ok(());
            }
            self.clear_footer(out)?;
            let (status, help) = self.footer(controls);
            write!(out, "{status}\r\n{help}")?;
            out.flush()?;
            self.footer_drawn = true;
            Ok(())
        }

        fn apply(
            &mut self,
            event: LoopEvent,
            out: &mut impl Write,
            controls: &RuntimeControls,
        ) -> std::io::Result<()> {
            match event {
                LoopEvent::Ready => self.ready = true,
                LoopEvent::Playback { playing } => self.playing = playing,
                LoopEvent::Timings { .. } => self.ui.apply(event),
                LoopEvent::Partial { .. } => {
                    self.clear_footer(out)?;
                    self.clear_partial(out)?;
                    self.ui.apply(event);
                    if let Some(line) = self.ui.transcript_text().lines().last() {
                        write!(out, "{line}\r\n")?;
                        self.partial_visible = true;
                    }
                }
                event => {
                    self.clear_footer(out)?;
                    self.clear_partial(out)?;
                    let is_inflight_assistant =
                        matches!(&event, LoopEvent::Assistant { is_last: false, .. });
                    self.ui.apply(event);
                    if is_inflight_assistant {
                        return self.draw_footer(out, controls);
                    }
                    let transcript = self.ui.transcript_text();
                    let lines: Vec<_> = transcript.lines().collect();
                    for line in &lines[self.printed_lines.min(lines.len())..] {
                        write!(out, "{line}\r\n")?;
                    }
                    self.printed_lines = lines.len();
                }
            }
            self.draw_footer(out, controls)
        }

        fn finish(&mut self, out: &mut impl Write, show_stats: bool) -> std::io::Result<()> {
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

    #[cfg(test)]
    #[test]
    fn final_stats_are_compact() {
        assert_eq!(
            compact_stats("STT 375ms  TTFT 255ms  TTFB 1.67s  total 7.64s"),
            "stt 375ms  llm 255ms  tts 1.67s  total 7.64s"
        );
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

        let mut configure_requested = false;
        let outcome = {
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
                        .draw_footer(&mut out, &controls)
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
                                KeyCode::Char('c') | KeyCode::Char('C') => {
                                    // Plain `c` opens yaml configure; Ctrl+C quits above.
                                    configure_requested = true;
                                    cancel.shutdown();
                                }
                                _ => {
                                    // Any other key still resets the idle clock.
                                }
                            }
                            renderer
                                .draw_footer(&mut out, &controls)
                                .map_err(Error::from)?;
                        }
                    }
                }
            };

            renderer
                .finish(&mut out, quit_requested && !configure_requested)
                .map_err(Error::from)?;
            outcome
        };

        if configure_requested {
            let cwd = std::env::current_dir()?;
            let path = AgentConfig::ensure_config_file(&cwd)?;
            println!(
                "Opening {} — save, quit the editor, then rerun `syllabix run`.",
                path.display()
            );
            crate::editor::open_path_in_editor(&path)?;
            println!("Rerun `syllabix run` to apply changes.");
        }

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
    fn tool_events_render_as_compact_summary() {
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
        assert!(text.contains("tools-called:2"), "got: {text}");
        assert!(text.contains("Agent: Done."), "got: {text}");
        // Raw tool detail must not appear.
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
        assert_eq!(ui.transcript_text(), "You: hey\nAgent: …");
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
}
