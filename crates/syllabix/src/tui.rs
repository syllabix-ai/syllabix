//! Transcript + latency TUI for `syllabix run`.

use syllabix_core::{LoopEvent, ThinkFilter, TurnId};

/// Rolling transcript shown in the TUI.
#[derive(Debug, Default)]
#[cfg_attr(coverage, allow(dead_code))]
pub struct TranscriptUi {
    lines: Vec<String>,
    current_agent: Option<(TurnId, String)>,
    think: ThinkFilter,
    latency: String,
}

#[cfg_attr(coverage, allow(dead_code))]
impl TranscriptUi {
    /// Apply one pipeline event.
    pub fn apply(&mut self, event: LoopEvent) {
        match event {
            LoopEvent::User { text, language, .. } => {
                self.flush_agent();
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
            LoopEvent::Timings { timings, .. } => {
                self.latency = timings.format_line();
            }
        }
    }

    fn flush_agent(&mut self) {
        if let Some((_, text)) = self.current_agent.take() {
            if !text.is_empty() {
                self.lines.push(format!("Agent: {text}"));
            }
        }
        self.think = ThinkFilter::default();
    }

    /// Visible transcript, including an in-flight assistant line.
    pub fn transcript_text(&self) -> String {
        let mut lines = self.lines.clone();
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

    /// Scroll so the last wrapped line stays in a pane of `width` × `height`.
    pub fn transcript_scroll(&self, width: u16, height: u16) -> u16 {
        scroll_offset(&self.transcript_text(), width, height)
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

fn wrapped_line_count(text: &str, width: usize) -> usize {
    if width == 0 {
        return 0;
    }
    text.split('\n')
        .map(|line| {
            let chars = line.chars().count();
            if chars == 0 {
                1
            } else {
                chars.div_ceil(width)
            }
        })
        .sum()
}

fn scroll_offset(text: &str, width: u16, height: u16) -> u16 {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 {
        return 0;
    }
    wrapped_line_count(text, width)
        .saturating_sub(height)
        .min(u16::MAX as usize) as u16
}

#[cfg(not(coverage))]
mod live_terminal {
    use super::TranscriptUi;
    use std::io::stdout;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    use crossterm::execute;
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
    use ratatui::Terminal;
    use syllabix_core::{run_live, AgentConfig, Cancel, Error, Result};

    struct RawTerminalGuard;

    impl Drop for RawTerminalGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            let _ = execute!(stdout(), LeaveAlternateScreen);
        }
    }

    pub fn run_conversation_tui(
        config: AgentConfig,
        cancel: Cancel,
        turn_debug: Option<syllabix_core::TurnDebug>,
        barge_in: bool,
    ) -> Result<()> {
        let (event_tx, event_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let loop_cancel = cancel.clone();
        thread::spawn(move || {
            let result = run_live(&config, loop_cancel, Some(event_tx), turn_debug, barge_in);
            let _ = done_tx.send(result);
        });

        enable_raw_mode().map_err(Error::from)?;
        execute!(stdout(), EnterAlternateScreen).map_err(Error::from)?;
        let _guard = RawTerminalGuard;
        let backend = CrosstermBackend::new(stdout());
        let mut terminal = Terminal::new(backend).map_err(Error::from)?;
        let mut ui = TranscriptUi::default();

        let outcome = loop {
            while let Ok(event) = event_rx.try_recv() {
                ui.apply(event);
            }
            if let Ok(done) = done_rx.try_recv() {
                break done;
            }
            if event::poll(Duration::from_millis(50)).map_err(Error::from)? {
                if let Event::Key(key) = event::read().map_err(Error::from)? {
                    if key.kind == KeyEventKind::Press
                        && (key.code == KeyCode::Char('q')
                            || key.code == KeyCode::Esc
                            || (key.code == KeyCode::Char('c')
                                && key.modifiers.contains(KeyModifiers::CONTROL)))
                    {
                        cancel.shutdown();
                    }
                }
            }
            terminal
                .draw(|frame| {
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Min(3), Constraint::Length(3)])
                        .split(frame.size());
                    let inner_width = chunks[0].width.saturating_sub(2);
                    let inner_height = chunks[0].height.saturating_sub(2);
                    let text = ui.transcript_text();
                    let transcript = Paragraph::new(text.clone())
                        .wrap(Wrap { trim: false })
                        .scroll((ui.transcript_scroll(inner_width, inner_height), 0))
                        .block(Block::default().borders(Borders::ALL).title("syllabix"));
                    let latency = Paragraph::new(ui.latency_line().to_string())
                        .block(Block::default().borders(Borders::ALL).title("latency"));
                    frame.render_widget(transcript, chunks[0]);
                    frame.render_widget(latency, chunks[1]);
                })
                .map_err(Error::from)?;
        };

        drop(terminal);
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
    fn transcript_scroll_keeps_the_last_line_visible() {
        let mut ui = TranscriptUi::default();
        for i in 0..20 {
            ui.apply(LoopEvent::User {
                turn: TurnId(i),
                text: format!("line{i}"),
                language: "en".into(),
            });
        }
        let text = ui.transcript_text();
        assert!(text.contains("line19"));
        assert_eq!(ui.transcript_scroll(20, 4), 16);
    }

    #[test]
    fn empty_ui_shows_listen_hint() {
        let ui = TranscriptUi::default();
        assert!(ui.transcript_text().contains("Listening"));
        assert!(ui.latency_line().contains("STT"));
    }
}
