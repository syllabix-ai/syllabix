//! Speak one line of text into a WAV file with the default Pocket TTS engine.
//!
//! Arguments: `<text> <out.wav>`. First run downloads Pocket TTS weights into
//! the shared cache (`SYLLABIX_CACHE_DIR` or `~/.cache/syllabix` by default).

use std::process::ExitCode;

use syllabix_tts::pocket::PocketTts;
use syllabix_tts::{speak_to_wav, Cancel};

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    match run() {
        Ok(path) => {
            println!("{path}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("speak-wav-local: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Res<String> {
    let mut args = std::env::args().skip(1);
    let (Some(text), Some(path)) = (args.next(), args.next()) else {
        return Err("usage: speak-wav-local <text> <out.wav>".into());
    };
    let mut tts = PocketTts::load()?;
    speak_to_wav(&mut tts, text, &path, &Cancel::new())?;
    Ok(path)
}
