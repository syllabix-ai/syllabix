//! Speak one line of text into a WAV file with the OpenAI-compatible remote engine.
//!
//! Arguments: `<text> <out.wav> <base-url> <model> [voice]`. The API key comes
//! from `SYLLABIX_TTS_API_KEY` only, or pass a key directly to
//! [`RemoteTts::new`]. Needs only the `remote` feature, so it builds without
//! CMake or ONNX Runtime.

use std::process::ExitCode;

use syllabix_tts::remote::{resolve_api_key, RemoteTts, RemoteTtsSettings, DEFAULT_VOICE};
use syllabix_tts::{speak_to_wav, Cancel};

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    match run() {
        Ok(path) => {
            println!("{path}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("speak-wav: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Res<String> {
    let mut args = std::env::args().skip(1);
    let (Some(text), Some(path), Some(base_url), Some(model)) =
        (args.next(), args.next(), args.next(), args.next())
    else {
        return Err("usage: speak-wav <text> <out.wav> <base-url> <model> [voice]".into());
    };
    let voice = args.next().unwrap_or_else(|| DEFAULT_VOICE.to_string());

    let settings = RemoteTtsSettings::new(base_url, model, voice);
    let api_key = resolve_api_key(|name| std::env::var(name).ok())?;
    let mut tts = RemoteTts::new(settings, api_key);

    speak_to_wav(&mut tts, text, &path, &Cancel::new())?;
    Ok(path)
}
