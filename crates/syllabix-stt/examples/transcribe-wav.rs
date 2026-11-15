//! Transcribe one WAV file with the OpenAI-compatible remote engine.
//!
//! Arguments: `<file.wav> <base-url> <model> [language]`. The API key comes
//! from `SYLLABIX_STT_API_KEY` only, or pass a key directly to
//! [`RemoteStt::new`]. Needs only the `remote` feature, so it builds without
//! CMake or ONNX Runtime.

use std::process::ExitCode;

use syllabix_stt::remote::{resolve_api_key, RemoteStt, RemoteSttSettings};
use syllabix_stt::{transcribe_wav, Cancel, STT_LANGUAGE};

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    match run() {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("transcribe-wav: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Res<String> {
    let mut args = std::env::args().skip(1);
    let (Some(path), Some(base_url), Some(model)) = (args.next(), args.next(), args.next()) else {
        return Err("usage: transcribe-wav <file.wav> <base-url> <model> [language]".into());
    };
    let language = args.next().unwrap_or_else(|| STT_LANGUAGE.to_string());

    let settings = RemoteSttSettings::new(base_url, model, language);
    let api_key = resolve_api_key(|name| std::env::var(name).ok())?;
    let mut stt = RemoteStt::new(settings, api_key);

    Ok(transcribe_wav(&mut stt, path, &Cancel::new())?.text)
}
