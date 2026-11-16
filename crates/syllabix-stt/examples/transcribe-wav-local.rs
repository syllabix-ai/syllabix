//! Transcribe one WAV file with the default Moonshine ONNX engine.
//!
//! Arguments: `<file.wav>`. First run downloads the selected model into the
//! shared cache (`SYLLABIX_CACHE_DIR` or `~/.cache/syllabix` by default).

use std::process::ExitCode;

use syllabix_stt::moonshine::MoonshineStt;
use syllabix_stt::{transcribe_wav, Cancel, SttModel};

type Res<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    match run() {
        Ok(text) => {
            println!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("transcribe-wav-local: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Res<String> {
    let Some(path) = std::env::args().nth(1) else {
        return Err("usage: transcribe-wav-local <file.wav>".into());
    };
    let mut stt = MoonshineStt::load(SttModel::MoonshineStreamingSmall)?;
    Ok(transcribe_wav(&mut stt, path, &Cancel::new())?.text)
}
