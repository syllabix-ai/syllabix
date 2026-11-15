//! Transcribe one WAV file with the OpenAI-compatible remote engine.
//!
//! Arguments: `<file.wav> <base-url> <model> [language]`. The API key comes
//! from `SYLLABIX_STT_API_KEY` only. Needs only the `remote` feature, so it
//! builds without CMake or ONNX Runtime.

use std::fs::File;
use std::process::ExitCode;

use syllabix_stt::audio::{f32_to_i16, i16_to_f32, read_wav, FrameSplitter, PcmConverter};
use syllabix_stt::audio::{PcmFormat, WavPcm};
use syllabix_stt::remote::{resolve_api_key, RemoteStt, RemoteSttSettings};
use syllabix_stt::{AudioFrame, Cancel, Stt, TurnId, Utterance, STT_LANGUAGE};

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

    let frames = pipeline_frames(&read_wav(File::open(&path)?)?)?;
    let settings = RemoteSttSettings::from_parts(Some(&base_url), &model, &language)?;
    let api_key = resolve_api_key(|name| std::env::var(name).ok())?;
    let mut stt = RemoteStt::new(settings, api_key);

    let utterance = Utterance {
        turn: TurnId(0),
        frames,
    };
    Ok(stt.transcribe(&utterance, &Cancel::new())?.text)
}

/// Any-rate WAV to the 16 kHz mono frames every engine expects.
fn pipeline_frames(wav: &WavPcm) -> Res<Vec<AudioFrame>> {
    let mut converter = PcmConverter::new(wav.format, PcmFormat::v0())?;
    let mut mono = converter.push(&i16_to_f32(&wav.samples));
    mono.extend(converter.flush());
    let mut splitter = FrameSplitter::new();
    let mut frames = splitter.push(&f32_to_i16(&mono))?;
    frames.extend(splitter.flush()?);
    Ok(frames)
}
