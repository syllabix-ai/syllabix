//! Speak one line of text into a WAV file with the OpenAI-compatible remote engine.
//!
//! Arguments: `<text> <out.wav> <base-url> <model> [voice]`. The API key comes
//! from `SYLLABIX_TTS_API_KEY` only. Needs only the `remote` feature, so it
//! builds without CMake or ONNX Runtime.

use std::fs::File;
use std::process::ExitCode;

use syllabix_tts::audio::{write_wav, PcmFormat, WavPcm};
use syllabix_tts::remote::{resolve_api_key, RemoteTts, RemoteTtsSettings, DEFAULT_VOICE};
use syllabix_tts::{Cancel, GenerationId, TokenChunk, Tts, TurnId};

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

    let settings = RemoteTtsSettings::from_parts(Some(&base_url), &model, &voice)?;
    let api_key = resolve_api_key(|name| std::env::var(name).ok())?;
    let mut tts = RemoteTts::new(settings, api_key);

    // The whole text as one final token: the engine speaks it sentence by sentence.
    let token = TokenChunk {
        turn: TurnId(0),
        generation: GenerationId(0),
        index: 0,
        text,
        is_last: true,
    };
    let samples = tts
        .synthesize_chunk(&token, &Cancel::new())?
        .into_iter()
        .flat_map(|chunk| chunk.samples)
        .collect();

    // Every engine returns 16 kHz mono PCM.
    let wav = WavPcm {
        format: PcmFormat::v0(),
        samples,
    };
    write_wav(File::create(&path)?, &wav)?;
    Ok(path)
}
