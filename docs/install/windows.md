# Install on Windows

Artifact: `syllabix-Windows-x86_64.exe`.

```powershell
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/syllabix-Windows-x86_64.exe -o syllabix.exe
curl.exe -L https://github.com/syllabix-ai/syllabix/releases/latest/download/SHA256SUMS -o SHA256SUMS
certutil -hashfile syllabix.exe SHA256
.\syllabix.exe run
```

Compare the `certutil` hash to the `SHA256SUMS` line for `syllabix-Windows-x86_64.exe` before you run.

## SmartScreen

On first open, SmartScreen may warn that the app is unrecognized. Choose **More info** → **Run anyway** after the checksum matches.

## Microphone

Settings → Privacy → Microphone. `run` needs a capture device and a playback device at launch.

## Models

First `run` fetches the default stack (~2.2 GB) into `%LOCALAPPDATA%\syllabix\cache\models\v1` (or `%SYLLABIX_CACHE_DIR%\models\v1`). The Windows artifact uses portable CPU for whisper.cpp and llama.cpp.

Cache and other OS: [../install.md](../install.md).
