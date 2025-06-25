# RealtimeSTT-mac
_A lightning-fast Apple Silicon-optimised Speech-to-Text & Voice-Activity-Detection library_

---

## About This Fork

This repository is a **macOS-only**, **Apple Silicon-only** fork of the excellent
[RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) project by **Kolja Beigel**.

### What Changed ❓

| Area | Upstream | This fork |
|------|----------|-----------|
| **Target HW** | cross-platform (CPU & CUDA) | **Apple Silicon only** |
| **Backend** | `faster-whisper` + CTranslate2 (generic) | `faster-whisper` + **CTranslate2 built with Apple _Accelerate_** |
| **Packaging** | `setup.py`, requirements.txt | **PEP 621** (`pyproject.toml`) + **uv** |
| **Porcupine** | pvporcupine==1.9.5 did not support Apple Silicon | pvporcupine>=3.0.0 does support Apple Silicon! |
| **Packaging** | `setup.py`, requirements.txt | **PEP 621** (`pyproject.toml`) + **uv** |
| **Server / WebSocket** | optional STT server & client | **Removed** – local library only |
| **Docker images** | CUDA / CPU images | **Removed** |
| **Wheels** | fetched from PyPI | **Bundled fat-wheels** (contain the Accelerate dylib) |

Big thanks to **Kolja Beigel** for the original codebase – all core algorithms,
examples and documentation originate from his work.
This fork simply re-targets the library for M-series Macs and trims anything
that is not useful on that platform.

---

## Features

* **Realtime VAD**
  * WebRTC-VAD (fast)
  * Optional Silero VAD (accurate, runs via PyTorch)
* **Realtime & Batch Transcription**
  Powered by `faster-whisper`, executed with CTranslate2 + Apple Accelerate
* **Wake-Word Detection**
  * Picovoice Porcupine (`pvporcupine`)
  * OpenWakeWord (`openwakeword`)
* **Low-latency audio pipeline**
  < 200 ms from speech to text on an M2 Pro
* **Pythonic callbacks** for start/stop, partial results, wake-word events
* **Pure Python install** – _no_ Xcode project, no CMake build required

---

## Installation (uv)

> Requires **macOS 12+** and **M-series** CPU (M1/M2/M3).

```bash
# Initialise a new uv project (if you haven't already)
uv init -p 3.11            # or -p 3.12

# Add RealtimeSTT-mac from Git or PyPI
uv add realtimestt-mac

# Activate & run
uv run python - <<'PY'
from RealtimeSTT import AudioToTextRecorder

rec = AudioToTextRecorder(device="cpu")  # Apple Accelerate is used automatically
print("Speak now…")
while True:
    rec.text(print)
PY
```

The wheels directory is bundled inside the package and contains pre-built,
self-contained CTranslate2 binaries for:

- `cp311-macosx_arm64`
- `cp312-macosx_arm64`

_No CUDA, x86_64 or Windows/Linux wheels are shipped._

---

## Quick Example

```python
from RealtimeSTT import AudioToTextRecorder

def on_text(txt):
    print("📝", txt)

with AudioToTextRecorder(
        device="cpu",                   # always "cpu" on Apple Silicon
        enable_realtime_transcription=True,
        wakeword_backend="pvporcupine", # say "Hey Siri" etc.
) as recorder:
    print("Speak or say your wake-word…")
    while True:
        recorder.text(on_text)
```

---

## Removing Server / Docker Components

The old WebSocket server, CLI client, Dockerfiles and GPU requirements have
been removed.  If you need a networked STT service you can still run the
upstream project.

---

## Dependencies

| Package | Why it stays |
|---------|--------------|
| **PyAudio** | low-latency microphone input |
| **torch / torchaudio** | Silero VAD & OpenWakeWord |
| **faster-whisper** | Whisper inference |
| **ctranslate2** | Execution backend (Accelerate build) |
| **pvporcupine, openwakeword** | Wake-word detection |
| **scipy, soundfile, webrtcvad-wheels** | signal processing |

_Removed: websockets, websocket-client, halo (spinner) and all CUDA-specific
dependencies._

---

## Limitations

* **Apple Silicon only** – will not install on Intel Macs or other OSes
* **CPU backend only** – Accelerate uses CPU + NEON; GPU (Metal) is not used
* **No CUDA / ROCm** – use upstream for NVIDIA/AMD GPUs
* **No server mode** – library-only usage

---

## Credits

* Original code ⊕ algorithm design: **Kolja Beigel**
  <https://github.com/KoljaB/RealtimeSTT>
* Apple Silicon port & maintenance: **Tom Whiting**
  <https://github.com/tomWhiting/RealtimeSTT-mac>

Licensed under the MIT License (see **LICENSE** file).

Enjoy ultra-fast transcription on your M-series Mac!
PRs and feedback welcome.
