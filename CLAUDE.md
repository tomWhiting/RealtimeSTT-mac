# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RealtimeSTT-mac is an Apple Silicon-optimized, real-time Speech-to-Text library specifically designed for M-series Macs. It's a macOS-only fork of the original RealtimeSTT project, re-targeted for Apple Silicon performance with sub-200ms latency.

**Hardware Requirements**: macOS 12+ with M-series CPU (M1/M2/M3) only - no Intel Mac support.

## Development Commands

### Environment Setup
```bash
# Initialize project with uv (recommended)
uv init -p 3.11
uv add realtimestt-mac

# Development dependencies
uv add --dev pytest black isort
```

### Testing
```bash
# Quick validation test
uv run python tests/simple_test.py

# Comprehensive test with rich UI
uv run python tests/realtimestt_test.py

# Wake word tests
uv run python tests/openwakeword_test.py
uv run python tests/test_apple_silicon_wake_word.py

# VAD testing
uv run python tests/vad_test.py
```

### Code Quality
```bash
# Format code
uv run black .
uv run isort .

# Run tests
uv run pytest
```

## Core Architecture

### Main Components
- **`RealtimeSTT/audio_recorder.py`**: Central `AudioToTextRecorder` class (33k+ lines) containing the complete transcription pipeline
- **`RealtimeSTT/audio_input.py`**: `AudioInput` class for PyAudio microphone handling and device management
- **`RealtimeSTT/safepipe.py`**: Thread-safe multiprocessing communication via `ParentPipe`

### Multiprocessing Design
- **Main Process**: Audio capture, VAD, callback management
- **Worker Process**: Isolated transcription using faster-whisper models  
- **Communication**: Custom SafePipe for thread-safe IPC between processes

### Key Features
- **Real-time VAD**: WebRTC-VAD (fast) or Silero VAD (accurate, PyTorch-based)
- **Wake Word Detection**: Porcupine (`pvporcupine>=3.0.0`) or OpenWakeWord support
- **Model Optimization**: Pre-converted FP32 models to eliminate conversion overhead
- **Callback System**: 17+ configurable callbacks for different lifecycle events

## Apple Silicon Optimizations

### CTranslate2 Backend
- Uses bundled CTranslate2 wheels with Apple Accelerate support
- CPU-only execution optimized for M-series processors
- Wheels located in `./wheels/` directory for Python 3.11/3.12

### Model Management
- **Registry**: `faster_whisper_models.json` maps model names to HuggingFace repos
- **Pre-conversion**: FP32 models avoid runtime conversion overhead
- **Loading**: Models are lazy-loaded when needed

## Development Patterns

### Basic Usage Pattern
```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(device="cpu")  # Always "cpu" on Apple Silicon
recorder.text(print)  # Blocking call for single transcription
```

### Callback-Driven Pattern
```python
def on_text(txt):
    print("📝", txt)

with AudioToTextRecorder(
    device="cpu",
    enable_realtime_transcription=True,
    wakeword_backend="pvporcupine"
) as recorder:
    while True:
        recorder.text(on_text)
```

### Configuration Philosophy
- **50+ parameters** available for fine-tuning
- **Sensible defaults** work out-of-the-box
- **Real-time focus**: Optimized for interactive applications
- **Process isolation**: Transcription worker runs in separate process for stability

## Important Dependencies

### Core Runtime
- **PyAudio**: Low-latency microphone input
- **faster-whisper**: Whisper model inference  
- **ctranslate2**: Apple Accelerate-optimized execution (bundled wheels)
- **torch/torchaudio**: Required for Silero VAD and OpenWakeWord

### Optional Features
- **pvporcupine**: Porcupine wake word engine
- **openwakeword**: Alternative wake word engine
- **webrtcvad-wheels**: WebRTC voice activity detection

## Limitations

- **Apple Silicon exclusive**: Will not work on Intel Macs or other platforms
- **CPU backend only**: No GPU/Metal acceleration used
- **Library-only**: No server/WebSocket components (removed from upstream)
- **macOS 12+ required**: System-level dependencies for Apple Accelerate

## Performance Characteristics

- **Latency**: <200ms speech-to-text on M2 Pro
- **Model Loading**: 2-5x faster with pre-converted FP32 models  
- **Memory Efficient**: Optimized for laptop/mobile deployment
- **Battery Optimized**: CPU-only processing designed for power efficiency