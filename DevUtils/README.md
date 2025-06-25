# DevUtils - Model Conversion and Benchmarking Tools

This directory contains development utilities for optimizing Whisper models for Apple Silicon.

## Overview

The main script `convert_and_benchmark_models.py` automates the process of:

1. **Downloading** all models from `faster_whisper_models.json`
2. **Converting** FP16 models to FP32 for Apple Silicon optimization
3. **Benchmarking** both versions (load time, transcription time, accuracy)
4. **Generating** enhanced model registry with performance data
5. **Creating** READMEs for converted models with benchmark comparisons
6. **Uploading** FP32 versions to HuggingFace

## Why FP32 for Apple Silicon?

Apple Silicon (M1/M2/M3) with CTranslate2's Accelerate backend doesn't have optimized FP16 support. FP16 models are automatically converted to FP32 at load time, which happens every time you initialize a model. 

Pre-converting to FP32 eliminates this repeated conversion, resulting in:
- **2-5x faster model loading**
- **Same transcription accuracy**
- **2x larger model size** (trade-off)

## Prerequisites

### Dependencies
```bash
pip install faster-whisper huggingface-hub soundfile
```

### Required Tools
- `ct2-transformers-converter` (from CTranslate2)
- HuggingFace account with API token
- Test audio file (.wav format)

## Usage

### Quick Start

1. **Prepare test audio**: Record a short audio clip (5-10 seconds) and note the exact transcription
2. **Set up configuration**: Copy `config.example.json` to `config.json` and fill in your details
3. **Run the script**:

```bash
python convert_and_benchmark_models.py \
  --hf-token YOUR_HF_TOKEN \
  --target-org your-hf-org \
  --test-audio test_audio.wav \
  --expected-text "The exact text from your audio file"
```

### Advanced Usage

**Process specific models only:**
```bash
python convert_and_benchmark_models.py \
  --hf-token YOUR_TOKEN \
  --target-org realtimestt-mac \
  --test-audio test.wav \
  --expected-text "test transcription" \
  --models "tiny,base,small"
```

**Test single model:**
```bash
python convert_and_benchmark_models.py \
  --hf-token YOUR_TOKEN \
  --target-org realtimestt-mac \
  --test-audio test.wav \
  --expected-text "test transcription" \
  --models "tiny"
```

## Output

### Enhanced Model Registry
The script generates `enhanced_model_registry.json` with structure:

```json
{
  "tiny": {
    "fp16": {
      "repo": "Systran/faster-whisper-tiny",
      "size_mb": 39.2,
      "readme_url": "https://huggingface.co/Systran/faster-whisper-tiny/blob/main/README.md",
      "benchmarks": {
        "load_time_ms": 1200.5,
        "transcribe_time_ms": 850.2,
        "transcription_accuracy": 98.5,
        "transcription_text": "the quick brown fox..."
      }
    },
    "fp32": {
      "repo": "realtimestt-mac/tiny-fp32",
      "size_mb": 78.4,
      "readme_url": "https://huggingface.co/realtimestt-mac/tiny-fp32/blob/main/README.md",
      "benchmarks": {
        "load_time_ms": 450.1,
        "transcribe_time_ms": 840.8,
        "transcription_accuracy": 98.5,
        "transcription_text": "the quick brown fox..."
      }
    }
  }
}
```

### Generated READMEs
Each converted model gets a comprehensive README with:
- Performance comparison tables
- Usage examples
- Integration instructions for RealtimeSTT-mac
- Technical details

### HuggingFace Repositories
Models are uploaded to `{target-org}/{model-name}-fp32` with:
- Converted model files
- Generated README with benchmarks
- Proper model metadata

## Example Output

```
🚀 Starting model conversion and benchmarking pipeline
Target organization: realtimestt-mac
Test audio: test_audio.wav
Expected text: The quick brown fox jumps over the lazy dog

============================================================
Processing model: tiny
============================================================
Downloading original model: Systran/faster-whisper-tiny
Benchmarking model: /path/to/tiny_original (fp16)
Converting tiny to FP32
Benchmarking model: /path/to/tiny_fp32 (fp32)
Uploading model to realtimestt-mac/tiny-fp32
✅ Successfully processed tiny

📊 Performance Summary for tiny:
   Load Time: 1200.5ms → 450.1ms (-62.5%)
   Model Size: 39.2MB → 78.4MB (+100.0%)
   Accuracy: 98.5% → 98.5% (+0.0%)

===============================================================================
📊 PERFORMANCE SUMMARY
===============================================================================

TINY:
  Load Time Improvement: -62.5%
  Size Increase: +100.0%
  Accuracy Change: +0.0%
```

## Configuration

### Test Audio Requirements
- **Format**: WAV file
- **Duration**: 5-15 seconds recommended
- **Content**: Clear speech, representative of typical use cases
- **Quality**: Good signal-to-noise ratio

### Model Selection
You can process:
- **All models** (default): Processes every model in `faster_whisper_models.json`
- **Specific models**: Use `--models` flag with comma-separated list
- **Model subsets**: Useful for testing or gradual deployment

## Troubleshooting

### Common Issues

**`ct2-transformers-converter` not found:**
```bash
pip install ctranslate2
# or
conda install -c conda-forge ctranslate2
```

**HuggingFace authentication failed:**
- Verify your token has write permissions
- Check token is correctly set in config or command line

**Model download fails:**
- Check internet connection
- Verify model names in `faster_whisper_models.json`
- Some models may be temporarily unavailable

**Conversion fails:**
- Ensure sufficient disk space (models can be large)
- Check CTranslate2 version compatibility
- Verify input model format

### Performance Notes

- **First run**: Takes longer due to model downloads
- **Disk space**: Requires ~2-3x model size during conversion
- **Memory**: Large models (medium+) may require 8GB+ RAM
- **Time**: Full pipeline can take 1-4 hours depending on model sizes

## File Structure

```
DevUtils/
├── README.md                           # This file
├── convert_and_benchmark_models.py     # Main conversion script
├── config.example.json                 # Example configuration
└── model_conversion_workspace/         # Temporary workspace (auto-created)
```

## Integration with RealtimeSTT-mac

Once models are converted and uploaded, you can update RealtimeSTT-mac to use the optimized versions by modifying the model mapping to point to your FP32 repositories.

This creates a seamless experience where users get Apple Silicon-optimized performance without any code changes.