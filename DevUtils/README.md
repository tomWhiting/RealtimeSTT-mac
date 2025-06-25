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
pip install faster-whisper huggingface-hub soundfile python-dotenv
```

### Required Tools
- `ct2-transformers-converter` (from CTranslate2)
- HuggingFace account with API token
- Test audio file (.wav format)

## Usage

### Quick Start

1. **Set up environment**: Copy `.env` file in project root and add your HuggingFace token
2. **Prepare test audio**: Create test audio using the helper script or record your own
3. **Run the script**:

```bash
# Create test audio (optional - you can use your own)
python create_test_audio.py --preset realtime --output test_audio.wav

# Convert and benchmark all models
python convert_and_benchmark_models.py
```

### Environment Setup

**Create/edit `.env` file in project root:**
```bash
HF_TOKEN=your_huggingface_token_here
HF_ORG=RocketFish
HF_COLLECTION_ID=685c24dfdef22726400ff961
TEST_AUDIO_PATH=DevUtils/test_audio.wav
EXPECTED_TEXT=This is a test of real-time speech to text transcription using Apple Silicon optimization
```

### Advanced Usage

**Process specific models only:**
```bash
python convert_and_benchmark_models.py --models "tiny,base,small"
```

**Override environment settings:**
```bash
python convert_and_benchmark_models.py \
  --hf-token YOUR_TOKEN \
  --target-org your-org \
  --test-audio custom_test.wav \
  --expected-text "custom test transcription" \
  --models "tiny"
```

**Create public repositories:**
```bash
python convert_and_benchmark_models.py --no-private
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
Models are uploaded to `{target-org}/{original-model-name}` with:
- Converted model files (FP32)
- Generated README with benchmarks
- Proper model metadata
- **Private by default** (can be made public later)
- **Auto-added to collection** for easy management

### Collection Management
- Models are automatically added to your HuggingFace collection
- Collection ID is configurable via environment variables
- Existing repositories are updated (not duplicated)
- Failed uploads don't break the entire pipeline

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

### Environment Variables (.env)
The script uses environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | Required |
| `HF_ORG` | Target organization | RocketFish |
| `HF_COLLECTION_ID` | Collection to add models to | Optional |
| `TEST_AUDIO_PATH` | Path to test audio | Required |
| `EXPECTED_TEXT` | Expected transcription | Required |
| `PRIVATE_REPOS` | Create private repos | true |

### Test Audio Requirements
- **Format**: WAV file (other formats auto-converted)
- **Duration**: 5-15 seconds recommended
- **Content**: Clear speech, representative of typical use cases
- **Quality**: Good signal-to-noise ratio

### Model Selection
You can process:
- **All models** (default): Processes every model in `faster_whisper_models.json`
- **Specific models**: Use `--models` flag with comma-separated list
- **Model subsets**: Useful for testing or gradual deployment

### Repository Management
- **Private by default**: All repositories created as private initially
- **Collection integration**: Automatically added to specified collection
- **Naming consistency**: Uses same names as original models
- **Update existing**: Overwrites existing repositories safely

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
├── create_test_audio.py                # Test audio generator
├── config.example.json                 # Example configuration
└── model_conversion_workspace/         # Temporary workspace (auto-created)

../.env                                 # Environment configuration (project root)
```

## Integration with RealtimeSTT-mac

Once models are converted and uploaded, you can update RealtimeSTT-mac to use the optimized versions by modifying the model mapping to point to your FP32 repositories.

This creates a seamless experience where users get Apple Silicon-optimized performance without any code changes.