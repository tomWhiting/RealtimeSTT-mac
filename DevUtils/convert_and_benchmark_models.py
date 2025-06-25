#!/usr/bin/env python3
"""
Model Conversion and Benchmarking Script for RealtimeSTT-mac

This script:
1. Downloads all models from faster_whisper_models.json
2. Converts FP16 models to FP32 for Apple Silicon optimization
3. Benchmarks both versions (load time, transcription time, accuracy)
4. Generates enhanced model registry with performance data
5. Creates READMEs for converted models
6. Uploads FP32 versions to HuggingFace

Usage:
    python convert_and_benchmark_models.py --hf-token YOUR_TOKEN --target-org realtimestt-mac
"""

import json
import os
import sys
import time
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import difflib

# Add parent directory to path to import RealtimeSTT
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import faster_whisper
    from huggingface_hub import HfApi, Repository, create_repo
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install faster-whisper huggingface-hub soundfile")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    load_time_ms: float
    transcribe_time_ms: float
    transcription_accuracy: float
    transcription_text: str
    model_size_mb: float

@dataclass
class ModelInfo:
    name: str
    precision: str
    repo: str
    size_mb: float
    readme_url: str
    benchmarks: BenchmarkResult

class ModelConverter:
    def __init__(self, hf_token: str, target_org: str, test_audio_path: str, expected_text: str):
        self.hf_token = hf_token
        self.target_org = target_org
        self.test_audio_path = test_audio_path
        self.expected_text = expected_text.strip()
        self.hf_api = HfApi(token=hf_token)
        self.work_dir = Path("./model_conversion_workspace")
        self.work_dir.mkdir(exist_ok=True)

        # Load original model mappings
        models_file = Path(__file__).parent.parent / "faster_whisper_models.json"
        with open(models_file, 'r') as f:
            self.original_models = json.load(f)

        self.enhanced_registry = {}

    def calculate_accuracy(self, predicted: str, expected: str) -> float:
        """Calculate transcription accuracy using word-level comparison"""
        predicted_words = predicted.lower().strip().split()
        expected_words = expected.lower().strip().split()

        if not expected_words:
            return 100.0 if not predicted_words else 0.0

        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, expected_words, predicted_words)
        accuracy = matcher.ratio() * 100
        return round(accuracy, 1)

    def get_model_size_mb(self, model_path: str) -> float:
        """Calculate total size of model directory in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
        return round(total_size / (1024 * 1024), 1)

    def benchmark_model(self, model_path: str, precision: str) -> BenchmarkResult:
        """Benchmark model performance"""
        logger.info(f"Benchmarking model: {model_path} ({precision})")

        # Measure load time
        start_time = time.time()
        try:
            model = faster_whisper.WhisperModel(
                model_size_or_path=model_path,
                device="cpu",
                compute_type="float32" if precision == "fp32" else "default"
            )
            load_time_ms = (time.time() - start_time) * 1000
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

        # Measure transcription time
        start_time = time.time()
        try:
            segments, info = model.transcribe(self.test_audio_path, language="en")
            transcription_text = " ".join(segment.text.strip() for segment in segments)
            transcribe_time_ms = (time.time() - start_time) * 1000
        except Exception as e:
            logger.error(f"Failed to transcribe with model {model_path}: {e}")
            raise

        # Calculate accuracy
        accuracy = self.calculate_accuracy(transcription_text, self.expected_text)

        # Get model size
        size_mb = self.get_model_size_mb(model_path)

        # Clean up model from memory
        del model

        return BenchmarkResult(
            load_time_ms=round(load_time_ms, 1),
            transcribe_time_ms=round(transcribe_time_ms, 1),
            transcription_accuracy=accuracy,
            transcription_text=transcription_text.strip(),
            model_size_mb=size_mb
        )

    def download_original_model(self, model_name: str, repo: str) -> str:
        """Download original model from HuggingFace"""
        logger.info(f"Downloading original model: {repo}")

        download_path = self.work_dir / f"{model_name}_original"
        download_path.mkdir(exist_ok=True)

        try:
            # Use faster-whisper's built-in download
            model = faster_whisper.WhisperModel(
                model_size_or_path=repo,
                device="cpu",
                download_root=str(download_path.parent)
            )

            # Find the downloaded model directory
            model_dirs = list(download_path.parent.glob(f"*{repo.split('/')[-1]}*"))
            if not model_dirs:
                # Fallback: look for any directory that might be the model
                model_dirs = [d for d in download_path.parent.iterdir() if d.is_dir() and d.name != download_path.name]

            if model_dirs:
                actual_path = str(model_dirs[0])
                logger.info(f"Model downloaded to: {actual_path}")
                return actual_path
            else:
                raise ValueError(f"Could not locate downloaded model for {repo}")

        except Exception as e:
            logger.error(f"Failed to download {repo}: {e}")
            raise

    def convert_to_fp32(self, original_path: str, model_name: str) -> str:
        """Convert model from FP16 to FP32"""
        logger.info(f"Converting {model_name} to FP32")

        fp32_path = self.work_dir / f"{model_name}_fp32"
        fp32_path.mkdir(exist_ok=True)

        # Use ct2-transformers-converter for conversion
        cmd = [
            "ct2-transformers-converter",
            "--model", original_path,
            "--output_dir", str(fp32_path),
            "--quantization", "float32",
            "--copy_files", "tokenizer.json", "preprocessor_config.json"
        ]

        try:
            logger.info(f"Running conversion command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Conversion completed successfully")
            return str(fp32_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("ct2-transformers-converter not found. Please install CTranslate2 tools.")
            raise

    def generate_readme(self, model_name: str, original_info: ModelInfo, fp32_info: ModelInfo) -> str:
        """Generate README for the converted model"""

        load_improvement = ((original_info.benchmarks.load_time_ms - fp32_info.benchmarks.load_time_ms) /
                          original_info.benchmarks.load_time_ms * 100)

        size_increase = ((fp32_info.benchmarks.model_size_mb - original_info.benchmarks.model_size_mb) /
                        original_info.benchmarks.model_size_mb * 100)

        readme_content = f"""# Faster-Whisper {model_name.title()} (FP32 - Apple Silicon Optimized)

This model has been converted from [{original_info.repo}](https://huggingface.co/{original_info.repo})
to FP32 precision for optimal performance on Apple Silicon devices.

## Why FP32 for Apple Silicon?

Apple Silicon (M1/M2/M3) with CTranslate2's Accelerate backend doesn't have optimized FP16 support,
so FP16 models are automatically converted to FP32 at load time. This pre-converted version
eliminates that conversion step, resulting in faster model loading.

## Performance Comparison

| Metric | FP16 (Original) | FP32 (This Model) | Improvement |
|--------|-----------------|-------------------|-------------|
| **Load Time** | {original_info.benchmarks.load_time_ms:.1f}ms | {fp32_info.benchmarks.load_time_ms:.1f}ms | **{load_improvement:+.1f}%** |
| **Transcription Time** | {original_info.benchmarks.transcribe_time_ms:.1f}ms | {fp32_info.benchmarks.transcribe_time_ms:.1f}ms | {((original_info.benchmarks.transcribe_time_ms - fp32_info.benchmarks.transcribe_time_ms) / original_info.benchmarks.transcribe_time_ms * 100):+.1f}% |
| **Model Size** | {original_info.benchmarks.model_size_mb:.1f}MB | {fp32_info.benchmarks.model_size_mb:.1f}MB | {size_increase:+.1f}% |
| **Transcription Accuracy** | {original_info.benchmarks.transcription_accuracy:.1f}% | {fp32_info.benchmarks.transcription_accuracy:.1f}% | {(fp32_info.benchmarks.transcription_accuracy - original_info.benchmarks.transcription_accuracy):+.1f}% |

## Usage

```python
from faster_whisper import WhisperModel

# Load the optimized model
model = WhisperModel("{self.target_org}/{model_name}-fp32", device="cpu")

# Transcribe audio
segments, info = model.transcribe("audio.wav")
for segment in segments:
    print(f"[{{segment.start:.2f}}s -> {{segment.end:.2f}}s] {{segment.text}}")
```

## Integration with RealtimeSTT-mac

This model is automatically used when you specify `model="{model_name}"` in RealtimeSTT-mac:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    model="{model_name}",  # Automatically uses this optimized version
    device="cpu"
)
```

## Technical Details

- **Original Model**: {original_info.repo}
- **Precision**: FP32 (32-bit floating point)
- **Backend**: CTranslate2 with Apple Accelerate
- **Target Platform**: Apple Silicon (M1/M2/M3)
- **Conversion Tool**: ct2-transformers-converter

## Benchmark Test Audio

The performance metrics above were measured using a standardized test audio clip with the following characteristics:
- **Expected Text**: "{self.expected_text}"
- **Test Environment**: Apple Silicon with CTranslate2 Accelerate backend

## License

Same as original model. See [{original_info.repo}](https://huggingface.co/{original_info.repo}) for details.
"""
        return readme_content

    def upload_model(self, model_path: str, model_name: str, readme_content: str) -> str:
        """Upload converted model to HuggingFace"""
        repo_name = f"{model_name}-fp32"
        repo_id = f"{self.target_org}/{repo_name}"

        logger.info(f"Uploading model to {repo_id}")

        try:
            # Create repository
            create_repo(
                repo_id=repo_id,
                token=self.hf_token,
                private=False,
                exist_ok=True
            )

            # Upload model files
            self.hf_api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                token=self.hf_token
            )

            # Upload README
            readme_path = Path(model_path) / "README.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)

            self.hf_api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                token=self.hf_token
            )

            logger.info(f"Successfully uploaded {repo_id}")
            return repo_id

        except Exception as e:
            logger.error(f"Failed to upload {repo_id}: {e}")
            raise

    def process_model(self, model_name: str, original_repo: str) -> Dict[str, ModelInfo]:
        """Process a single model: download, convert, benchmark, upload"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")

        model_results = {}

        try:
            # Download original model
            original_path = self.download_original_model(model_name, original_repo)

            # Benchmark original model
            original_benchmark = self.benchmark_model(original_path, "fp16")
            original_info = ModelInfo(
                name=model_name,
                precision="fp16",
                repo=original_repo,
                size_mb=original_benchmark.model_size_mb,
                readme_url=f"https://huggingface.co/{original_repo}/blob/main/README.md",
                benchmarks=original_benchmark
            )
            model_results["fp16"] = original_info

            # Convert to FP32
            fp32_path = self.convert_to_fp32(original_path, model_name)

            # Benchmark FP32 model
            fp32_benchmark = self.benchmark_model(fp32_path, "fp32")

            # Generate README
            readme_content = self.generate_readme(model_name, original_info,
                                                ModelInfo(model_name, "fp32", "",
                                                        fp32_benchmark.model_size_mb, "", fp32_benchmark))

            # Upload FP32 model
            fp32_repo = self.upload_model(fp32_path, model_name, readme_content)

            fp32_info = ModelInfo(
                name=model_name,
                precision="fp32",
                repo=fp32_repo,
                size_mb=fp32_benchmark.model_size_mb,
                readme_url=f"https://huggingface.co/{fp32_repo}/blob/main/README.md",
                benchmarks=fp32_benchmark
            )
            model_results["fp32"] = fp32_info

            logger.info(f"✅ Successfully processed {model_name}")

            # Log performance summary
            load_improvement = ((original_benchmark.load_time_ms - fp32_benchmark.load_time_ms) /
                              original_benchmark.load_time_ms * 100)
            logger.info(f"📊 Performance Summary for {model_name}:")
            logger.info(f"   Load Time: {original_benchmark.load_time_ms:.1f}ms → {fp32_benchmark.load_time_ms:.1f}ms ({load_improvement:+.1f}%)")
            logger.info(f"   Model Size: {original_benchmark.model_size_mb:.1f}MB → {fp32_benchmark.model_size_mb:.1f}MB")
            logger.info(f"   Accuracy: {original_benchmark.transcription_accuracy:.1f}% → {fp32_benchmark.transcription_accuracy:.1f}%")

        except Exception as e:
            logger.error(f"❌ Failed to process {model_name}: {e}")
            # Continue with other models

        finally:
            # Cleanup workspace for this model
            for path in [f"{model_name}_original", f"{model_name}_fp32"]:
                full_path = self.work_dir / path
                if full_path.exists():
                    shutil.rmtree(full_path, ignore_errors=True)

        return model_results

    def run(self) -> None:
        """Main conversion and benchmarking pipeline"""
        logger.info("🚀 Starting model conversion and benchmarking pipeline")
        logger.info(f"Target organization: {self.target_org}")
        logger.info(f"Test audio: {self.test_audio_path}")
        logger.info(f"Expected text: {self.expected_text}")

        if not os.path.exists(self.test_audio_path):
            logger.error(f"Test audio file not found: {self.test_audio_path}")
            sys.exit(1)

        # Process all models
        for model_name, original_repo in self.original_models.items():
            model_results = self.process_model(model_name, original_repo)
            if model_results:
                self.enhanced_registry[model_name] = {
                    precision: {
                        "repo": info.repo,
                        "size_mb": info.size_mb,
                        "readme_url": info.readme_url,
                        "benchmarks": {
                            "load_time_ms": info.benchmarks.load_time_ms,
                            "transcribe_time_ms": info.benchmarks.transcribe_time_ms,
                            "transcription_accuracy": info.benchmarks.transcription_accuracy,
                            "transcription_text": info.benchmarks.transcription_text
                        }
                    } for precision, info in model_results.items()
                }

        # Save enhanced registry
        output_file = Path(__file__).parent.parent / "enhanced_model_registry.json"
        with open(output_file, 'w') as f:
            json.dump(self.enhanced_registry, f, indent=2)

        logger.info(f"✅ Enhanced model registry saved to: {output_file}")
        logger.info("🎉 Model conversion and benchmarking pipeline completed!")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print performance summary for all models"""
        logger.info("\n" + "="*80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("="*80)

        for model_name, precisions in self.enhanced_registry.items():
            if "fp16" in precisions and "fp32" in precisions:
                fp16_data = precisions["fp16"]["benchmarks"]
                fp32_data = precisions["fp32"]["benchmarks"]

                load_improvement = ((fp16_data["load_time_ms"] - fp32_data["load_time_ms"]) /
                                  fp16_data["load_time_ms"] * 100)

                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"  Load Time Improvement: {load_improvement:+.1f}%")
                logger.info(f"  Size Increase: {((precisions['fp32']['size_mb'] - precisions['fp16']['size_mb']) / precisions['fp16']['size_mb'] * 100):+.1f}%")
                logger.info(f"  Accuracy Change: {(fp32_data['transcription_accuracy'] - fp16_data['transcription_accuracy']):+.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Convert and benchmark Whisper models for Apple Silicon")
    parser.add_argument("--hf-token", required=True, help="HuggingFace API token")
    parser.add_argument("--target-org", required=True, help="Target HuggingFace organization")
    parser.add_argument("--test-audio", required=True, help="Path to test audio file (.wav)")
    parser.add_argument("--expected-text", required=True, help="Expected transcription text")
    parser.add_argument("--models", help="Comma-separated list of models to process (default: all)")

    args = parser.parse_args()

    converter = ModelConverter(
        hf_token=args.hf_token,
        target_org=args.target_org,
        test_audio_path=args.test_audio,
        expected_text=args.expected_text
    )

    # Filter models if specified
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
        converter.original_models = {k: v for k, v in converter.original_models.items() if k in model_list}
        logger.info(f"Processing only specified models: {model_list}")

    converter.run()

if __name__ == "__main__":
    main()
