#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "faster-whisper>=1.1.0",
#   "huggingface-hub>=0.20.0",
#   "soundfile>=0.12.0",
#   "python-dotenv>=1.0.0",
#   "numpy>=1.24.0",
#   "torch>=2.0.0",
#   "transformers>=4.20.0",
#   "ctranslate2>=4.0.0"
# ]
# ///
"""
Model Conversion and Benchmarking Script for RealtimeSTT-mac

This script:
1. Downloads all models from model_selection.json (only those with include=true)
2. Converts FP16 models to FP32 for Apple Silicon optimization
3. Benchmarks both versions using benchmark_config.json
4. Generates model registry with performance data
5. Creates READMEs for converted models
6. Uploads FP32 versions to HuggingFace

Usage:
    uv run convert_and_benchmark_models.py

Environment variables (from .env):
    HF_TOKEN, HF_ORG, HF_COLLECTION_ID
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
from dotenv import load_dotenv

# Add parent directory to path to import RealtimeSTT
sys.path.insert(0, str(Path(__file__).parent.parent))

import faster_whisper
from huggingface_hub import HfApi, Repository, create_repo, add_collection_item, delete_repo, hf_hub_download, login, snapshot_download
import soundfile as sf
import numpy as np

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
    def __init__(self, hf_token: str, target_org: str, test_audio_path: str, expected_text: str,
                 collection_id: str = None, private_repos: bool = True):
        self.hf_token = hf_token
        self.target_org = target_org
        self.test_audio_path = test_audio_path
        self.expected_text = expected_text.strip()
        self.collection_id = collection_id
        self.private_repos = private_repos
        # Login to HuggingFace Hub using best practices
        login(token=hf_token)
        self.hf_api = HfApi(token=hf_token)
        self.work_dir = Path("./model_conversion_workspace")
        self.work_dir.mkdir(exist_ok=True)

        # Load model selection configuration
        models_file = Path(__file__).parent / "model_selection.json"
        with open(models_file, 'r') as f:
            model_config = json.load(f)

        # Filter models based on include flag
        self.original_models = {}
        self.conversion_files = {}
        for model_name, config in model_config["models"].items():
            if config.get("include", False):
                self.original_models[model_name] = config["source_repo"]
                self.conversion_files[model_name] = config.get("conversion_files", ["tokenizer.json"])
        
        # Load conversion settings
        self.conversion_settings = model_config.get("conversion_settings", {
            "base_command": "ct2-transformers-converter",
            "default_quantization": "float32",
            "fallback_quantization": "float16",
            "output_dir_pattern": "{model_name}-fp32",
            "force_conversion": True
        })
        
        # Load faster-whisper model mappings for naming and FP16 comparisons
        faster_whisper_file = Path(__file__).parent.parent / "faster_whisper_models.json"
        with open(faster_whisper_file, 'r') as f:
            self.faster_whisper_models = json.load(f)

        # Load benchmark configuration
        benchmark_config_file = Path(__file__).parent / "benchmark_config.json"
        with open(benchmark_config_file, 'r') as f:
            self.benchmark_config = json.load(f)

        # Get active benchmark sample
        active_sample = None
        for sample in self.benchmark_config["benchmark_samples"]:
            if sample.get("active", False):
                active_sample = sample
                break

        if not active_sample:
            raise ValueError("No active benchmark sample found in benchmark_config.json")

        # Override test audio and expected text from benchmark config
        self.test_audio_path = str(Path(__file__).parent / active_sample["file_path"])
        self.expected_text = active_sample["expected_transcript"].strip()
        self.benchmark_metadata = active_sample["metadata"]

        self.model_registry = {}
        self.benchmarks = {}

    def get_faster_whisper_repo(self, model_name: str) -> str:
        """Get the corresponding Systran faster-whisper repo for a model"""
        return self.faster_whisper_models.get(model_name, None)

    def get_upload_repo_name(self, model_name: str) -> str:
        """Get the repository name for our FP32 upload (mirrors faster-whisper naming)"""
        faster_repo = self.get_faster_whisper_repo(model_name)
        if faster_repo:
            # Extract just the model name part (e.g., "faster-whisper-tiny.en")
            return faster_repo.split('/')[-1]
        else:
            # Fallback for models not in faster_whisper_models.json
            if "distil" in model_name:
                return f"faster-distil-whisper-{model_name.replace('distil-', '')}"
            else:
                return f"faster-whisper-{model_name}"

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
        """Benchmark model performance with proper memory clearing"""
        logger.info(f"Benchmarking model: {model_path} ({precision})")

        # Clear any cached models and force garbage collection
        import gc
        gc.collect()

        # Force clear faster-whisper cache if possible
        try:
            if hasattr(faster_whisper, '_cached_models'):
                faster_whisper._cached_models.clear()
        except:
            pass

        # Measure load time (cold start)
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

        # Clean up model from memory IMMEDIATELY
        del model
        gc.collect()

        return BenchmarkResult(
            load_time_ms=round(load_time_ms, 1),
            transcribe_time_ms=round(transcribe_time_ms, 1),
            transcription_accuracy=accuracy,
            transcription_text=transcription_text.strip(),
            model_size_mb=size_mb
        )

    def download_original_model(self, model_name: str, repo: str) -> str:
        """Download original model from HuggingFace using Hub best practices"""
        logger.info(f"Downloading original model: {repo}")

        download_path = self.work_dir / f"{model_name}_original"

        try:
            # Use HuggingFace Hub's snapshot_download for better control
            actual_path = snapshot_download(
                repo_id=repo,
                cache_dir=str(self.work_dir),
                token=self.hf_token,
                local_files_only=False
            )

            logger.info(f"Model downloaded to: {actual_path}")
            return actual_path

        except Exception as e:
            logger.error(f"Failed to download {repo}: {e}")
            raise

    def download_faster_whisper_model(self, model_name: str) -> str:
        """Download the corresponding Systran faster-whisper model for FP16 benchmarking"""
        faster_repo = self.get_faster_whisper_repo(model_name)
        if not faster_repo:
            raise ValueError(f"No corresponding faster-whisper model found for {model_name}")
        
        logger.info(f"Downloading Systran faster-whisper model: {faster_repo}")

        download_path = self.work_dir / f"{model_name}_systran_fp16"

        try:
            # Use HuggingFace Hub's snapshot_download for better control
            actual_path = snapshot_download(
                repo_id=faster_repo,
                cache_dir=str(self.work_dir),
                token=self.hf_token,
                local_files_only=False
            )

            logger.info(f"Systran model downloaded to: {actual_path}")
            return actual_path

        except Exception as e:
            logger.error(f"Failed to download {faster_repo}: {e}")
            raise

    def convert_to_fp32(self, original_path: str, model_name: str) -> str:
        """Convert Transformers model to FP32 CTranslate2 format using ct2-transformers-converter"""
        logger.info(f"Converting {model_name} to FP32 using ct2-transformers-converter (replicating Systran process)")

        # Use the output pattern from conversion settings
        output_pattern = self.conversion_settings.get("output_dir_pattern", "{model_name}-fp32")
        fp32_path = self.work_dir / output_pattern.format(model_name=model_name)

        # Remove existing directory if it exists to avoid conflicts
        if fp32_path.exists():
            import shutil
            shutil.rmtree(fp32_path)

        fp32_path.mkdir(exist_ok=True)

        try:
            # Get conversion files for this model
            copy_files = self.conversion_files.get(model_name, ["tokenizer.json"])
            
            # Build the ct2-transformers-converter command exactly like Systran does
            cmd = [
                self.conversion_settings.get("base_command", "ct2-transformers-converter"),
                "--model", original_path,
                "--output_dir", str(fp32_path),
                "--quantization", self.conversion_settings.get("default_quantization", "float32")
            ]
            
            # Add copy_files parameter if files are specified
            if copy_files:
                cmd.extend(["--copy_files"] + copy_files)
            
            # Add force flag if specified
            if self.conversion_settings.get("force_conversion", True):
                cmd.append("--force")

            logger.info(f"Running conversion command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Conversion successful!")
            if result.stdout:
                logger.info(f"STDOUT: {result.stdout}")
            
            # Verify the conversion worked by checking for model files
            if not (fp32_path / "model.bin").exists():
                raise RuntimeError(f"Conversion failed - no model.bin found in {fp32_path}")
                
            logger.info(f"✅ Successfully converted {model_name} to FP32 CTranslate2 format")
            return str(fp32_path)

        except subprocess.CalledProcessError as e:
            logger.error(f"ct2-transformers-converter failed with return code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            
            # Try fallback with float16 if float32 fails
            if self.conversion_settings.get("default_quantization") != self.conversion_settings.get("fallback_quantization"):
                logger.info(f"Trying fallback quantization: {self.conversion_settings.get('fallback_quantization')}")
                try:
                    fallback_cmd = cmd.copy()
                    # Replace the quantization parameter
                    quantization_index = fallback_cmd.index("--quantization")
                    fallback_cmd[quantization_index + 1] = self.conversion_settings.get("fallback_quantization", "float16")
                    
                    result = subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
                    logger.info(f"Fallback conversion successful with {self.conversion_settings.get('fallback_quantization')}")
                    return str(fp32_path)
                    
                except subprocess.CalledProcessError as fallback_error:
                    logger.error(f"Fallback conversion also failed: {fallback_error.stderr}")
            
            raise e

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise

    def get_original_readme(self, repo: str) -> str:
        """Fetch original README from HuggingFace repository using Hub best practices"""
        try:
            readme_path = hf_hub_download(
                repo_id=repo,
                filename="README.md",
                token=self.hf_token
            )
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not fetch original README from {repo}: {e}")
            return f"# {repo.split('/')[-1]}\n\nOriginal model repository: [{repo}](https://huggingface.co/{repo})"

    def generate_readme(self, model_name: str, original_info: ModelInfo, fp32_info: ModelInfo) -> str:
        """Generate README with model card metadata and conversion notice"""

        # Get original README content
        original_readme = self.get_original_readme(original_info.repo)

        # Generate model card metadata YAML (use original OpenAI repo for metadata)
        source_repo = self.original_models.get(model_name, original_info.repo)
        yaml_metadata = self.generate_model_metadata(model_name, source_repo)

        # Create conversion notice and benchmark table
        conversion_notice = f"""# FP32 Optimized Model for Apple Silicon

This model has been converted to FP32 precision for optimized Apple Silicon performance, eliminating runtime FP16→FP32 conversion overhead.

## Performance Comparison

| Model Version | Load Time | Transcription Time | Accuracy (%) | Size (MB) |
|---------------|-----------|-------------------|--------------|-----------|
| FP16 (Systran faster-whisper) | {original_info.benchmarks.load_time_ms:.1f}ms | {original_info.benchmarks.transcribe_time_ms:.1f}ms | {original_info.benchmarks.transcription_accuracy:.1f}% | {original_info.benchmarks.model_size_mb:.1f}MB |
| FP32 (This Model) | {fp32_info.benchmarks.load_time_ms:.1f}ms | {fp32_info.benchmarks.transcribe_time_ms:.1f}ms | {fp32_info.benchmarks.transcription_accuracy:.1f}% | {fp32_info.benchmarks.model_size_mb:.1f}MB |

**Benchmark baseline:** [{original_info.repo}](https://huggingface.co/{original_info.repo})

---

"""

        # Combine metadata, conversion notice, and original README
        return yaml_metadata + "\n" + conversion_notice + original_readme

    def generate_simple_readme(self, model_name: str, original_repo: str, fp32_info: ModelInfo) -> str:
        """Generate a simple README when FP16 comparison is not available"""
        
        # Get original README content
        original_readme = self.get_original_readme(original_repo)

        # Generate model card metadata YAML
        yaml_metadata = self.generate_model_metadata(model_name, original_repo)

        # Create conversion notice without comparison
        conversion_notice = f"""# FP32 Converted Model

This model has been converted from the original model to FP32 precision for optimized Apple Silicon performance.
Original model: [{original_repo}](https://huggingface.co/{original_repo})

## Model Performance

| Model Version | Load Time | Transcription Time | Accuracy (%) | Size (MB) |
|---------------|-----------|-------------------|--------------|-----------|
| FP32 (This Model) | {fp32_info.benchmarks.load_time_ms:.1f}ms | {fp32_info.benchmarks.transcribe_time_ms:.1f}ms | {fp32_info.benchmarks.transcription_accuracy:.1f}% | {fp32_info.benchmarks.model_size_mb:.1f}MB |

---

"""

        # Combine metadata, conversion notice, and original README
        return yaml_metadata + "\n" + conversion_notice + original_readme

    def generate_model_metadata(self, model_name: str, original_repo: str) -> str:
        """Generate HuggingFace model card metadata YAML"""
        
        # Determine if this is an English-only model
        is_english_only = ".en" in model_name
        is_distil = "distil" in model_name
        
        # Set up languages
        if is_english_only:
            languages = ["en"]
        else:
            # Common Whisper languages
            languages = ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]
        
        # Set up tags
        tags = ["whisper", "speech-recognition", "ctranslate2", "apple-silicon", "float32"]
        if is_distil:
            tags.append("distillation")
        if is_english_only:
            tags.append("english")
        
        # Generate YAML metadata
        yaml_content = f"""---
library_name: ctranslate2
pipeline_tag: automatic-speech-recognition
license: mit
base_model: {original_repo}
tags:"""
        
        # Add tags
        for tag in tags:
            yaml_content += f"\n  - {tag}"
        
        yaml_content += f"""
language:"""
        
        # Add languages
        for lang in languages:
            yaml_content += f"\n  - {lang}"
        
        yaml_content += f"""
metrics:
  - accuracy
  - latency
model_size: {model_name}
precision: float32
framework: ctranslate2
inference: true
---"""
        
        return yaml_content

    def upload_model(self, model_path: str, model_name: str, readme_content: str) -> str:
        """Upload converted model to HuggingFace"""
        # Use the same naming pattern as original models
        repo_name = model_name  # e.g., "faster-whisper-tiny.en"
        repo_id = f"{self.target_org}/{repo_name}"

        logger.info(f"Uploading model to {repo_id}")

        try:
            # Check if repository already exists
            try:
                repo_info = self.hf_api.repo_info(repo_id=repo_id, token=self.hf_token)
                logger.info(f"Repository {repo_id} already exists - will update it")
                repo_exists = True
            except Exception:
                repo_exists = False

            # Create repository if it doesn't exist
            if not repo_exists:
                create_repo(
                    repo_id=repo_id,
                    token=self.hf_token,
                    private=self.private_repos,
                    exist_ok=True
                )
                logger.info(f"Created {'private' if self.private_repos else 'public'} repository: {repo_id}")

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

            # Add to collection if specified
            if self.collection_id and not repo_exists:
                try:
                    # Use the proper collection slug format
                    collection_slug = f"{self.target_org}/faster-whisper-float32-{self.collection_id}"
                    add_collection_item(
                        collection_slug=collection_slug,
                        item_id=repo_id,
                        item_type="model",
                        token=self.hf_token
                    )
                    logger.info(f"Added {repo_id} to collection")
                except Exception as e:
                    logger.warning(f"Failed to add {repo_id} to collection: {e}")

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
            # Download original model (in Transformers format)
            original_path = self.download_original_model(model_name, original_repo)

            # Convert to FP32 CTranslate2 format directly
            try:
                fp32_path = self.convert_to_fp32(original_path, model_name)
                logger.info(f"✅ Successfully converted {model_name} to FP32")
            except Exception as conversion_error:
                logger.error(f"❌ Conversion failed for {model_name}: {conversion_error}")
                raise conversion_error  # Re-raise to prevent upload

            # Clean up original downloaded model after conversion
            logger.info(f"🧹 Cleaning up original downloaded model for {model_name}")
            import shutil
            if Path(original_path).exists():
                shutil.rmtree(original_path, ignore_errors=True)
                logger.info(f"🗑️ Deleted original model files: {original_path}")

            # Download Systran faster-whisper model for FP16 benchmarking
            try:
                systran_path = self.download_faster_whisper_model(model_name)
                logger.info(f"✅ Successfully downloaded Systran model for {model_name}")
                
                # Benchmark Systran FP16 model
                systran_benchmark = self.benchmark_model(systran_path, "fp16")
                systran_repo = self.get_faster_whisper_repo(model_name)
                systran_info = ModelInfo(
                    name=model_name,
                    precision="fp16",
                    repo=systran_repo,
                    size_mb=systran_benchmark.model_size_mb,
                    readme_url=f"https://huggingface.co/{systran_repo}/blob/main/README.md",
                    benchmarks=systran_benchmark
                )
                model_results["fp16"] = systran_info
                
                # Clean up Systran model after benchmarking
                if Path(systran_path).exists():
                    shutil.rmtree(systran_path, ignore_errors=True)
                    logger.info(f"🗑️ Deleted Systran model files: {systran_path}")
                    
            except Exception as systran_error:
                logger.warning(f"⚠️ Systran model download/benchmark failed for {model_name}: {systran_error}")
                # If Systran download fails, we'll skip the comparison but continue with FP32

            # Benchmark FP32 model
            fp32_benchmark = self.benchmark_model(fp32_path, "fp32")

            # Generate README
            fp32_info_for_readme = ModelInfo(model_name, "fp32", "",
                                            fp32_benchmark.model_size_mb, "", fp32_benchmark)
            
            if "fp16" in model_results:
                readme_content = self.generate_readme(model_name, model_results["fp16"], fp32_info_for_readme)
            else:
                # Generate a simpler README if we don't have FP16 comparison
                readme_content = self.generate_simple_readme(model_name, original_repo, fp32_info_for_readme)

            # Upload FP32 model with faster-whisper naming pattern
            upload_repo_name = self.get_upload_repo_name(model_name)
            try:
                fp32_repo = self.upload_model(fp32_path, upload_repo_name, readme_content)
                logger.info(f"✅ Successfully uploaded {model_name} to {fp32_repo}")
                
                # Clean up model files immediately after successful upload
                logger.info(f"🧹 Cleaning up model files for {model_name}")
                import shutil
                if Path(fp32_path).exists():
                    shutil.rmtree(fp32_path, ignore_errors=True)
                    logger.info(f"🗑️ Deleted FP32 model files: {fp32_path}")
                
            except Exception as upload_error:
                logger.error(f"❌ Upload failed for {model_name}: {upload_error}")
                raise upload_error  # Re-raise to mark as failed

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

            # Log performance summary (if we have both FP16 and FP32 benchmarks)
            if "fp16" in model_results:
                original_benchmark = model_results["fp16"].benchmarks
                load_improvement = ((original_benchmark.load_time_ms - fp32_benchmark.load_time_ms) /
                                  original_benchmark.load_time_ms * 100)
                logger.info(f"📊 Performance Summary for {model_name}:")
                logger.info(f"   Load Time: {original_benchmark.load_time_ms:.1f}ms → {fp32_benchmark.load_time_ms:.1f}ms ({load_improvement:+.1f}%)")
                logger.info(f"   Model Size: {original_benchmark.model_size_mb:.1f}MB → {fp32_benchmark.model_size_mb:.1f}MB")
                logger.info(f"   Accuracy: {original_benchmark.transcription_accuracy:.1f}% → {fp32_benchmark.transcription_accuracy:.1f}%")
            else:
                logger.info(f"📊 FP32 Performance for {model_name}:")
                logger.info(f"   Load Time: {fp32_benchmark.load_time_ms:.1f}ms")
                logger.info(f"   Model Size: {fp32_benchmark.model_size_mb:.1f}MB")
                logger.info(f"   Accuracy: {fp32_benchmark.transcription_accuracy:.1f}%")

        except Exception as e:
            logger.error(f"❌ Failed to process {model_name}: {e}")
            # Don't create registry entries for failed models
            return {}

        finally:
            # Final cleanup - remove any remaining files (shouldn't be needed due to immediate cleanup)
            import shutil
            for path_pattern in [f"{model_name}-fp32", f"{model_name}-fp16"]:
                full_path = self.work_dir / path_pattern
                if full_path.exists():
                    logger.warning(f"🧹 Final cleanup: removing {full_path}")
                    shutil.rmtree(full_path, ignore_errors=True)

        return model_results

    def run(self) -> None:
        """Main conversion and benchmarking pipeline"""
        logger.info("🚀 Starting model conversion and benchmarking pipeline")
        logger.info(f"Target organization: {self.target_org}")
        logger.info(f"Collection ID: {self.collection_id}")
        logger.info(f"Private repositories: {self.private_repos}")
        logger.info(f"Test audio: {self.test_audio_path}")
        logger.info(f"Expected text: {self.expected_text}")

        if not os.path.exists(self.test_audio_path):
            logger.error(f"Test audio file not found: {self.test_audio_path}")
            sys.exit(1)

        # Process all models
        for model_name, original_repo in self.original_models.items():
            model_results = self.process_model(model_name, original_repo)
            if model_results:
                # Calculate additional performance metrics
                for precision, info in model_results.items():
                    audio_duration = self.benchmark_metadata["duration_seconds"]
                    word_count = self.benchmark_metadata["word_count"]

                    transcribe_time_per_audio_second = info.benchmarks.transcribe_time_ms / audio_duration
                    transcribe_time_per_word = info.benchmarks.transcribe_time_ms / word_count

                    # Try to get exact download size, fall back to MB estimate
                    try:
                        download_size_kb = info.benchmarks.model_size_mb * 1024
                    except:
                        download_size_kb = info.benchmarks.model_size_mb * 1024

                # Store in model registry
                self.model_registry[model_name] = {}
                for precision, info in model_results.items():
                    audio_duration = self.benchmark_metadata["duration_seconds"]
                    word_count = self.benchmark_metadata["word_count"]

                    transcribe_time_per_audio_second = info.benchmarks.transcribe_time_ms / audio_duration
                    transcribe_time_per_word = info.benchmarks.transcribe_time_ms / word_count
                    download_size_kb = info.benchmarks.model_size_mb * 1024

                    self.model_registry[model_name][precision] = {
                        "repo": info.repo,
                        "repo_url": f"https://huggingface.co/{info.repo}",
                        "readme_url": f"https://huggingface.co/{info.repo}/blob/main/README.md",
                        "size_mb": info.benchmarks.model_size_mb,
                        "download_size_kb": round(download_size_kb, 1),
                        "performance_data": {
                            "model_load_time_ms": info.benchmarks.load_time_ms,
                            "transcription_time_per_audio_second_ms": round(transcribe_time_per_audio_second, 1),
                            "transcription_time_per_word_ms": round(transcribe_time_per_word, 1),
                            "transcription_accuracy_percent": info.benchmarks.transcription_accuracy
                        }
                    }

                # Store benchmark data separately
                self.benchmarks[model_name] = {
                    precision: {
                        "load_time_ms": info.benchmarks.load_time_ms,
                        "transcription_time_ms": info.benchmarks.transcribe_time_ms,
                        "transcription_accuracy": info.benchmarks.transcription_accuracy,
                        "transcription_text": info.benchmarks.transcription_text
                    } for precision, info in model_results.items()
                }

                # Save files after each model (progressive updates)
                output_file = Path(__file__).parent.parent / "model_registry.json"
                with open(output_file, 'w') as f:
                    json.dump(self.model_registry, f, indent=2)

                benchmark_file = Path(__file__).parent / "benchmarks.json"
                with open(benchmark_file, 'w') as f:
                    json.dump(self.benchmarks, f, indent=2)

                logger.info(f"📝 Updated model registry and benchmarks after {model_name}")

        logger.info("🎉 Model conversion and benchmarking pipeline completed!")
        logger.info(f"📊 Processed {len(self.model_registry)} models")
        logger.info(f"📄 Final registry: {Path(__file__).parent.parent / 'model_registry.json'}")
        logger.info(f"📄 Final benchmarks: {Path(__file__).parent / 'benchmarks.json'}")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print performance summary for all models"""
        logger.info("\n" + "="*80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("="*80)

        for model_name, precisions in self.model_registry.items():
            if "fp16" in precisions and "fp32" in precisions:
                fp16_data = precisions["fp16"]["performance_data"]
                fp32_data = precisions["fp32"]["performance_data"]

                load_improvement = ((fp16_data["model_load_time_ms"] - fp32_data["model_load_time_ms"]) /
                                  fp16_data["model_load_time_ms"] * 100)

                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"  Load Time Improvement: {load_improvement:+.1f}%")
                logger.info(f"  Size Increase: {((precisions['fp32']['size_mb'] - precisions['fp16']['size_mb']) / precisions['fp16']['size_mb'] * 100):+.1f}%")
                logger.info(f"  Accuracy Change: {(fp32_data['transcription_accuracy_percent'] - fp16_data['transcription_accuracy_percent']):+.1f}%")

def main():
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(description="Convert and benchmark Whisper models for Apple Silicon")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HuggingFace API token")
    parser.add_argument("--target-org", default=os.getenv("HF_ORG", "RocketFish"), help="Target HuggingFace organization")
    parser.add_argument("--collection-id", default=os.getenv("HF_COLLECTION_ID"), help="HuggingFace collection ID")
    parser.add_argument("--test-audio", default=os.getenv("TEST_AUDIO_PATH"), help="Path to test audio file (.wav)")
    parser.add_argument("--expected-text", default=os.getenv("EXPECTED_TEXT"), help="Expected transcription text")
    parser.add_argument("--models", help="Comma-separated list of models to process (default: all)")
    parser.add_argument("--private", action="store_true", default=os.getenv("PRIVATE_REPOS", "true").lower() == "true",
                        help="Create private repositories")

    args = parser.parse_args()

    # Validate required arguments
    if not args.hf_token:
        print("❌ HuggingFace token required. Set HF_TOKEN environment variable or use --hf-token")
        print("💡 Get your token from: https://huggingface.co/settings/tokens")
        print("💡 Make sure it has 'write' permissions for creating repositories")
        sys.exit(1)

    # Validate token format
    if not args.hf_token.startswith('hf_'):
        print("⚠️  Warning: HuggingFace tokens typically start with 'hf_'")
        print("💡 Make sure you're using a valid token from https://huggingface.co/settings/tokens")
    if not args.test_audio:
        print("❌ Test audio path required. Set TEST_AUDIO_PATH environment variable or use --test-audio")
        sys.exit(1)
    if not args.expected_text:
        print("❌ Expected text required. Set EXPECTED_TEXT environment variable or use --expected-text")
        sys.exit(1)

    converter = ModelConverter(
        hf_token=args.hf_token,
        target_org=args.target_org,
        test_audio_path=args.test_audio,
        expected_text=args.expected_text,
        collection_id=args.collection_id,
        private_repos=args.private
    )

    # Filter models if specified
    if args.models:
        model_list = [m.strip() for m in args.models.split(",")]
        converter.original_models = {k: v for k, v in converter.original_models.items() if k in model_list}
        logger.info(f"Processing only specified models: {model_list}")

    converter.run()

if __name__ == "__main__":
    main()
