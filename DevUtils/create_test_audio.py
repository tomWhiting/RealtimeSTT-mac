#!/usr/bin/env python3
"""
Test Audio Generator for Model Benchmarking

This script creates a standardized test audio file for benchmarking
Whisper model performance. It generates a WAV file with clear speech
that can be used to measure transcription accuracy and timing.

Usage:
    python create_test_audio.py --output test_audio.wav --text "Your test text here"
    python create_test_audio.py --preset quick  # Uses predefined text
"""

import argparse
import sys
from pathlib import Path
import tempfile
import subprocess

# Test text presets for different scenarios
PRESETS = {
    "quick": {
        "text": "The quick brown fox jumps over the lazy dog",
        "description": "Standard pangram - good for basic testing"
    },
    "numbers": {
        "text": "The year twenty twenty four has twelve months and three hundred sixty five days",
        "description": "Contains numbers and dates - tests number transcription"
    },
    "technical": {
        "text": "Machine learning models like Whisper use transformer architectures for speech recognition",
        "description": "Technical vocabulary - tests domain-specific transcription"
    },
    "punctuation": {
        "text": "Hello, world! How are you today? I'm fine, thanks. What about you?",
        "description": "Mixed punctuation - tests sentence boundary detection"
    },
    "realtime": {
        "text": "This is a test of real-time speech to text transcription using Apple Silicon optimization",
        "description": "Project-specific vocabulary - relevant to RealtimeSTT use cases"
    }
}

def check_dependencies():
    """Check if required text-to-speech tools are available"""
    tools = []

    # macOS built-in say command
    try:
        subprocess.run(["say", "--version"], capture_output=True, check=True)
        tools.append("say")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # espeak (cross-platform)
    try:
        subprocess.run(["espeak", "--version"], capture_output=True, check=True)
        tools.append("espeak")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return tools

def generate_audio_say(text: str, output_path: str, voice: str = "Alex", rate: int = 180):
    """Generate audio using macOS 'say' command"""
    cmd = [
        "say",
        "-v", voice,
        "-r", str(rate),
        "-o", output_path,
        "--data-format=LEI16@16000",  # 16-bit linear PCM, 16kHz
        text
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, "Generated using macOS say command"
    except subprocess.CalledProcessError as e:
        return False, f"say command failed: {e.stderr}"

def generate_audio_espeak(text: str, output_path: str, speed: int = 160):
    """Generate audio using espeak"""
    cmd = [
        "espeak",
        "-s", str(speed),
        "-w", output_path,
        text
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, "Generated using espeak"
    except subprocess.CalledProcessError as e:
        return False, f"espeak command failed: {e.stderr}"

def convert_to_whisper_format(input_path: str, output_path: str):
    """Convert audio to Whisper-compatible format using ffmpeg"""
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # Mono
        "-c:a", "pcm_s16le",  # 16-bit PCM
        "-y",            # Overwrite output
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, "Converted to Whisper format (16kHz mono WAV)"
    except subprocess.CalledProcessError as e:
        return False, f"ffmpeg conversion failed: {e.stderr}"

def create_test_audio(text: str, output_path: str):
    """Create test audio file with the given text"""
    print(f"Creating test audio: '{text}'")
    print(f"Output: {output_path}")

    # Check available TTS tools
    available_tools = check_dependencies()
    if not available_tools:
        print("❌ No text-to-speech tools found!")
        print("\nPlease install one of the following:")
        print("- macOS: Built-in 'say' command (should be available)")
        print("- Linux/Windows: espeak (apt install espeak / brew install espeak)")
        return False

    print(f"Available TTS tools: {', '.join(available_tools)}")

    # Use temporary file if we need format conversion
    needs_conversion = not output_path.endswith('.wav')
    if needs_conversion or 'say' in available_tools:
        # macOS say can output directly to the right format
        temp_file = None
        final_output = output_path
    else:
        # espeak outputs to WAV but might need format adjustment
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_file.close()
        final_output = temp_file.name

    # Generate audio
    success = False
    message = ""

    if "say" in available_tools:
        success, message = generate_audio_say(text, final_output)
    elif "espeak" in available_tools:
        success, message = generate_audio_espeak(text, final_output)

    if not success:
        print(f"❌ Audio generation failed: {message}")
        return False

    print(f"✅ {message}")

    # Convert format if needed
    if temp_file and success:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            success, conv_message = convert_to_whisper_format(final_output, output_path)
            if success:
                print(f"✅ {conv_message}")
            else:
                print(f"⚠️  Format conversion failed: {conv_message}")
                print("Using original format - may still work for testing")
                # Copy temp file to final destination
                import shutil
                shutil.copy2(final_output, output_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  ffmpeg not found - using original format")
            import shutil
            shutil.copy2(final_output, output_path)

    # Cleanup temp file
    if temp_file:
        Path(temp_file.name).unlink(missing_ok=True)

    # Verify output file exists
    if Path(output_path).exists():
        file_size = Path(output_path).stat().st_size
        print(f"✅ Test audio created: {output_path} ({file_size:,} bytes)")
        print(f"\n📝 Expected transcription: \"{text}\"")
        print(f"\n🎯 Use this exact text when running the benchmark script:")
        print(f"   --expected-text \"{text}\"")
        return True
    else:
        print("❌ Output file was not created")
        return False

def list_presets():
    """List available text presets"""
    print("Available presets:")
    print("-" * 50)
    for preset_name, preset_data in PRESETS.items():
        print(f"{preset_name:12} | {preset_data['text']}")
        print(f"{'':12} | {preset_data['description']}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Generate test audio for Whisper model benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use a preset
    python create_test_audio.py --preset quick --output test_audio.wav

    # Custom text
    python create_test_audio.py --text "Your custom text here" --output test.wav

    # List available presets
    python create_test_audio.py --list-presets
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Custom text to synthesize")
    group.add_argument("--preset", choices=PRESETS.keys(), help="Use predefined test text")
    group.add_argument("--list-presets", action="store_true", help="List available presets")

    parser.add_argument("--output", "-o", default="test_audio.wav", help="Output audio file path")
    parser.add_argument("--voice", default="Alex", help="Voice to use (macOS only)")
    parser.add_argument("--speed", type=int, default=180, help="Speech rate (words per minute)")

    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        return 0

    # Determine text to use
    if args.preset:
        text = PRESETS[args.preset]["text"]
        print(f"Using preset '{args.preset}': {PRESETS[args.preset]['description']}")
    else:
        text = args.text

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate audio
    success = create_test_audio(text, str(output_path))

    if success:
        print(f"\n🎉 Ready to benchmark! Run:")
        print(f"python convert_and_benchmark_models.py \\")
        print(f"  --test-audio {args.output} \\")
        print(f"  --expected-text \"{text}\" \\")
        print(f"  --hf-token YOUR_TOKEN \\")
        print(f"  --target-org YOUR_ORG")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
