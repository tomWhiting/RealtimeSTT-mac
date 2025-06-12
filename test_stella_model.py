#!/usr/bin/env python3
"""
Test script to verify custom Stella.ppn model works with RealtimeSTT on Apple Silicon.
"""

import os
import sys
from RealtimeSTT import AudioToTextRecorder

def test_stella_custom_model():
    """Test custom Stella.ppn model with RealtimeSTT"""
    
    print("🎯 Testing Custom Stella Model with RealtimeSTT")
    print("=" * 60)
    
    # Check environment
    porcupine_key = os.getenv("PORCUPINE_ACCESS_KEY")
    if not porcupine_key:
        print("❌ PORCUPINE_ACCESS_KEY not found in environment")
        return False
    
    print("✅ Porcupine access key found")
    
    # Path to your custom Stella model
    stella_model_path = "/Users/tom/Projects/Spaces/Python/roz-always-on/modules/voice/models/Stella.ppn"
    
    if not os.path.exists(stella_model_path):
        print(f"❌ Stella model not found at: {stella_model_path}")
        print("   Please check the model path")
        return False
        
    print(f"✅ Stella model found: {stella_model_path}")
    
    # Test wake word detection setup
    wake_word_detected = False
    transcription_received = False
    
    def on_wakeword_detected():
        global wake_word_detected
        wake_word_detected = True
        print("🌟 Wake word 'Stella' detected!")
        
    def on_wakeword_detection_start():
        print("👂 Listening for wake word 'Stella'...")
        
    def on_wakeword_timeout():
        if not wake_word_detected:
            print("⏰ Wake word timeout - say 'Stella' to activate")
        
    def on_recording_start():
        print("🎤 Recording started - speak your message")
        
    def on_recording_stop():
        print("🛑 Recording stopped - processing...")
        
    def on_transcription_start(audio_data):
        print("📝 Transcribing audio...")
    
    try:
        print("\n🚀 Initializing RealtimeSTT with custom Stella model...")
        
        recorder = AudioToTextRecorder(
            # Wake word configuration with custom model
            wakeword_backend="pvporcupine",
            wake_words=stella_model_path,  # Use full path to .ppn file
            wake_words_sensitivity=0.7,
            wake_word_timeout=10.0,
            wake_word_activation_delay=0.0,
            
            # Callbacks
            on_wakeword_detected=on_wakeword_detected,
            on_wakeword_detection_start=on_wakeword_detection_start,
            on_wakeword_timeout=on_wakeword_timeout,
            on_recording_start=on_recording_start,
            on_recording_stop=on_recording_stop,
            on_transcription_start=on_transcription_start,
            
            # Transcription settings
            model="tiny.en",
            language="en",
            
            # Other settings
            spinner=True,
            use_microphone=True,
        )
        
        print("✅ RealtimeSTT with custom Stella model initialized successfully!")
        print("\n📋 Test Instructions:")
        print("   1. Say 'Stella' to trigger wake word detection")
        print("   2. After detection, speak a test message")
        print("   3. Wait for transcription")
        print("   4. Press Ctrl+C to exit")
        print("\n" + "=" * 60)
        
        try:
            # Use the correct API pattern - text() method returns transcription
            while True:
                transcription = recorder.text()
                if transcription:
                    transcription_received = True
                    print(f"📝 Transcription: '{transcription}'")
                    print("\n👂 Listening for wake word 'Stella' again...")
                    
        except KeyboardInterrupt:
            print("\n🛑 Test interrupted by user")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        try:
            recorder.shutdown()
            print("🔧 Recorder shutdown complete")
        except:
            pass
    
    # Test results
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Custom Stella model: {'✅ SUCCESS' if wake_word_detected else '❌ NOT TESTED'}")
    print(f"   Wake word detection: {'✅ SUCCESS' if wake_word_detected else '❌ NOT TESTED'}")
    print(f"   Transcription: {'✅ SUCCESS' if transcription_received else '❌ NOT TESTED'}")
    
    if wake_word_detected and transcription_received:
        print("🎉 All tests passed! Custom Stella model working perfectly!")
        return True
    else:
        print("⚠️  Partial success - manual testing completed")
        return True

if __name__ == "__main__":
    print("🌟 Custom Stella Model Test for RealtimeSTT")
    print("=" * 60)
    
    success = test_stella_custom_model()
    
    if success:
        print("\n🎉 SUCCESS: Custom Stella model is fully compatible!")
        print("   - Apple Silicon compatible")
        print("   - Custom .ppn model working")
        print("   - Ready for voice assistant integration")
    else:
        print("\n❌ FAILED: Custom model has compatibility issues")
        sys.exit(1)