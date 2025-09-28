from pathlib import Path
from dotenv import load_dotenv
import asyncio
import os
import sys

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# --- FIX for ModuleNotFoundError ---
# Add the 'ai-voice-assistant' directory to Python's path
# This allows the script to find the 'src' module
project_root = Path(__file__).parent
source_directory = project_root / 'ai-voice-assistant'
sys.path.insert(0, str(source_directory))
# ------------------------------------

# Ensure the logs directory exists
(project_root / "logs").mkdir(exist_ok=True)


async def main():
    """
    Main asynchronous function to test the KokoroTTS service.
    """
    print("--- Kokoro TTS Test Script ---")
    print("This script will initialize the KokoroTTS service and synthesize a test sentence.")
    print("You should hear audio output from your default speakers.")
    print("NOTE: The first run may be slow as the model needs to be downloaded (~300MB).")

    try:
        # Import necessary components now that the path is corrected
        from assistant.src.config.logging_config import setup_logging
        from assistant.src.config.settings import settings
        from assistant.src.services.tts import KokoroTTS

        # 1. Set up logging
        setup_logging()
        print("\n[Step 1/4] Logging configured.")

        # 2. Initialize settings
        # The `settings` object is a singleton imported from config
        print(f"[Step 2/4] Settings initialized. Using voice: '{settings.tts_voice.value}' on device: '{settings.tts_device}'.")

        # 3. Initialize the KokoroTTS service
        print("[Step 3/4] Initializing KokoroTTS service... (This might take a moment)")
        try:
            tts_service = KokoroTTS(settings)
            print("   -> KokoroTTS service initialized successfully.")
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize KokoroTTS service: {e}")
            print("Please ensure you have run 'pip install kokoro>=0.9.4 sounddevice' and have a working internet connection.")
            return

        # 4. Synthesize a test sentence
        test_text = "Hello, this is a test of the Kokoro text-to-speech engine. I hope you can hear me clearly."
        print(f"\n[Step 4/4] Synthesizing and playing test sentence: '{test_text}'")
        try:
            await tts_service.synthesize(test_text)
            print("   -> Synthesis and playback complete.")
        except Exception as e:
            print(f"\n[ERROR] An error occurred during synthesis: {e}")

        print("\n--- Test Finished ---")

    except ImportError as e:
        print(f"\n[ERROR] Failed to import a required module: {e}")
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Ensure you have espeak-ng installed on your system
    # Linux: sudo apt-get update && sudo apt-get install espeak-ng
    # Mac:   brew install espeak-ng
    # Windows: Download from https://github.com/espeak-ng/espeak-ng/releases
    
    # Check for espeak-ng
    if os.name != 'nt' and os.system("command -v espeak-ng > /dev/null 2>&1") != 0:
         print("\n[WARNING] espeak-ng not found in PATH.")
         print("KokoroTTS uses espeak-ng for phonemizing words not in its dictionary.")
         print("The model may fail on certain words without it. Please install it for best results.")

    asyncio.run(main())

