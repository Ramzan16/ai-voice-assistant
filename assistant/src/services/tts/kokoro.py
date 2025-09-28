import asyncio
import sounddevice as sd
import numpy as np
from loguru import logger


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.config.settings import Settings
from .base import TextToSpeech

# Conditional import for type checking
try:
    from kokoro import KPipeline, KModel
except ImportError:
    logger.error("Kokoro TTS library not found. Please run 'pip install kokoro'.")
    KPipeline = None
    KModel = None


class KokoroTTS(TextToSpeech):
    """
    A Text-to-Speech implementation using the lightweight Kokoro TTS model.

    This class handles the initialization of the Kokoro pipeline and model,
    and provides an asynchronous method to synthesize and play audio from text.
    Audio playback is handled in a separate thread to prevent blocking the
    main asyncio event loop.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the Kokoro TTS service.

        Args:
            settings: The application settings object containing configuration
                      for TTS voice, device, and audio sample rate.
        """
        if KPipeline is None or KModel is None:
            raise ImportError("Kokoro TTS dependencies are not installed.")

        self.settings = settings
        self.voice = self.settings.tts_voice.value
        self.sample_rate = self.settings.sample_rate

        logger.info(f"Initializing KokoroTTS on device: {self.settings.tts_device}")
        try:
            # Initialize the model once and share it if needed
            self.model = KModel(repo_id='hexgrad/Kokoro-82M').to(self.settings.tts_device).eval()
            # Initialize the pipeline for American English ('a') as all voices are 'a*'
            self.pipeline = KPipeline(lang_code='a', model=self.model)
            logger.success("KokoroTTS initialized successfully.")
        except Exception as e:
            logger.opt(exception=e).critical("Failed to initialize KokoroTTS model or pipeline.")
            raise

    def _play_audio(self, text: str) -> None:
        """
        Synchronous method to generate and play audio.
        This is designed to be run in a separate thread.
        """
        try:
            logger.info(f"Synthesizing speech for: '{text[:50]}...'")
            generator = self.pipeline(text, voice=self.voice)

            audio_chunks = []
            for result in generator:
                if result.audio is not None:
                    audio_chunks.append(result.audio.numpy())

            if not audio_chunks:
                logger.warning("No audio was generated for the provided text.")
                return

            # Concatenate all audio chunks into a single audio stream
            full_audio = np.concatenate(audio_chunks)

            logger.info("Playing synthesized audio...")
            sd.play(full_audio, self.sample_rate)
            sd.wait()  # Block execution in this thread until audio finishes
            logger.info("Finished playing audio.")

        except Exception as e:
            logger.opt(exception=e).error("An error occurred during audio synthesis or playback.")

    async def synthesize(self, text: str) -> None:
        """
        Asynchronously synthesizes text into speech and plays it.

        The actual synthesis and blocking playback are run in a separate
        thread to avoid blocking the main event loop.

        Args:
            text: The text to be converted to speech.
        """
        if not text.strip():
            logger.warning("Synthesize called with empty text. Skipping.")
            return

        # Run the synchronous, blocking _play_audio method in a separate thread
        await asyncio.to_thread(self._play_audio, text)
