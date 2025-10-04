import asyncio
from typing import AsyncIterator
import numpy as np
from loguru import logger

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ...config.settings import Settings
from .base import TextToSpeech
from ...core.exceptions import TTSError

try:
    from kokoro import KPipeline, KModel
except ImportError:
    logger.error("Kokoro TTS library not found. Please run 'pip install kokoro'.")
    KPipeline = None
    KModel = None


class KokoroTTS(TextToSpeech):
    """
    A streaming Text-to-Speech implementation using the Kokoro TTS model.
    """
    def __init__(self, settings: Settings):
        if KPipeline is None or KModel is None:
            raise ImportError("Kokoro TTS dependencies are not installed.")
        self.settings = settings
        self.voice = self.settings.tts_voice.value
        logger.info(f"Initializing KokoroTTS on device: {self.settings.tts_device}")
        try:
            self.model = KModel(repo_id='hexgrad/Kokoro-82M').to(self.settings.tts_device).eval()
            self.pipeline = KPipeline(lang_code='a', model=self.model)
            logger.success("KokoroTTS initialized successfully.")
        except Exception as e:
            raise TTSError("Could not initialize Kokoro TTS") from e

    def _generate_chunks_and_queue(self, text: str, queue: asyncio.Queue):
        """[SYNC] Runs in a thread to generate audio and queue it."""
        try:
            generator = self.pipeline(text, voice=self.voice)
            for result in generator:
                if result.audio is not None:
                    queue.put_nowait(result.audio.numpy())
        except Exception as e:
            logger.opt(exception=e).error("Error during TTS generation.")
        finally:
            queue.put_nowait(None) # Sentinel to signal the end

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """[ASYNC] Implements the abstract method by yielding audio chunks."""
        if not text.strip():
            return

        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        loop.run_in_executor(None, self._generate_chunks_and_queue, text, queue)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk