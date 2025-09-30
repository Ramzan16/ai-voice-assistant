import asyncio
from typing import AsyncIterator

from loguru import logger
from RealtimeSTT import AudioToTextRecorder

from src.config import Settings
from .base import SpeechToText
# from src.core.exceptions import STTError


class RealtimeSTTService(SpeechToText):
    """
    Handles capturing and transcribing audio from a microphone using the
    RealtimeSTT library, designed to work within an asyncio environment.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the RealtimeSTT recorder with configuration from settings.

        Args:
            settings: The application settings object.
        """
        self.settings = settings
        logger.info("Initializing RealtimeSTT...")

        try:
            self.recorder = AudioToTextRecorder(
                model=self.settings.stt_model.value,
                language="en",
                device=self.settings.microphone_index,
                spinner=False,
                wake_words=",".join(self.settings.wake_words),
                wakeword_backend="openwakeword",
                level=self.settings.log_level.value,
            )
            logger.success("RealtimeSTT initialized successfully.")
        except Exception as e:
            logger.opt(exception=e).critical("Failed to initialize RealtimeSTT recorder.")
            # raise STTError("Could not initialize the RealtimeSTT recorder.") from e
            raise

    async def transcribe(self) -> AsyncIterator[str]:
        """
        Listens for the wake word, records speech, and transcribes it.

        This method runs the blocking I/O operation in a separate thread to
        avoid blocking the main asyncio event loop. It yields the final
        transcribed text once available, as per the interface contract.
        """
        try:
            logger.info(f"Listening for wake words: {self.settings.wake_words}...")

            # The .text() method is blocking, so we run it in a separate thread.
            text = await asyncio.to_thread(self.recorder.text)

            if text:
                text = text.strip()
                logger.info(f"Transcription complete: '{text}'")
                yield text
            else:
                logger.warning("Transcription resulted in empty text (likely silence).")
                # Intentionally yield nothing if no speech was detected.

        except Exception as e:
            logger.opt(exception=e).error("An error occurred during transcription.")
            # In case of an error, the generator simply stops.