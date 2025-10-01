import asyncio
from typing import List

from loguru import logger

from assistant.src.config import Settings
from assistant.src.core.exceptions import VoiceAssistantError
from assistant.src.core.interfaces import Agent, SpeechToText, TextToSpeech
from assistant.src.core.types import ChatMessage, Role
from assistant.src.services.agent import GeminiAgent
from assistant.src.services.stt import RealtimeSTTService
from assistant.src.services.tools import ToolRegistry
from assistant.src.services.tts import KokoroTTS


class VoiceAssistant:
    """
    The main orchestrator for the voice assistant.

    This class initializes all the necessary services (STT, Agent, TTS) and
    manages the main application loop, coordinating the flow of information
    from spoken input to spoken output.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the voice assistant and its components.

        Args:
            settings: The application settings object.
        """
        self.settings = settings
        self.history: List[ChatMessage] = []

        logger.info("Initializing voice assistant components...")
        try:
            self.tool_registry = ToolRegistry(self.settings)
            self.stt_service: SpeechToText = RealtimeSTTService(self.settings)
            self.agent_service: Agent = GeminiAgent(self.settings, self.tool_registry)
            self.tts_service: TextToSpeech = KokoroTTS(self.settings)
            logger.success("Voice assistant initialized successfully.")
        except VoiceAssistantError as e:
            logger.opt(exception=e).critical(f"A critical error occurred during initialization: {e}")
            raise
        except Exception as e:
            logger.opt(exception=e).critical("An unexpected error occurred during initialization.")
            raise

    def _update_history(self, *messages: ChatMessage) -> None:
        """Appends new messages to the conversation history."""
        self.history.extend(messages)

    async def run(self) -> None:
        """
        The main application loop.

        This method continuously listens for user input, processes it with the agent,
        and synthesizes the response as speech.
        """
        logger.info("Starting the main assistant loop...")
        while True:
            try:
                final_transcription = ""
                async for text in self.stt_service.transcribe():
                    final_transcription = text

                if not final_transcription:
                    continue  # No speech detected, loop again.

                user_message = ChatMessage(role=Role.USER, content=final_transcription)

                # --- THINK ---
                logger.info("Agent is processing the request...")
                full_response_text = ""
                response_stream = self.agent_service.get_response(
                    history=self.history,
                    last_user_message=user_message.content
                )

                async for chunk in response_stream:
                    if chunk.content:
                        full_response_text += chunk.content

                if not full_response_text.strip():
                    logger.warning("Agent produced an empty response.")
                    continue

                assistant_message = ChatMessage(role=Role.ASSISTANT, content=full_response_text)
                logger.info(f"Assistant response: '{full_response_text}'")

                # --- SPEAK ---
                await self.tts_service.synthesize(assistant_message.content)

                # --- UPDATE HISTORY ---
                self._update_history(user_message, assistant_message)

            except VoiceAssistantError as e:
                logger.error(f"A handled error occurred in the main loop: {e}")
                await self.tts_service.synthesize("I'm sorry, a problem occurred.")
            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.info("Assistant shutting down.")
                break
            except Exception as e:
                logger.opt(exception=e).error("An unexpected error occurred in the main loop.")
                await self.tts_service.synthesize("I'm sorry, an unexpected error happened.")
                # Optional: Add a small delay to prevent rapid-fire error loops
                await asyncio.sleep(2)