from abc import ABC, abstractmethod
from typing import AsyncIterator, List

from .types import ChatMessage


class SpeechToText(ABC):
    """Abstract base class for speech-to-text services."""

    @abstractmethod
    async def transcribe(self) -> AsyncIterator[str]:
        """
        Captures audio and yields transcribed text.

        The generator should yield intermediate transcription results as they become
        available. The **final yielded string** must be the complete and finalized
        transcription of the user's speech.
        """
        raise NotImplementedError


class TextToSpeech(ABC):
    """Abstract base class for text-to-speech services."""

    @abstractmethod
    async def synthesize(self, text: str) -> None:
        """
        Synthesizes text into speech and plays it.

        Args:
            text (str): The text to be converted to speech.

        Note:
            This method returns None ("fire and forget"), but could be extended
            to return playback metadata (e.g., duration) if needed.
        """
        raise NotImplementedError


class Agent(ABC):
    """Abstract base class for the conversational agent."""

    @abstractmethod
    async def get_response(
        self,
        history: List[ChatMessage],
        last_user_message: str,
    ) -> AsyncIterator[ChatMessage]:
        """
        Processes user input and generates a stream of response messages.

        The conversation flow for tool use is as follows:
        1. The agent yields a ChatMessage with `tool_calls`.
        2. The orchestrator executes the tools and sends back a ChatMessage
           with `role="tool"` and the `content` from a ToolResult.
        3. The agent processes the tool result and yields the final assistant
           message with `role="assistant"` and content to be spoken.

        Args:
            history (List[ChatMessage]): The conversation history up to this point.
            last_user_message (str): The latest transcribed message from the user.
        """
        raise NotImplementedError