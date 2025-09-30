"""
Services for Speech-to-Text (STT) transcription.

This package provides concrete implementations of the `SpeechToText` interface.
"""

from .base import SpeechToText
from .realtime_stt import RealtimeSTTService

__all__ = [
    "SpeechToText",
    "RealtimeSTTService",
]