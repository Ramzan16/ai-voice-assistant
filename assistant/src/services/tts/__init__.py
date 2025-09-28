"""
Services for Text-to-Speech (TTS) synthesis.

This package provides concrete implementations of the `TextToSpeech` interface.
"""

from .base import TextToSpeech
from .kokoro import KokoroTTS

__all__ = [
    "TextToSpeech",
    "KokoroTTS",
]
