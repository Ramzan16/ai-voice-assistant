"""
Base interfaces for Text-to-Speech services.

This module re-exports the abstract base class `TextToSpeech` from the core
interfaces to provide a clear and consistent entry point for all TTS implementations.
"""

from assistant.src.core.interfaces import TextToSpeech

__all__ = ["TextToSpeech"]
