"""
Base interfaces for Speech-to-Text services.

This module re-exports the abstract base class `SpeechToText` from the core
interfaces to provide a clear and consistent entry point for all STT implementations.
"""

from ...core.interfaces import SpeechToText

__all__ = ["SpeechToText"]