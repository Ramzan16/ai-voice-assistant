"""config package — re-export Settings and the singleton settings instance."""

from .settings import settings, Settings, LogLevel, STTModel, TTSVoice
from .logging_config import setup_logging

__all__ = [
    "settings",
    "Settings",
    "setup_logging",
    "LogLevel",
    "STTModel",
    "TTSVoice",
]
