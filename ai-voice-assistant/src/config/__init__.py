"""config package — re-export Settings and the singleton settings instance."""

from .settings import settings, Settings, LogLevel, STTModel, TTSVoice

__all__ = [
    "settings",
    "Settings",
    "LogLevel",
    "STTModel",
    "TTSVoice",
]
