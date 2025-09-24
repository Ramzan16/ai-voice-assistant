from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional, List
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class STTModel(str, Enum):
    BASE = "base"
    BASE_EN = "base.en"
    TINY = "tiny"
    MEDIUM = "medium"
    LARGE = "large"

class TTSVoice(str, Enum):
    BELLA = "af_bella"
    HEART = "af_heart"
    NICOLE = "af_nicole"
    SARAH = "af_sarah"
    MICHAEL = "am_michael"
    
class Settings(BaseSettings):
    """Application settings with validation and type checking."""
    
    # API Keys
    gemini_api_key: str = Field(..., min_length=1)
    tavily_api_key: Optional[str] = Field(None, min_length=1)
    notion_api_key: Optional[str] = Field(None, min_length=1)
    spotify_api_key: Optional[str] = Field(None, min_length=1)

    # Model Configuration
    llm_model: str = Field("gemini-2.0-flash-exp", description="LLM model to use")
    llm_temperature: float = Field(0.0, ge=0.0, le=2.0)
    stt_model: STTModel = Field(STTModel.TINY)
    tts_voice: TTSVoice = Field(TTSVoice.BELLA)
    tts_device: str = Field("auto", description="Device for TTS: auto, cpu, or cuda")
    
    # Audio Configuration
    sample_rate: int = Field(24000, gt=0)
    microphone_index: Optional[int] = Field(None, ge=0)
    
    # Application Configuration
    log_level: LogLevel = Field(LogLevel.INFO)
    wake_words: List[str] = Field(["jarvis", "assistant"])
    max_recording_duration: float = Field(30.0, gt=0)
    
    # Tool Configuration
    enable_web_search: bool = Field(True)
    enable_calculator: bool = Field(True)
    calculator_timeout: float = Field(5.0, gt=0)
    web_search_timeout: float = Field(10.0, gt=0)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("tts_device")
    def validate_device(cls, v):
        if v == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

# Singleton instance
settings = Settings()