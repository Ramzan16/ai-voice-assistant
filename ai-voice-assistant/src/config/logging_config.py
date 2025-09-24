# src/config/logging_config.py
import sys
from loguru import logger
from .settings import settings

def setup_logging() -> None:
    """Configure global Loguru logging."""
    logger.remove()  # Remove default handler
    
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=settings.log_level.value,  # Uses your enum
        enqueue=True,  # For multi-thread/multi-process safety
        backtrace=True,  # Rich tracebacks
        diagnose=True   # Show variable values in tracebacks
    )

    # Example: log to file as well (optional)
    logger.add(
        "logs/voice_assistant.log",
        rotation="10 MB",    # Rotate after 10 MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip",   # Compress old logs
        level=settings.log_level.value,
        enqueue=True,
    )