import sys
import logging
import logging.handlers
from .settings import settings
import os

def setup_logging() -> None:
    """Configure global Python logging."""
    
    # Define the format for the logs
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Set the global logging level from your settings
    # Note: Ensure settings.log_level.value corresponds to a logging level
    # e.g., "INFO" should map to logging.INFO
    log_level = getattr(logging, settings.log_level.value.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # --- Console Handler ---
    # This handler prints logs to the console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)
    
    # --- File Handler ---
    # This handler writes logs to a file with rotation and retention
    
    # Ensure the 'logs' directory exists
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Use TimedRotatingFileHandler for time-based retention ("7 days")
    # This will create a new log file every day at midnight and keep 7 old files.
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=f"{log_dir}/voice_assistant.log",
        when="midnight",  # Rotate at midnight
        interval=1,       # Daily rotation
        backupCount=7,    # Keep 7 old log files (equivalent to 7 days retention)
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    # Optional: If you prefer size-based rotation ("10 MB") instead of time-based:
    # file_handler = logging.handlers.RotatingFileHandler(
    #     filename="logs/voice_assistant.log",
    #     maxBytes=10 * 1024 * 1024,  # 10 MB
    #     backupCount=5,              # Keep 5 old log files
    #     encoding="utf-8",
    # )

# Example of how to use it in another file:
# import logging
# logger = logging.getLogger(__name__)
# logger.info("This is an informational message.")