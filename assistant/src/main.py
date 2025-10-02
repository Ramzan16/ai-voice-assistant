"""
Main entry point for the AI Voice Assistant application.

This script initializes the application by:
1. Setting up the global logging configuration.
2. Loading application settings from environment variables and/or a .env file.
3. Instantiating the main VoiceAssistant orchestrator.
4. Running the assistant's main asynchronous loop.
"""

import asyncio
from loguru import logger

# Note: The import paths are now relative to the `src` directory,
# assuming you run the application as a module (e.g., `python -m src.main`).
from config import settings, setup_logging
from orchestrator import VoiceAssistant
from core.exceptions import VoiceAssistantError


async def main() -> None:
    """
    Initializes and runs the voice assistant.
    """
    # 1. Configure logging as the very first step
    setup_logging()

    logger.info("Application starting up...")

    try:
        # 2. The `settings` object is already imported, loading the configuration.
        # We can log some settings for verification, but avoid logging secrets.
        logger.debug(f"Log Level set to: {settings.log_level.value}")
        logger.debug(f"LLM Model: {settings.llm_model}")
        logger.debug(f"Wake words: {settings.wake_words}")

        # 3. Create an instance of the main orchestrator
        assistant = VoiceAssistant(settings)

        # 4. Start the main application loop
        await assistant.run()

    except VoiceAssistantError as e:
        logger.critical(f"A critical, unrecoverable error occurred during startup: {e}")
    except Exception as e:
        logger.opt(exception=e).critical("An unexpected critical error occurred.")
    finally:
        logger.info("Application shutting down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting.")
    except asyncio.CancelledError:
        logger.info("Asyncio loop was cancelled. Exiting.")
