"""
Custom exception hierarchy for the voice assistant application.

This module defines a set of custom exceptions to allow for more specific
error handling and clearer, more organized error-catching logic throughout
the application. All custom exceptions inherit from a common VoiceAssistantError base class.
"""


class VoiceAssistantError(Exception):
    """Base exception class for all application-specific errors."""
    def __init__(self, message="An unexpected error occurred in the voice assistant."):
        self.message = message
        super().__init__(self.message)


# --- Configuration and Orchestration Errors ---

class ConfigurationError(VoiceAssistantError):
    """Raised for configuration-related issues, like missing API keys."""
    def __init__(self, message="A configuration error was detected."):
        super().__init__(message)


class OrchestratorError(VoiceAssistantError):
    """Raised for errors in the main orchestration logic."""
    def __init__(self, message="An error occurred in the main application orchestrator."):
        super().__init__(message)


# --- Service Errors ---

class ServiceError(VoiceAssistantError):
    """Base class for errors related to external or internal services."""
    def __init__(self, message="A service-level error occurred."):
        super().__init__(message)


class STTError(ServiceError):
    """Raised for errors in the Speech-to-Text service."""
    def __init__(self, message="An error occurred in the Speech-to-Text service."):
        super().__init__(message)


class TTSError(ServiceError):
    """Raised for errors in the Text-to-Speech service."""
    def __init__(self, message="An error occurred in the Text-to-Speech service."):
        super().__init__(message)


class AgentError(ServiceError):
    """Raised for errors within the conversational agent or LLM."""
    def __init__(self, message="An error occurred within the conversational agent."):
        super().__init__(message)


# --- Tool Errors ---

class ToolError(VoiceAssistantError):
    """Raised when a tool fails to execute properly."""
    def __init__(self, message="A tool failed during execution."):
        super().__init__(message)