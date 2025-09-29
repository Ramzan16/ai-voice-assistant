"""
Tools package for the voice assistant.

This package contains all the tools the agent can use, the base class for
creating new tools, and the registry that discovers and manages them.
"""

from .base import Tool
from .registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry"]