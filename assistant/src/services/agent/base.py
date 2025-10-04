"""
Base interfaces for conversational agents.

This module re-exports the abstract base class `Agent` from the core
interfaces to provide a clear and consistent entry point for all agent
implementations.
"""

from ...core.interfaces import Agent

__all__ = ["Agent"]