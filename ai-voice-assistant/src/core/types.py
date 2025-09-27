from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Enumeration for the roles in a chat message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FunctionCall(BaseModel):
    """
    A structured representation of a function call.
    """
    name: str = Field(..., description="The name of the function to call.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="A dictionary of arguments to pass to the function."
    )


class ToolCall(BaseModel):
    """
    Represents a tool call requested by the agent.
    """
    id: str = Field(..., description="A unique identifier for this tool call.")
    function: FunctionCall = Field(..., description="The function to be called.")


class ToolResult(BaseModel):
    """
    Represents the result of a tool execution.
    This will be converted to a ChatMessage with role='tool'.
    """
    tool_call_id: str
    content: str = Field(
        ...,
        description="The string-serialized result of the tool function (e.g., JSON)."
    )


class ChatMessage(BaseModel):
    """Represents a single message in the conversation history."""
    role: Role
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


ChatMessage.model_rebuild()