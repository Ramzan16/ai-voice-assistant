import os
import importlib
import inspect
from typing import Dict, List, Any
from loguru import logger
from langchain_core.messages import ToolMessage

# --- FIX: Use a relative import to reliably find modules within the same package ---
from ...config import Settings
from .base import Tool


class ToolRegistry:
    """
    Dynamically discovers, loads, and manages all available tools.
    This allows for easy extension by simply adding new tool files.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the registry and loads tools from the specified directory.

        Args:
            settings: The application settings object, passed to each tool.
        """
        self.settings = settings
        self.tools: Dict[str, Tool] = {}
        self._load_tools()

    def _load_tools(self):
        """
        Scans the 'tools' directory, imports modules, and instantiates tool classes.
        """
        tools_dir = os.path.dirname(__file__)
        logger.info(f"Scanning for tools in: {tools_dir}")

        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and not filename.startswith(("_", "base", "registry")):
                module_name = filename[:-3]
                
                # --- FIX: Use __package__ for robust dynamic imports ---
                module_path = f"{__package__}.{module_name}"
                
                try:
                    module = importlib.import_module(module_path)
                    for _, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Tool) and obj is not Tool:
                            tool_instance = obj(self.settings)
                            if tool_instance.name in self.tools:
                                logger.warning(f"Duplicate tool name '{tool_instance.name}' found. Overwriting.")
                            self.tools[tool_instance.name] = tool_instance
                            logger.success(f"Successfully loaded tool: '{tool_instance.name}'")
                except Exception as e:
                    logger.opt(exception=e).error(f"Failed to load tool from {module_name}")

    def get_tool(self, name: str) -> Tool | None:
        """Retrieves a tool instance by its name."""
        return self.tools.get(name)

    def get_all_tools(self) -> List[Tool]:
        """Returns a list of all loaded tool instances."""
        return list(self.tools.values())

    def get_all_tools_langchain(self) -> List:
        """Returns all tools converted to their LangChain-compatible format."""
        return [tool.to_langchain_tool() for tool in self.tools.values()]
        
    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any], tool_call_id: str) -> ToolMessage:
        """
        Finds and executes a tool by name with the given arguments.
        """
        tool_to_call = self.get_tool(tool_name)
        if not tool_to_call:
            error_msg = f"Error: Tool '{tool_name}' not found."
            logger.error(error_msg)
            return ToolMessage(content=error_msg, tool_call_id=tool_call_id)

        try:
            output = await tool_to_call._execute(**tool_args)
            return ToolMessage(content=str(output), tool_call_id=tool_call_id)
        except Exception as e:
            logger.opt(exception=e).error(f"Error executing tool '{tool_name}'")
            return ToolMessage(content=f"Error: {e}", tool_call_id=tool_call_id)
