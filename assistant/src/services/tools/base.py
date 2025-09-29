from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_core.tools import BaseTool


class Tool(ABC):
    """
    Abstract base class for all tools that the agent can use.
    It defines the common interface that the ToolRegistry and Agent expect.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool (e.g., 'calculator')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """A detailed description of what the tool does and when to use it."""
        raise NotImplementedError

    @property
    @abstractmethod
    def args_schema(self) -> type:
        """The Pydantic model defining the arguments for the tool."""
        raise NotImplementedError

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> str:
        """
        The core logic of the tool. This method is called with arguments
        validated against the `args_schema`.

        Returns:
            A string representing the result of the tool's execution.
        """
        raise NotImplementedError

    def to_langchain_tool(self) -> BaseTool:
        """Converts this tool instance into a LangChain-compatible tool."""
        from langchain_core.tools import tool
        
        # This is a bit of a clever trick to dynamically create a LangChain tool
        # from our interface. It wraps the `_execute` method with the necessary
        # decorators and metadata.
        @tool(name=self.name, description=self.description, args_schema=self.args_schema)
        async def dynamic_tool(**kwargs: Any) -> str:
            return await self._execute(**kwargs)
        
        return dynamic_tool