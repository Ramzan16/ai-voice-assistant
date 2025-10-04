import asyncio
from pydantic import BaseModel, Field
from tavily import TavilyClient
from loguru import logger

from ...config import Settings
from .base import Tool


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to find information on the web.")


class WebSearchTool(Tool):
    """A tool for performing web searches using the Tavily API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if not self.settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set in the environment or .env file.")
        self.client = TavilyClient(api_key=self.settings.tavily_api_key)

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Searches the web for the given query. Use this for questions about "
            "recent events, facts, or information not in your knowledge base."
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return WebSearchInput

    async def _execute(self, query: str) -> str:
        logger.info(f"Executing web search for query: '{query}'")
        try:
            loop = asyncio.get_event_loop()
            
            # Tavily's search is async-compatible, but we run it in an executor
            # to be consistent and apply a timeout easily.
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    lambda: self.client.search(query=query, search_depth="basic")
                ),
                timeout=self.settings.web_search_timeout
            )

            if response and response.get('results'):
                return response['results'][0]['content']
            else:
                logger.warning(f"No web search results found for query: '{query}'")
                return "Sorry, I couldn't find any information for that query."

        except asyncio.TimeoutError:
            logger.error(f"Web search timed out for query: {query}")
            return f"Web search took too long (>{self.settings.web_search_timeout}s)."
        except Exception as e:
            logger.opt(exception=e).error(f"Tavily search failed for query: {query}")
            return f"Sorry, the web search failed. Error: {e}"