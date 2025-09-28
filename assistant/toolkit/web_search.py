from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
from loguru import logger
import os

# Load environment variables from a .env file at the module level
load_dotenv()

@tool
def search_tool(query: str) -> str:
    """
    Searches the web for the given query using the Tavily search engine.
    Returns a concise answer based on the search results.
    Use this for questions about recent events, facts, or information not in your knowledge base.
    """
    logger.info(f"Executing web search for query: '{query}'")
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.error("TAVILY_API_KEY not found in environment variables.")
            return "Sorry, the web search tool is not configured. Missing API key."
            
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query=query, search_depth="basic")
        
        # Check if there are any results before accessing them
        if response and response.get('results'):
            return response['results'][0]['content']
        else:
            logger.warning(f"No results found for query: '{query}'")
            return "Sorry, I couldn't find any information for that query."

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return f"Sorry, I couldn't perform the web search. Error: {e}"