# from .web_search import search_tool
from .calculator import calculator_tool

# This function acts as a registry for all available tools.
# When you create a new tool, simply import it and add it to this list.
def load_all_tools():
    """
    Loads and returns a list of all available tool functions.
    """
    return [
        # search_tool,
        calculator_tool,
        # Add new tools here
    ]