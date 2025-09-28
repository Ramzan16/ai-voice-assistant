from langchain.tools import tool
import logging

@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluates a simple mathematical expression.
    Use this for calculations like '5*7', '10+20/2', etc.
    It cannot solve complex algebraic equations.
    """
    logging.info(f"Executing calculation for expression: '{expression}'")
    try:
        # WARNING: eval() can be dangerous if the input is not sanitized.
        # For a production system, use a safer math parsing library.
        allowed_chars = "0123456789+-*/.() "
        if all(char in allowed_chars for char in expression):
            result = eval(expression)
            return f"The result of '{expression}' is {result}."
        else:
            return "Error: Invalid characters in expression."
    except Exception as e:
        logging.error(f"Calculation failed: {e}")
        return f"Sorry, I couldn't calculate that. Error: {e}"
