import ast
import operator
import asyncio
from pydantic import BaseModel, Field
from loguru import logger

from src.config import Settings
from src.core.exceptions import ToolError
from .base import Tool


class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate.")


class SafeCalculator:
    """A safe mathematical expression evaluator using Python's AST."""

    OPERATORS = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
    }
    
    def evaluate(self, expression: str) -> float | int:
        """Safely evaluates a mathematical expression string."""
        try:
            node = ast.parse(expression, mode='eval').body
            return self._eval_node(node)
        except (SyntaxError, TypeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Invalid calculator expression '{expression}': {e}")
            raise ToolError(f"Invalid expression: {e}")
        except Exception as e:
            logger.opt(exception=e).error(f"Error evaluating expression: {expression}")
            raise ToolError(f"Calculation error: {e}")

    def _eval_node(self, node):
        """Recursively evaluates an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            return self.OPERATORS[type(node.op)](operand)
        else:
            raise ToolError(f"Unsupported expression type: {type(node).__name__}")


class CalculatorTool(Tool):
    """Calculator tool for the agent to perform mathematical calculations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.calculator = SafeCalculator()

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluates mathematical expressions safely. Supports +, -, *, /, **."

    @property
    def args_schema(self) -> type[BaseModel]:
        return CalculatorInput

    async def _execute(self, expression: str) -> str:
        logger.info(f"Calculating: {expression}")
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.calculator.evaluate, expression),
                timeout=self.settings.calculator_timeout
            )
            
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            
            return f"The result is {result}"

        except asyncio.TimeoutError:
            logger.error(f"Calculation timeout for: {expression}")
            return f"Calculation took too long (>{self.settings.calculator_timeout}s)"
        except ToolError as e:
            logger.error(f"Calculation error for '{expression}': {e}")
            return f"Calculation error: {e}"