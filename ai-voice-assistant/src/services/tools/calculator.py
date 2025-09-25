import ast
import operator
import asyncio
from typing import Any, Dict
from functools import wraps
from concurrent.futures import TimeoutError
from src.core.interfaces import ToolInterface
from src.core.exceptions import ToolError
from loguru import logger

class SafeCalculator:
    """Mathematical expression evaluator."""
    
    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Supported functions
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
    }
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.current_depth = 0
    
    def evaluate(self, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Calculated result
            
        Raises:
            ToolError: If expression is invalid or unsafe
        """
        try:
            # Parse expression into AST
            node = ast.parse(expression, mode='eval')
            return self._eval_node(node.body)
        except (SyntaxError, ValueError) as e:
            raise ToolError(f"Invalid expression: {e}")
        except RecursionError:
            raise ToolError("Expression too complex")
        except Exception as e:
            raise ToolError(f"Calculation error: {e}")
    
    def _eval_node(self, node: ast.AST) -> float:
        """Recursively evaluate AST nodes."""
        
        # Check recursion depth
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            raise RecursionError("Maximum evaluation depth exceeded")
        
        try:
            # Numbers
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ToolError(f"Unsupported constant type: {type(node.value)}")
            
            # Binary operations
            elif isinstance(node, ast.BinOp):
                op = type(node.op)
                if op not in self.OPERATORS:
                    raise ToolError(f"Unsupported operator: {op.__name__}")
                
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                
                # Prevent division by zero
                if op in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                    raise ToolError("Division by zero")
                
                return self.OPERATORS[op](left, right)
            
            # Unary operations
            elif isinstance(node, ast.UnaryOp):
                op = type(node.op)
                if op not in self.OPERATORS:
                    raise ToolError(f"Unsupported unary operator: {op.__name__}")
                
                operand = self._eval_node(node.operand)
                return self.OPERATORS[op](operand)
            
            # Function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.FUNCTIONS:
                        raise ToolError(f"Unsupported function: {func_name}")
                    
                    args = [self._eval_node(arg) for arg in node.args]
                    return self.FUNCTIONS[func_name](*args)
                else:
                    raise ToolError("Complex function calls not supported")
            
            else:
                raise ToolError(f"Unsupported expression type: {type(node).__name__}")
                
        finally:
            self.current_depth -= 1


class CalculatorTool(ToolInterface):
    """Calculator tool for the agent."""
    
    def __init__(self, timeout: float = 5.0):
        self.calculator = SafeCalculator()
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return """Evaluates mathematical expressions safely.
        Supports: +, -, *, /, **, %, //
        Functions: abs, round, min, max, sum
        Examples: '5*7', '10+20/2', 'round(3.14159, 2)'"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str) -> str:
        """
        Execute calculator with timeout protection.
        
        Args:
            expression: Mathematical expression
            
        Returns:
            Calculation result as string
        """
        logger.info(f"Calculating: {expression}")
        
        try:
            # Run calculation with timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.calculator.evaluate, expression),
                timeout=self.timeout
            )
            
            # Format result nicely
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            
            return f"The result of '{expression}' is {result}"
            
        except asyncio.TimeoutError:
            logger.error(f"Calculation timeout for: {expression}")
            return f"Calculation took too long (>{self.timeout}s)"
        except ToolError as e:
            logger.error(f"Calculation error: {e}")
            return f"Calculation error: {e}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Unexpected error during calculation: {e}"
    
    async def validate_input(self, expression: str) -> bool:
        """Validate expression before execution."""
        if not expression or not isinstance(expression, str):
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = ['import', '__', 'eval', 'exec', 'compile', 'open']
        expression_lower = expression.lower()
        
        for pattern in suspicious_patterns:
            if pattern in expression_lower:
                logger.warning(f"Suspicious pattern '{pattern}' in expression: {expression}")
                return False
        
        return True