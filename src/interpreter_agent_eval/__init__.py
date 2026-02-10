"""Interpreter Agent Evaluation Framework.

A framework for evaluating interpreter/translator agents between users speaking different languages.
"""

from .user import User
from .interpreter import InterpreterAgent
from .evaluator import EvaluationFramework

__version__ = "0.1.0"
__all__ = ["User", "InterpreterAgent", "EvaluationFramework"]
