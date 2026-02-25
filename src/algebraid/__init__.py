"""
ALGEBRAID: A Procedurally Generated Benchmark for Compositional Generalization
Using Formal Algebraic Structures.
"""

from .generator import AlgebraidGenerator
from .evaluator import AlgebraidEvaluator, EvalReport
from .tasks.verifier import check_answer, extract_answer
from .task_model import Task, TaskSet, TaskFamily, CompositionDimension
from .complexity import compute_complexity, AlgebraicComplexity

__version__ = "2.0.1"

__all__: list = [
    "AlgebraidGenerator",
    "AlgebraidEvaluator",
    "EvalReport",
    "check_answer",
    "extract_answer",
    "Task",
    "TaskSet",
    "TaskFamily",
    "CompositionDimension",
    "compute_complexity",
    "AlgebraicComplexity",
]
