"""
ALGEBRAID — A procedurally generated benchmark for compositional
generalization using formal algebraic structures.
"""

from .generator import AlgebraidGenerator
from .evaluator import AlgebraidEvaluator, EvalReport
from .tasks.verifier import check_answer, extract_answer
from .tasks.validator import TaskValidator, validate_file, print_report
from .task_model import Task, TaskSet, TaskFamily, CompositionDimension
from .complexity import compute_complexity, AlgebraicComplexity

__version__ = "2.1.0"

__all__: list = [
    "AlgebraidGenerator",
    "AlgebraidEvaluator",
    "EvalReport",
    "check_answer",
    "extract_answer",
    "TaskValidator",
    "validate_file",
    "print_report",
    "Task",
    "TaskSet",
    "TaskFamily",
    "CompositionDimension",
    "compute_complexity",
    "AlgebraicComplexity",
]
