"""
ALGEBRAID: A Procedurally Generated Benchmark for Compositional Generalization
Using Formal Algebraic Structures.
"""

from .generator import AlgebraidGenerator
from .evaluator import AlgebraidEvaluator, EvalReport
from .tasks.verifier import check_answer, extract_answer
from .tasks.validator import TaskValidator, validate_jsonl_file, print_validation_report
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
    "validate_jsonl_file",
    "print_validation_report",
    "Task",
    "TaskSet",
    "TaskFamily",
    "CompositionDimension",
    "compute_complexity",
    "AlgebraicComplexity",
]
