"""
ALGEBRAID - A procedurally generated benchmark for compositional
generalization using formal algebraic structures.
"""

from .generator import AlgebraidGenerator
from .evaluator import AlgebraidEvaluator, EvalReport, EvalResult
from .tasks.verifier import check_answer, extract_answer
from .tasks.validator import TaskValidator, validate_file, print_report
from .task_model import Task, TaskSet, TaskFamily, CompositionDimension
from .complexity import (
    compute_complexity, AlgebraicComplexity,
    compute_conceptual_depth, compute_adversarial_strength,
)
from .proof import verify_task, verify_set, print_proof_report, ProofResult
from .analysis import (
    fit_scaling_law, fit_scaling_law_by_family,
    find_phase_transition,
    hallucination_onset, stability_breakdown, run_analysis, print_analysis,
)
from .splits import (
    split_by_depth, split_by_commutativity, split_by_structure,
    split_by_family, split_summary,
)

__version__ = "2.1.0"

__all__: list = [
    # Core
    "AlgebraidGenerator",
    "AlgebraidEvaluator",
    "EvalReport",
    "EvalResult",
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
    "compute_conceptual_depth",
    "compute_adversarial_strength",
    # Proof (Blocker 1)
    "verify_task",
    "verify_set",
    "print_proof_report",
    "ProofResult",
    # Analysis (Blocker 2)
    "fit_scaling_law",
    "fit_scaling_law_by_family",
    "find_phase_transition",
    "hallucination_onset",
    "stability_breakdown",
    "run_analysis",
    "print_analysis",
    # Splits (Blocker 4)
    "split_by_depth",
    "split_by_commutativity",
    "split_by_structure",
    "split_by_family",
    "split_summary",
]
