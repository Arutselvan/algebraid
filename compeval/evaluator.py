"""
COMPEVAL Evaluator - Multi-Dimensional Evaluation Suite.

Computes accuracy metrics across multiple axes:
- Overall accuracy
- Accuracy by composition depth (the primary "depth curve")
- Accuracy by task family
- Accuracy by Hupkes compositionality dimension
- Compositional ceiling (max depth where accuracy >= threshold)
- Algebraic Complexity Metrics (H_alg, D_comm, O_c, I_s)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .task_model import Task, TaskSet, TaskFamily, CompositionDimension
from .tasks.verifier import check_answer
from .complexity import compute_complexity, AlgebraicComplexity


@dataclass
class EvalResult:
    """Result of evaluating a single task."""
    task_id: str
    correct: bool
    model_response: str
    ground_truth: str
    depth: int
    family: str
    dimension: str
    complexity: Optional[AlgebraicComplexity] = None


@dataclass
class EvalReport:
    """Comprehensive evaluation report."""
    model_name: str
    task_set_name: str
    total_tasks: int
    total_correct: int

    # Accuracy breakdowns
    accuracy_overall: float
    accuracy_by_depth: Dict[int, Dict[str, Any]]
    accuracy_by_family: Dict[str, Dict[str, Any]]
    accuracy_by_dimension: Dict[str, Dict[str, Any]]

    # Compositional ceiling
    compositional_ceiling_50: Optional[int]
    compositional_ceiling_25: Optional[int]

    # Algebraic Complexity Metrics (averages across all tasks)
    avg_algebraic_entropy: float = 0.0
    avg_commutativity_distance: float = 0.0
    avg_orbit_complexity: float = 0.0
    avg_structural_interference: float = 0.0

    # Raw results
    results: List[EvalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_set_name": self.task_set_name,
            "total_tasks": self.total_tasks,
            "total_correct": self.total_correct,
            "accuracy_overall": round(self.accuracy_overall, 4),
            "compositional_ceiling_50": self.compositional_ceiling_50,
            "compositional_ceiling_25": self.compositional_ceiling_25,
            "algebraic_complexity": {
                "avg_algebraic_entropy": round(self.avg_algebraic_entropy, 4),
                "avg_commutativity_distance": round(self.avg_commutativity_distance, 4),
                "avg_orbit_complexity": round(self.avg_orbit_complexity, 4),
                "avg_structural_interference": round(self.avg_structural_interference, 4),
            },
            "accuracy_by_depth": {
                str(k): {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": round(v["accuracy"], 4),
                }
                for k, v in sorted(self.accuracy_by_depth.items())
            },
            "accuracy_by_family": {
                k: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": round(v["accuracy"], 4),
                }
                for k, v in self.accuracy_by_family.items()
            },
            "accuracy_by_dimension": {
                k: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": round(v["accuracy"], 4),
                }
                for k, v in self.accuracy_by_dimension.items()
                if v["total"] > 0
            },
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  COMPEVAL Evaluation Report")
        print(f"  Model: {self.model_name}")
        print(f"  Task Set: {self.task_set_name}")
        print(f"{'='*60}")
        print(f"\n  Overall Accuracy: {self.accuracy_overall:.1%} ({self.total_correct}/{self.total_tasks})")
        print(f"  Compositional Ceiling (50%): depth {self.compositional_ceiling_50}")
        print(f"  Compositional Ceiling (25%): depth {self.compositional_ceiling_25}")

        print(f"\n  Algebraic Complexity Metrics (averages):")
        print(f"    H_alg  (Algebraic Entropy):        {self.avg_algebraic_entropy:.4f}")
        print(f"    D_comm (Commutativity Distance):   {self.avg_commutativity_distance:.4f}")
        print(f"    O_c    (Orbit Complexity):         {self.avg_orbit_complexity:.4f}")
        print(f"    I_s    (Structural Interference):  {self.avg_structural_interference:.4f}")

        print(f"\n  {'Depth':<8} {'Correct':<10} {'Total':<8} {'Accuracy':<10}")
        print(f"  {'-'*40}")
        for depth in sorted(self.accuracy_by_depth.keys()):
            d = self.accuracy_by_depth[depth]
            print(f"  {depth:<8} {d['correct']:<10} {d['total']:<8} {d['accuracy']:.1%}")

        print(f"\n  {'Family':<30} {'Correct':<10} {'Total':<8} {'Accuracy':<10}")
        print(f"  {'-'*55}")
        for family, d in self.accuracy_by_family.items():
            print(f"  {family:<30} {d['correct']:<10} {d['total']:<8} {d['accuracy']:.1%}")

        dim_data = {k: v for k, v in self.accuracy_by_dimension.items() if v["total"] > 0}
        if dim_data:
            print(f"\n  {'Dimension':<25} {'Correct':<10} {'Total':<8} {'Accuracy':<10}")
            print(f"  {'-'*50}")
            for dim, d in dim_data.items():
                print(f"  {dim:<25} {d['correct']:<10} {d['total']:<8} {d['accuracy']:.1%}")

        print(f"\n{'='*60}\n")


class CompEvalEvaluator:
    """Evaluates model predictions against a COMPEVAL task set."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def evaluate(
        self,
        task_set: TaskSet,
        predictions: Dict[str, str],
        model_name: str = "unknown",
    ) -> EvalReport:
        results: List[EvalResult] = []
        depth_stats: defaultdict = defaultdict(lambda: {"correct": 0, "total": 0})
        family_stats: defaultdict = defaultdict(lambda: {"correct": 0, "total": 0})
        dim_stats: defaultdict = defaultdict(lambda: {"correct": 0, "total": 0})

        complexity_totals = {
            "algebraic_entropy": 0.0,
            "commutativity_distance": 0.0,
            "orbit_complexity": 0.0,
            "structural_interference": 0.0,
        }
        complexity_count = 0

        for task in task_set:
            response: str = predictions.get(task.task_id, "")
            correct: bool = check_answer(response, task.answer, strict=self.strict)

            # Compute algebraic complexity
            try:
                complexity = compute_complexity(task)
                complexity_totals["algebraic_entropy"] += complexity.algebraic_entropy
                complexity_totals["commutativity_distance"] += complexity.commutativity_distance
                complexity_totals["orbit_complexity"] += complexity.orbit_complexity
                complexity_totals["structural_interference"] += complexity.structural_interference
                complexity_count += 1
            except Exception:
                complexity = None

            result = EvalResult(
                task_id=task.task_id,
                correct=correct,
                model_response=response,
                ground_truth=task.answer,
                depth=task.depth,
                family=task.family.value if hasattr(task.family, "value") else task.family,
                dimension=task.dimension.value if hasattr(task.dimension, "value") else task.dimension,
                complexity=complexity,
            )
            results.append(result)

            depth_stats[task.depth]["total"] += 1
            family_stats[result.family]["total"] += 1
            dim_stats[result.dimension]["total"] += 1

            if correct:
                depth_stats[task.depth]["correct"] += 1
                family_stats[result.family]["correct"] += 1
                dim_stats[result.dimension]["correct"] += 1

        total = len(results)
        total_correct = sum(1 for r in results if r.correct)

        for stats_dict in [depth_stats, family_stats, dim_stats]:
            for k, v in stats_dict.items():
                v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0

        ceiling_50 = self._find_ceiling(depth_stats, 0.50)
        ceiling_25 = self._find_ceiling(depth_stats, 0.25)

        n = complexity_count or 1
        return EvalReport(
            model_name=model_name,
            task_set_name=task_set.name,
            total_tasks=total,
            total_correct=total_correct,
            accuracy_overall=total_correct / total if total > 0 else 0.0,
            accuracy_by_depth=dict(depth_stats),
            accuracy_by_family=dict(family_stats),
            accuracy_by_dimension=dict(dim_stats),
            compositional_ceiling_50=ceiling_50,
            compositional_ceiling_25=ceiling_25,
            avg_algebraic_entropy=complexity_totals["algebraic_entropy"] / n,
            avg_commutativity_distance=complexity_totals["commutativity_distance"] / n,
            avg_orbit_complexity=complexity_totals["orbit_complexity"] / n,
            avg_structural_interference=complexity_totals["structural_interference"] / n,
            results=results,
        )

    def _find_ceiling(self, depth_stats: Dict, threshold: float) -> Optional[int]:
        ceiling = None
        for depth in sorted(depth_stats.keys()):
            if depth_stats[depth]["accuracy"] >= threshold:
                ceiling = depth
        return ceiling
