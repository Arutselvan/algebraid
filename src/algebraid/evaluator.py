"""
Multi-dimensional evaluation suite.

Scores model predictions along composition depth, task family, compositional
dimension, and four algebraic complexity metrics.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import re

from .task_model import Task, TaskSet, TaskFamily, CompositionDimension
from .tasks.verifier import check_answer, _dihedral_canonical, _quaternion_canonical
from .complexity import AlgebraicComplexity


def _compute_stats(results, key_fn):
    """Compute {key: {correct, total, accuracy}} from a list of EvalResults."""
    stats = {}
    for r in results:
        k = key_fn(r)
        if k not in stats:
            stats[k] = {"correct": 0, "total": 0}
        stats[k]["total"] += 1
        if r.correct:
            stats[k]["correct"] += 1
    for v in stats.values():
        v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0
    return stats


_HALLUCINATION_RE = re.compile(
    r"cannot|undefined|infinity|idk|unknown|impossible|not defined|n/a|none|sorry|don.t know",
    re.IGNORECASE,
)


def _to_num(s: str) -> Optional[float]:
    """Try to extract a leading number from a string."""
    try:
        token = s.strip().split()[0] if s.strip() else ""
        cleaned = re.sub(r"[^0-9\-\.]", "", token)
        return float(cleaned) if cleaned else None
    except (ValueError, IndexError):
        return None


def _classify_error(result: "EvalResult") -> str:
    """Classify a wrong answer into a broad failure mode."""
    resp = result.model_response.strip()
    if _HALLUCINATION_RE.search(resp):
        return "hallucination"
    if result.dimension == CompositionDimension.ADVERSARIAL.value:
        return "adversarial_trap"
    is_tuple_answer = result.ground_truth.strip().startswith("(")
    resp_num = None if is_tuple_answer else _to_num(resp)
    gt_num = None if is_tuple_answer else _to_num(result.ground_truth)
    if resp_num is not None and gt_num is not None:
        if abs(resp_num - gt_num) == 1:
            return "off_by_one"
        if resp_num != 0 and abs(resp_num + gt_num) <= 1:
            return "inverse_confusion"
    if resp.lower() in ("0", "1", "e", "identity", "(0)", "(1)"):
        return "identity_confusion"
    return "wrong_value"


# Families where depth meaningfully corresponds to task difficulty.
# Shared with analysis.py — define once here to prevent drift.
CHAIN_FAMILIES: frozenset = frozenset({
    "intra-structure composition",
    "inter-structure composition",
    "field arithmetic",
})

# Adversarial and intermediate tasks share the intra-structure family label
# but have artificially constructed depths; exclude them from depth-scaling analyses.
CHAIN_EXCLUDED_DIMENSIONS: frozenset = frozenset({"adversarial", "intermediate_state"})


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
    error_category: Optional[str] = None


class EvalReport:
    """Comprehensive evaluation report."""

    def __init__(
        self,
        model_name: str,
        task_set_name: str,
        total_tasks: int,
        total_correct: int,
        accuracy_overall: float,
        compositional_ceiling_50: Optional[int],
        compositional_ceiling_25: Optional[int],
        avg_algebraic_entropy: float = 0.0,
        avg_commutativity_distance: float = 0.0,
        avg_orbit_complexity: float = 0.0,
        avg_structural_interference: float = 0.0,
        run_id: str = "",
        timestamp: str = "",
        missing_predictions: int = 0,
        errored_predictions: int = 0,
        results: Optional[List[EvalResult]] = None,
    ):
        self.model_name = model_name
        self.task_set_name = task_set_name
        self.total_tasks = total_tasks
        self.total_correct = total_correct
        self.accuracy_overall = accuracy_overall
        self.compositional_ceiling_50 = compositional_ceiling_50
        self.compositional_ceiling_25 = compositional_ceiling_25
        self.avg_algebraic_entropy = avg_algebraic_entropy
        self.avg_commutativity_distance = avg_commutativity_distance
        self.avg_orbit_complexity = avg_orbit_complexity
        self.avg_structural_interference = avg_structural_interference
        self.run_id = run_id
        self.timestamp = timestamp
        self.missing_predictions = missing_predictions
        self.errored_predictions = errored_predictions
        self.results = results if results is not None else []

    @property
    def accuracy_by_depth(self) -> Dict[int, Dict[str, Any]]:
        return _compute_stats(self.results, lambda r: r.depth)

    @property
    def accuracy_by_family(self) -> Dict[str, Dict[str, Any]]:
        return _compute_stats(self.results, lambda r: r.family)

    @property
    def accuracy_by_dimension(self) -> Dict[str, Dict[str, Any]]:
        return _compute_stats(self.results, lambda r: r.dimension)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "task_set_name": self.task_set_name,
            "total_tasks": self.total_tasks,
            "total_correct": self.total_correct,
            "missing_predictions": self.missing_predictions,
            "errored_predictions": self.errored_predictions,
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
            # Compact per-task results for downstream analysis (response capped at 512 chars)
            "results": [
                {
                    "task_id": r.task_id,
                    "correct": r.correct,
                    "model_response": r.model_response[:512],
                    "ground_truth": r.ground_truth,
                    "depth": r.depth,
                    "family": r.family,
                    "dimension": r.dimension,
                    **({"error_category": r.error_category} if r.error_category else {}),
                    **({"complexity": {
                        "H_alg":  round(r.complexity.algebraic_entropy,         4),
                        "D_comm": round(r.complexity.commutativity_distance,    4),
                        "O_c":    round(r.complexity.orbit_complexity,          4),
                        "I_s":    round(r.complexity.structural_interference,   4),
                    }} if r.complexity is not None else {}),
                }
                for r in self.results
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalReport":
        """Reconstruct an EvalReport from a saved dict, including per-task results."""
        results = [
            EvalResult(
                task_id=r["task_id"],
                correct=r["correct"],
                model_response=r.get("model_response", ""),
                ground_truth=r.get("ground_truth", ""),
                depth=r.get("depth", 0),
                family=r.get("family", ""),
                dimension=r.get("dimension", ""),
                complexity=(
                    AlgebraicComplexity(
                        algebraic_entropy=r["complexity"].get("H_alg", 0.0),
                        commutativity_distance=r["complexity"].get("D_comm", 0.0),
                        orbit_complexity=r["complexity"].get("O_c", 0.0),
                        structural_interference=r["complexity"].get("I_s", 0.0),
                    ) if r.get("complexity") else None
                ),
                error_category=r.get("error_category"),
            )
            for r in d.get("results", [])
        ]
        cx = d.get("algebraic_complexity", {})
        return cls(
            run_id=d.get("run_id", ""),
            timestamp=d.get("timestamp", ""),
            model_name=d.get("model_name", "unknown"),
            task_set_name=d.get("task_set_name", "unknown"),
            total_tasks=d.get("total_tasks", 0),
            total_correct=d.get("total_correct", 0),
            missing_predictions=d.get("missing_predictions", 0),
            errored_predictions=d.get("errored_predictions", 0),
            accuracy_overall=d.get("accuracy_overall", 0.0),
            compositional_ceiling_50=d.get("compositional_ceiling_50"),
            compositional_ceiling_25=d.get("compositional_ceiling_25"),
            avg_algebraic_entropy=cx.get("avg_algebraic_entropy", 0.0),
            avg_commutativity_distance=cx.get("avg_commutativity_distance", 0.0),
            avg_orbit_complexity=cx.get("avg_orbit_complexity", 0.0),
            avg_structural_interference=cx.get("avg_structural_interference", 0.0),
            results=results,
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  ALGEBRAID Evaluation Report")
        if self.run_id:
            print(f"  Run      : {self.run_id}")
        print(f"  Model    : {self.model_name}")
        print(f"  Task Set : {self.task_set_name}")
        print(f"{'='*60}")
        print(f"\n  Overall Accuracy: {self.accuracy_overall:.1%}"
              f" ({self.total_correct}/{self.total_tasks})")
        if self.missing_predictions:
            print(f"  Missing Predictions: {self.missing_predictions} (scored as wrong)")
        if self.errored_predictions:
            print(f"  Errored Predictions: {self.errored_predictions} (excluded from scoring)")
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


class AlgebraidEvaluator:
    """Evaluates model predictions against an ALGEBRAID task set."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def evaluate(
        self,
        task_set: TaskSet,
        predictions: Dict[str, str],
        model_name: str = "unknown",
        run_id: str = "",
        timestamp: str = "",
    ) -> EvalReport:
        if not isinstance(predictions, dict):
            raise TypeError(
                f"predictions must be a dict mapping task_id -> response, "
                f"got {type(predictions).__name__}"
            )

        results: List[EvalResult] = []
        missing = 0
        errored = 0

        complexity_totals = {
            "algebraic_entropy": 0.0,
            "commutativity_distance": 0.0,
            "orbit_complexity": 0.0,
            "structural_interference": 0.0,
        }
        complexity_count = 0

        for task in task_set:
            if task.task_id not in predictions:
                missing += 1
            response: str = predictions.get(task.task_id, "")

            # Skip API error responses entirely — do not count as right or wrong
            if response.strip() == "[ERROR]":
                errored += 1
                continue

            try:
                correct: bool = check_answer(response, task.answer, strict=self.strict)
                if not correct and task.answer_raw and task.answer_raw != task.answer:
                    correct = check_answer(response, task.answer_raw, strict=self.strict)
                # Dihedral notation: normalise spacing/composition variants and retry
                if not correct and task.structures and len(task.structures) == 1:
                    m = re.match(r'^D_(\d+)$', task.structures[0])
                    if m:
                        canon = _dihedral_canonical(response, int(m.group(1)))
                        if canon:
                            correct = check_answer(canon, task.answer_raw, strict=self.strict)
                # Q_8 notation: normalise alternative quaternion representations
                if not correct and task.structures == ["Q_8"]:
                    canon = _quaternion_canonical(response)
                    if canon:
                        correct = check_answer(canon, task.answer_raw, strict=self.strict)
            except Exception:
                correct = False

            # Read complexity from task metadata (embedded at generation time
            # or backfilled by TaskSet.from_jsonl for older files).
            meta_cx = (task.metadata or {}).get("complexity")
            if meta_cx:
                complexity = AlgebraicComplexity(
                    algebraic_entropy=meta_cx.get("H_alg", 0.0),
                    commutativity_distance=meta_cx.get("D_comm", 0.0),
                    orbit_complexity=meta_cx.get("O_c", 0.0),
                    structural_interference=meta_cx.get("I_s", 0.0),
                )
                complexity_totals["algebraic_entropy"]       += complexity.algebraic_entropy
                complexity_totals["commutativity_distance"]  += complexity.commutativity_distance
                complexity_totals["orbit_complexity"]        += complexity.orbit_complexity
                complexity_totals["structural_interference"] += complexity.structural_interference
                complexity_count += 1
            else:
                complexity = None

            family_val = task.family.value if hasattr(task.family, "value") else str(task.family)
            dim_val = task.dimension.value if hasattr(task.dimension, "value") else str(task.dimension)

            result = EvalResult(
                task_id=task.task_id,
                correct=correct,
                model_response=response,
                ground_truth=task.answer,
                depth=task.depth,
                family=family_val,
                dimension=dim_val,
                complexity=complexity,
            )
            if not correct:
                result.error_category = _classify_error(result)
            results.append(result)

        total_scored = len(results)
        total = total_scored + errored  # include errored in total for honest reporting
        total_correct = sum(1 for r in results if r.correct)

        # Compositional ceiling: computed on chain families only so that
        # conceptual (always depth=1), rule, adversarial, and intermediate
        # tasks do not distort the depth-accuracy relationship.
        chain_results = [
            r for r in results
            if r.family in CHAIN_FAMILIES and r.dimension not in CHAIN_EXCLUDED_DIMENSIONS
        ]
        chain_depth_stats = _compute_stats(chain_results, lambda r: r.depth)

        ceiling_50 = self._find_ceiling(chain_depth_stats, 0.50)
        ceiling_25 = self._find_ceiling(chain_depth_stats, 0.25)

        n = complexity_count or 1
        return EvalReport(
            run_id=run_id,
            timestamp=timestamp,
            model_name=model_name,
            task_set_name=task_set.name,
            total_tasks=total,
            total_correct=total_correct,
            missing_predictions=missing,
            errored_predictions=errored,
            accuracy_overall=total_correct / total_scored if total_scored > 0 else 0.0,
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
