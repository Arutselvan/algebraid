"""
Task validation for generated benchmark instances.

Provides structural and semantic checks to ensure that every task in a
generated set is well-formed, self-consistent, and solvable from its
prompt alone.  Validation is organized into four independent check
categories:

    1. Schema - required fields, types, and value ranges.
    2. Prompt - completeness, readability, and absence of template
       artifacts.
    3. Answer - consistency between the answer, raw answer, and
       solution trace.
    4. Trace - structural soundness of the solution trace when present.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List

from ..task_model import Task, TaskSet, CompositionDimension


# -- Result container --------------------------------------------------------

class ValidationResult:
    """Outcome of validating a single task."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def __repr__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationResult({self.task_id}: {tag}, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings)"
        )


# -- Core validator ----------------------------------------------------------

class TaskValidator:
    """Run validation checks on individual tasks or entire task sets."""

    def validate(self, task: Task) -> ValidationResult:
        """Validate a single task across all check categories."""
        result = ValidationResult(task.task_id)
        self._check_schema(task, result)
        self._check_prompt(task, result)
        self._check_answer(task, result)
        if task.solution_trace:
            self._check_trace(task, result)
        return result

    def validate_set(self, task_set: TaskSet) -> Dict[str, Any]:
        """Validate every task in *task_set* and return a summary report."""
        results: List[ValidationResult] = []
        seen_ids: set = set()
        duplicate_ids: List[str] = []

        for task in task_set:
            if task.task_id in seen_ids:
                duplicate_ids.append(task.task_id)
            seen_ids.add(task.task_id)
            results.append(self.validate(task))

        passed = sum(1 for r in results if r.passed)
        all_errors = [f"[{r.task_id}] {e}" for r in results for e in r.errors]
        all_warnings = [f"[{r.task_id}] {w}" for r in results for w in r.warnings]

        return {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results) * 100, 1) if results else 0.0,
            "duplicate_ids": duplicate_ids,
            "errors": all_errors,
            "warnings": all_warnings,
        }

    # -- 1. Schema checks ---------------------------------------------------

    def _check_schema(self, task: Task, r: ValidationResult) -> None:
        if not task.task_id:
            r.error("Missing task_id.")
        if not task.prompt or len(task.prompt.strip()) < 10:
            r.error("Prompt is missing or too short (< 10 chars).")
        if not task.answer and task.answer != "0":
            r.error("Missing answer.")
        if task.answer_raw is None and task.answer_raw != 0:
            r.error("Missing answer_raw.")
        if task.depth < 0:
            r.error(f"Invalid depth: {task.depth}.")

    # -- 2. Prompt checks ---------------------------------------------------

    # Phrases that indicate an operation handler fell through to a generic
    # default instead of producing a specific, solvable instruction.
    _UNDERSPECIFIED_PATTERNS: List[str] = [
        r"(?:step\s+\d+|^\s*\d+[\.\)])\s*:?\s*apply a shuffle",
        r"(?:step\s+\d+|^\s*\d+[\.\)])\s*:?\s*apply a symmetry",
        r"(?:step\s+\d+|^\s*\d+[\.\)])\s*:?\s*apply a tile move",
        r"(?:step\s+\d+|^\s*\d+[\.\)])\s*:?\s*rearrange the seats",
        r"(?:step\s+\d+|^\s*\d+[\.\)])\s*:?\s*apply a code transformation",
        r"(?:step\s+\d+|^\s*\d+[\.\)])\s*:?\s*apply a modular operation",
    ]

    def _check_prompt(self, task: Task, r: ValidationResult) -> None:
        prompt = task.prompt

        # Detect underspecified operation instructions.
        for pattern in self._UNDERSPECIFIED_PATTERNS:
            if re.search(pattern, prompt, re.MULTILINE | re.IGNORECASE):
                r.error(
                    "Prompt contains an underspecified operation instruction "
                    "that does not provide enough information to solve the task."
                )
                break  # one error per task is sufficient

        # Detect unresolved template variables (e.g. {op_name}).
        unresolved = re.findall(r"\{[a-z_]+\}", prompt)
        if unresolved:
            r.error(f"Unresolved template variables: {unresolved}.")

        # Formatting: double periods (but not ellipsis).
        if ".." in prompt and "..." not in prompt:
            r.warn("Double period detected in prompt text.")

    # -- 3. Answer checks ---------------------------------------------------

    def _check_answer(self, task: Task, r: ValidationResult) -> None:
        if not task.solution_trace:
            return
        _, final_val = task.solution_trace[-1]
        if str(final_val) != str(task.answer_raw):
            r.error(
                f"Trace final value '{final_val}' does not match "
                f"answer_raw '{task.answer_raw}'."
            )

    # -- 4. Trace checks ----------------------------------------------------

    def _check_trace(self, task: Task, r: ValidationResult) -> None:
        trace = task.solution_trace
        if trace[0][0] != "start":
            r.warn("Solution trace does not begin with a 'start' entry.")
        expected_length = task.depth + 1
        if len(trace) != expected_length:
            # Intermediate-state tasks store only the prefix of the chain up to
            # the queried step, so a shorter trace is by design - not a defect.
            if task.dimension == CompositionDimension.INTERMEDIATE_STATE:
                return
            r.warn(
                f"Trace length ({len(trace)}) differs from expected "
                f"({expected_length} = depth {task.depth} + 1)."
            )


# -- Convenience helpers -----------------------------------------------------

def validate_file(path: str) -> Dict[str, Any]:
    """Load a JSONL task file and return a validation report."""
    return TaskValidator().validate_set(TaskSet.from_jsonl(path))


def print_report(report: Dict[str, Any]) -> None:
    """Print a human-readable validation summary to stdout."""
    sep = "=" * 56
    print(f"\n{sep}")
    print("  ALGEBRAID Validation Report")
    print(sep)
    print(f"  Total tasks : {report['total']}")
    print(f"  Passed      : {report['passed']}")
    print(f"  Failed      : {report['failed']}")
    print(f"  Pass rate   : {report['pass_rate']}%")

    if report["duplicate_ids"]:
        print(f"\n  Duplicate IDs ({len(report['duplicate_ids'])}):")
        for tid in report["duplicate_ids"][:10]:
            print(f"    - {tid}")

    if report["errors"]:
        print(f"\n  Errors (first 20):")
        for err in report["errors"][:20]:
            print(f"    ERROR  {err}")

    if report["warnings"]:
        print(f"\n  Warnings (first 20):")
        for w in report["warnings"][:20]:
            print(f"    WARN   {w}")

    print(sep)
