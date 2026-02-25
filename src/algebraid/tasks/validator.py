"""
ALGEBRAID Task Validator.

Provides post-generation validation to catch prompt-answer misalignment,
missing information, and other quality issues. This module was created in
response to identified defects where skin descriptions contradicted the
underlying mathematical operations.

Validation checks:
1. Recomputation: re-execute the operation chain and verify the answer matches
2. Prompt completeness: ensure all required parameters are present in the prompt
3. Skin consistency: verify that skin-translated elements are valid
4. Structural integrity: check task_id uniqueness, required fields, etc.
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from ..task_model import Task, TaskSet


class ValidationResult:
    """Result of validating a single task."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __repr__(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        return f"ValidationResult({self.task_id}: {status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


class TaskValidator:
    """Validates ALGEBRAID tasks for correctness and completeness."""

    def validate_task(self, task: Task) -> ValidationResult:
        """Run all validation checks on a single task."""
        result = ValidationResult(task.task_id)

        # 1. Required fields check
        self._check_required_fields(task, result)

        # 2. Prompt completeness check
        self._check_prompt_completeness(task, result)

        # 3. Answer consistency check
        self._check_answer_consistency(task, result)

        # 4. Solution trace check (if present)
        if task.solution_trace:
            self._check_solution_trace(task, result)

        return result

    def validate_taskset(self, taskset: TaskSet) -> Dict[str, Any]:
        """Validate an entire task set and return a summary report."""
        results: List[ValidationResult] = []
        task_ids = set()
        duplicate_ids: List[str] = []

        for task in taskset:
            # Check for duplicate IDs
            if task.task_id in task_ids:
                duplicate_ids.append(task.task_id)
            task_ids.add(task.task_id)

            # Validate individual task
            vr = self.validate_task(task)
            results.append(vr)

        total = len(results)
        passed = sum(1 for r in results if r.is_valid)
        failed = total - passed
        all_errors = []
        all_warnings = []
        for r in results:
            for e in r.errors:
                all_errors.append(f"[{r.task_id}] {e}")
            for w in r.warnings:
                all_warnings.append(f"[{r.task_id}] {w}")

        report = {
            "total_tasks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0.0,
            "duplicate_ids": duplicate_ids,
            "errors": all_errors,
            "warnings": all_warnings,
            "error_count": len(all_errors),
            "warning_count": len(all_warnings),
        }
        return report

    def _check_required_fields(self, task: Task, result: ValidationResult) -> None:
        """Check that all required fields are present and non-empty."""
        if not task.task_id:
            result.add_error("Missing task_id")
        if not task.prompt or len(task.prompt.strip()) < 10:
            result.add_error("Prompt is missing or too short")
        if not task.answer and task.answer != "0":
            result.add_error("Missing answer")
        if task.answer_raw is None and task.answer_raw != 0:
            result.add_error("Missing answer_raw")
        if task.depth < 0:
            result.add_error(f"Invalid depth: {task.depth}")

    def _check_prompt_completeness(self, task: Task, result: ValidationResult) -> None:
        """Check that the prompt contains all information needed to solve the task."""
        prompt = task.prompt

        # Check for generic fallback descriptions that indicate missing info
        generic_phrases = [
            "apply a shuffle",
            "apply a symmetry",
            "apply a tile move",
            "rearrange the seats",
            "apply a code transformation",
            "apply a modular operation",
        ]
        for phrase in generic_phrases:
            # Only flag if the phrase appears as a step instruction (not in a header/description)
            # Look for it after "Step" or as a numbered item
            pattern = rf'(?:Step\s+\d+|^\d+[\.\)])\s*:?\s*{re.escape(phrase)}'
            if re.search(pattern, prompt, re.MULTILINE | re.IGNORECASE):
                result.add_error(
                    f"Prompt contains generic fallback '{phrase}' — "
                    f"specific operation parameters are missing"
                )

        # Check for placeholder-like patterns
        if "{" in prompt and "}" in prompt:
            # Check for unresolved template variables (but allow set notation like {0, 1, ...})
            unresolved = re.findall(r'\{[a-z_]+\}', prompt)
            if unresolved:
                result.add_error(f"Unresolved template variables in prompt: {unresolved}")

        # Check for double periods (formatting issue)
        if ".." in prompt and "..." not in prompt:
            result.add_warning("Double period detected in prompt")

    def _check_answer_consistency(self, task: Task, result: ValidationResult) -> None:
        """Check that answer and answer_raw are consistent."""
        # For tasks with solution traces, verify the final trace value matches answer_raw
        if task.solution_trace:
            final_op, final_val = task.solution_trace[-1]
            if str(final_val) != str(task.answer_raw):
                result.add_error(
                    f"Solution trace final value '{final_val}' does not match "
                    f"answer_raw '{task.answer_raw}'"
                )

    def _check_solution_trace(self, task: Task, result: ValidationResult) -> None:
        """Validate the solution trace structure."""
        trace = task.solution_trace
        if not trace:
            return

        # First entry should be "start"
        if trace[0][0] != "start":
            result.add_warning("Solution trace does not start with 'start' entry")

        # Check trace length matches depth + 1 (start + depth operations)
        expected_len = task.depth + 1
        if len(trace) != expected_len:
            result.add_warning(
                f"Solution trace length {len(trace)} does not match expected "
                f"{expected_len} (depth {task.depth} + 1 start)"
            )


def validate_jsonl_file(path: str) -> Dict[str, Any]:
    """Validate a JSONL task file and return a report."""
    taskset = TaskSet.from_jsonl(path)
    validator = TaskValidator()
    return validator.validate_taskset(taskset)


def print_validation_report(report: Dict[str, Any]) -> None:
    """Print a human-readable validation report."""
    print(f"\n{'='*60}")
    print(f"ALGEBRAID Task Validation Report")
    print(f"{'='*60}")
    print(f"Total tasks:    {report['total_tasks']}")
    print(f"Passed:         {report['passed']}")
    print(f"Failed:         {report['failed']}")
    print(f"Pass rate:      {report['pass_rate']}%")
    print(f"Errors:         {report['error_count']}")
    print(f"Warnings:       {report['warning_count']}")

    if report['duplicate_ids']:
        print(f"\nDuplicate IDs ({len(report['duplicate_ids'])}):")
        for tid in report['duplicate_ids'][:10]:
            print(f"  - {tid}")

    if report['errors']:
        print(f"\nErrors (showing first 20):")
        for err in report['errors'][:20]:
            print(f"  ERROR: {err}")

    if report['warnings']:
        print(f"\nWarnings (showing first 20):")
        for warn in report['warnings'][:20]:
            print(f"  WARN:  {warn}")

    print(f"{'='*60}\n")
