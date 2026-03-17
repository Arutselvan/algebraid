"""
Tests for the proof verification system (proof.py).

Covers:
  - Inter-structure task verification (DirectProduct reconstruction)
  - Rule-induction task verification (solution_trace now present)
  - _parse_structure_list with 2 and 3 components
"""

import pytest

from algebraid.generator import AlgebraidGenerator
from algebraid.proof import (
    verify_task,
    _parse_structure_list,
)


class TestParseStructureList:
    """Test that _parse_structure_list correctly builds DirectProducts."""

    def test_single_structure(self):
        s = _parse_structure_list(["Z_5"])
        assert s is not None
        assert s.name == "Z_5"

    def test_two_components(self):
        s = _parse_structure_list(["Z_3", "Z_4"])
        assert s is not None
        assert s.name == "Z_3 x Z_4"

    def test_three_components(self):
        s = _parse_structure_list(["Z_2", "Z_3", "Z_5"])
        assert s is not None
        # Left-to-right nesting: (Z_2 x Z_3) x Z_5
        assert "Z_2 x Z_3" in s.name
        assert s.name.endswith("Z_5")

    def test_unknown_returns_none(self):
        assert _parse_structure_list(["FooBar"]) is None

    def test_empty_returns_none(self):
        assert _parse_structure_list([]) is None


class TestInterStructureProof:
    """Verify that generated inter-structure tasks pass proof verification."""

    def test_inter_tasks_verified(self):
        gen = AlgebraidGenerator(seed=42)
        ts = gen.generate(
            depths=[1, 2],
            tasks_per_depth=5,
            families=["inter"],
        )
        assert len(ts.tasks) > 0
        for task in ts:
            result = verify_task(task)
            assert result.verified, (
                f"{task.task_id}: {result.error_message}"
            )
            assert result.steps_checked > 0, (
                f"{task.task_id}: inter task should have steps_checked > 0"
            )


class TestRuleInductionProof:
    """Verify that generated rule-induction tasks now have traces and pass proof."""

    def test_rule_tasks_have_trace(self):
        gen = AlgebraidGenerator(seed=42)
        ts = gen.generate(
            depths=[1, 2],
            tasks_per_depth=5,
            families=["rule"],
        )
        assert len(ts.tasks) > 0
        for task in ts:
            assert task.solution_trace is not None, (
                f"{task.task_id}: rule task should have a solution_trace"
            )
            assert len(task.solution_trace) >= 2, (
                f"{task.task_id}: trace should have start + at least 1 step"
            )

    def test_rule_tasks_verified(self):
        gen = AlgebraidGenerator(seed=42)
        ts = gen.generate(
            depths=[1, 2],
            tasks_per_depth=5,
            families=["rule"],
        )
        for task in ts:
            result = verify_task(task)
            assert result.verified, (
                f"{task.task_id}: {result.error_message}"
            )
            assert result.steps_checked > 0, (
                f"{task.task_id}: rule task should have steps_checked > 0"
            )
