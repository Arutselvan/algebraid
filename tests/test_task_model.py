"""
Tests for task_model.py: Task, TaskSet, TaskFamily, CompositionDimension.
"""

import json
import tempfile
import os
import pytest

from algebraid.task_model import Task, TaskSet, TaskFamily, CompositionDimension


# ── Enum values ───────────────────────────────────────────────────────────────

class TestTaskFamilyEnum:
    def test_original_families(self):
        assert TaskFamily.INTRA_STRUCTURE.value == "intra-structure composition"
        assert TaskFamily.INTER_STRUCTURE.value == "inter-structure composition"
        assert TaskFamily.FIELD_ARITHMETIC.value == "field arithmetic"
        assert TaskFamily.RULE_INDUCTION.value == "rule induction"

    def test_new_family(self):
        assert TaskFamily.CONCEPTUAL_QUERY.value == "conceptual query"

    def test_roundtrip(self):
        for fam in TaskFamily:
            assert TaskFamily(fam.value) == fam


class TestCompositionDimensionEnum:
    def test_original_dimensions(self):
        assert CompositionDimension.GENERAL.value == "general"
        assert CompositionDimension.SYSTEMATICITY.value == "systematicity"
        assert CompositionDimension.PRODUCTIVITY.value == "productivity"
        assert CompositionDimension.SUBSTITUTIVITY.value == "substitutivity"
        assert CompositionDimension.OVERGENERALIZATION.value == "overgeneralization"

    def test_new_dimensions(self):
        assert CompositionDimension.ADVERSARIAL.value == "adversarial"
        assert CompositionDimension.INTERMEDIATE_STATE.value == "intermediate_state"

    def test_roundtrip(self):
        for dim in CompositionDimension:
            assert CompositionDimension(dim.value) == dim


# ── Task dataclass ────────────────────────────────────────────────────────────

class TestTask:
    def test_creation(self, simple_task):
        assert simple_task.task_id == "AG-test000001"
        assert simple_task.answer == "3"
        assert simple_task.depth == 1

    def test_to_dict_has_all_fields(self, simple_task):
        d = simple_task.to_dict()
        required = ["task_id", "prompt", "answer", "answer_raw", "depth",
                    "family", "dimension", "structures", "metadata", "solution_trace"]
        for key in required:
            assert key in d, f"Missing field: {key}"

    def test_to_dict_family_is_string(self, simple_task):
        d = simple_task.to_dict()
        assert isinstance(d["family"], str)

    def test_to_dict_dimension_is_string(self, simple_task):
        d = simple_task.to_dict()
        assert isinstance(d["dimension"], str)

    def test_to_dict_null_metadata(self, simple_task):
        # Empty dict should serialize as null per to_dict logic
        simple_task.metadata = {}
        d = simple_task.to_dict()
        assert d["metadata"] is None

    def test_to_dict_nonempty_metadata(self, conceptual_task):
        d = conceptual_task.to_dict()
        assert d["metadata"] is not None
        assert d["metadata"]["query_subtype"] == "identity"

    def test_solution_trace_null_when_none(self, simple_task):
        d = simple_task.to_dict()
        assert d["solution_trace"] is None

    def test_conceptual_task_family(self, conceptual_task):
        assert conceptual_task.family == TaskFamily.CONCEPTUAL_QUERY

    def test_adversarial_task_dimension(self, adversarial_task):
        assert adversarial_task.dimension == CompositionDimension.ADVERSARIAL

    def test_intermediate_task_metadata(self, intermediate_task):
        assert intermediate_task.metadata["query_step"] == 1
        assert intermediate_task.metadata["total_steps"] == 3


# ── TaskSet ───────────────────────────────────────────────────────────────────

class TestTaskSet:
    def test_len(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=5)
        assert len(ts) == 5 * 7  # 7 families

    def test_getitem_int(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=3)
        t = ts[0]
        assert isinstance(t, Task)

    def test_getitem_task_id(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=3)
        first = ts[0]
        assert ts[first.task_id] == first

    def test_iteration(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=3)
        count = sum(1 for _ in ts)
        assert count == len(ts)

    def test_summary_contains_family_names(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=3)
        summary = ts.summary()
        assert "intra-structure composition" in summary

    def test_jsonl_roundtrip(self, gen):
        ts = gen.generate(depths=[1, 2], tasks_per_depth=5)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            ts.to_jsonl(path)
            ts2 = TaskSet.from_jsonl(path)
            assert len(ts2) == len(ts)
            for t1, t2 in zip(ts, ts2):
                assert t1.task_id == t2.task_id
                assert t1.answer == t2.answer
                assert t1.family == t2.family
                assert t1.dimension == t2.dimension
        finally:
            os.unlink(path)

    def test_jsonl_new_families_roundtrip(self, gen):
        # Conceptual and adversarial families should survive JSON roundtrip
        ts = gen.generate(
            depths=[1], tasks_per_depth=3,
            families=["conceptual", "adversarial"],
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
        try:
            ts.to_jsonl(path)
            ts2 = TaskSet.from_jsonl(path)
            families = {t.family for t in ts2}
            assert TaskFamily.CONCEPTUAL_QUERY in families
        finally:
            os.unlink(path)

    def test_empty_metadata_roundtrip_is_dict_not_none(self):
        """Empty metadata must reload as {} not None (Bug 2 regression)."""
        task = Task(
            task_id="AG-t1",
            prompt="p",
            answer="0",
            answer_raw="0",
            depth=1,
            family=TaskFamily.INTRA_STRUCTURE,
            metadata={},  # empty -> serialised as null
        )
        d = task.to_dict()
        assert d["metadata"] is None  # serializer converts {} to null

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            path = f.name
            f.write(json.dumps(d) + "\n")
        try:
            ts = TaskSet.from_jsonl(path)
            reloaded = ts[0]
            # Must be a dict so callers can safely call .get() on it
            assert isinstance(reloaded.metadata, dict)
            assert reloaded.metadata == {}
        finally:
            os.unlink(path)
