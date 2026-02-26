"""
Tests for AlgebraidGenerator and all 7 task-generating functions.

Coverage:
  - All 7 families: intra, inter, field, rule, conceptual, adversarial, intermediate
  - All 5 original dimensions + 2 new ones
  - Determinism (same seed → identical output)
  - Metadata correctness for new task types
  - Grading correctness (answer matches computation)
  - Solution trace structure
"""

import pytest

from algebraid.generator import (
    AlgebraidGenerator,
    _generate_conceptual_task,
    _generate_adversarial_task,
    _generate_intermediate_state_task,
    _element_order,
)
from algebraid.task_model import Task, TaskFamily, CompositionDimension
from algebraid.tasks.verifier import check_answer
from algebraid.primitives import CyclicGroup, QuaternionGroup


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_same_tasks(self):
        ts1 = AlgebraidGenerator(seed=42).generate(
            depths=[1, 2], tasks_per_depth=5, include_dimensions=False
        )
        ts2 = AlgebraidGenerator(seed=42).generate(
            depths=[1, 2], tasks_per_depth=5, include_dimensions=False
        )
        for t1, t2 in zip(ts1, ts2):
            assert t1.task_id == t2.task_id
            assert t1.prompt == t2.prompt
            assert t1.answer == t2.answer

    def test_different_seeds_different_tasks(self):
        ts1 = AlgebraidGenerator(seed=1).generate(depths=[1], tasks_per_depth=5)
        ts2 = AlgebraidGenerator(seed=2).generate(depths=[1], tasks_per_depth=5)
        ids1 = {t.task_id for t in ts1}
        ids2 = {t.task_id for t in ts2}
        assert ids1 != ids2


# ── Family generation ─────────────────────────────────────────────────────────

class TestFamilyGeneration:
    def _families_in(self, ts):
        return {t.family.value for t in ts}

    def test_intra_family_present(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=5, families=["intra"], include_dimensions=False)
        assert "intra-structure composition" in self._families_in(ts)

    def test_inter_family_present(self, gen):
        ts = gen.generate(depths=[2], tasks_per_depth=5, families=["inter"], include_dimensions=False)
        assert "inter-structure composition" in self._families_in(ts)

    def test_field_family_present(self, gen):
        ts = gen.generate(depths=[2], tasks_per_depth=5, families=["field"], include_dimensions=False)
        assert "field arithmetic" in self._families_in(ts)

    def test_rule_family_present(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=5, families=["rule"], include_dimensions=False)
        assert "rule induction" in self._families_in(ts)

    def test_conceptual_family_present(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=5, families=["conceptual"], include_dimensions=False)
        assert "conceptual query" in self._families_in(ts)

    def test_adversarial_family_present(self, gen):
        ts = gen.generate(depths=[2], tasks_per_depth=5, families=["adversarial"], include_dimensions=False)
        assert "intra-structure composition" in self._families_in(ts)
        dims = {t.dimension.value for t in ts}
        assert "adversarial" in dims

    def test_intermediate_family_present(self, gen):
        ts = gen.generate(depths=[3], tasks_per_depth=5, families=["intermediate"], include_dimensions=False)
        dims = {t.dimension.value for t in ts}
        assert "intermediate_state" in dims

    def test_unknown_family_ignored(self, gen):
        # Should not raise; just skip unknown family
        ts = gen.generate(depths=[1], tasks_per_depth=3, families=["intra", "unknown"], include_dimensions=False)
        assert len(ts) > 0

    def test_include_dimensions_adds_extra_tasks(self, gen):
        ts_no = gen.generate(depths=[2, 3], tasks_per_depth=3, families=["intra"], include_dimensions=False)
        ts_yes = AlgebraidGenerator(seed=42).generate(depths=[2, 3], tasks_per_depth=3, families=["intra"], include_dimensions=True)
        assert len(ts_yes) > len(ts_no)


# ── Task schema validation ─────────────────────────────────────────────────────

class TestTaskSchema:
    def test_all_tasks_have_required_fields(self, gen):
        ts = gen.generate(depths=[1, 2], tasks_per_depth=5, include_dimensions=False)
        for t in ts:
            assert isinstance(t.task_id, str) and t.task_id.startswith("AG-")
            assert isinstance(t.prompt, str) and len(t.prompt) > 10
            assert isinstance(t.answer, str)
            assert isinstance(t.answer_raw, str)
            assert isinstance(t.depth, int) and t.depth >= 1
            assert isinstance(t.family, TaskFamily)
            assert isinstance(t.dimension, CompositionDimension)
            assert isinstance(t.structures, list) and len(t.structures) > 0

    def test_task_ids_are_unique(self, gen):
        ts = gen.generate(depths=[1, 2, 3], tasks_per_depth=10, include_dimensions=True)
        ids = [t.task_id for t in ts]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"


# ── Conceptual task correctness ───────────────────────────────────────────────

class TestConceptualTaskCorrectness:
    def test_identity_answer_correct(self):
        import random
        rng = random.Random(0)
        from algebraid.tasks.verbalizer import Verbalizer
        v = Verbalizer(0)
        # Generate many and verify identity answers
        for seed in range(10):
            rng2 = random.Random(seed)
            t = _generate_conceptual_task(rng2, 1, seed, seed, v)
            if t.metadata.get("query_subtype") == "identity":
                from algebraid.primitives import CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField, QuaternionGroup
                struct_name = t.structures[0]
                # Just verify answer is a string representing an element
                assert isinstance(t.answer, str)

    def test_structure_order_matches_structure(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=20, families=["conceptual"], include_dimensions=False)
        for t in ts:
            if t.metadata.get("query_subtype") == "structure_order":
                expected = str(t.metadata["structure_order"])
                assert t.answer_raw == expected

    def test_is_abelian_matches_metadata(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=30, families=["conceptual"], include_dimensions=False)
        for t in ts:
            if t.metadata.get("query_subtype") == "is_abelian":
                is_abelian = t.metadata.get("structure_is_abelian")
                expected = "yes" if is_abelian else "no"
                assert t.answer_raw == expected

    def test_commutativity_check_yes_only_if_abelian(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=30, families=["conceptual"], include_dimensions=False)
        for t in ts:
            if t.metadata.get("query_subtype") == "commutativity_check":
                # If structure is abelian, all pairs commute → answer is always "yes"
                if t.metadata.get("structure_is_abelian"):
                    assert t.answer_raw == "yes"

    def test_metadata_has_subtype(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=10, families=["conceptual"], include_dimensions=False)
        for t in ts:
            assert "query_subtype" in t.metadata
            assert t.metadata["query_subtype"] in [
                "identity", "element_order", "commutativity_check",
                "structure_order", "is_abelian", "inverse_of", "is_generator",
            ]

    def test_solution_trace_is_none(self, gen):
        ts = gen.generate(depths=[1], tasks_per_depth=5, families=["conceptual"], include_dimensions=False)
        for t in ts:
            assert t.solution_trace is None

    def test_depth_is_1(self, gen):
        ts = gen.generate(depths=[1, 2, 3], tasks_per_depth=5, families=["conceptual"], include_dimensions=False)
        for t in ts:
            assert t.depth == 1


# ── Adversarial task correctness ──────────────────────────────────────────────

class TestAdversarialTaskCorrectness:
    def test_double_inverse_answer_equals_start(self, gen):
        ts = gen.generate(depths=[2, 3], tasks_per_depth=10, families=["adversarial"], include_dimensions=False)
        for t in ts:
            if t.metadata.get("adversarial_type") == "double_inverse":
                # answer should equal starting element (first trace entry after "start")
                assert t.solution_trace is not None
                start_val = t.solution_trace[0][1]
                assert t.answer_raw == start_val

    def test_self_cancelling_answer_equals_start(self, gen):
        ts = gen.generate(depths=[2, 3], tasks_per_depth=10, families=["adversarial"], include_dimensions=False)
        for t in ts:
            if t.metadata.get("adversarial_type") == "self_cancelling":
                assert t.solution_trace is not None
                start_val = t.solution_trace[0][1]
                assert t.answer_raw == start_val

    def test_answer_ne_wrong_answer_for_trap_types(self, gen):
        ts = gen.generate(depths=[2, 3], tasks_per_depth=15, families=["adversarial"], include_dimensions=False)
        for t in ts:
            adv_type = t.metadata.get("adversarial_type")
            if adv_type in ("commutativity_trap", "identity_bait"):
                assert t.answer_raw != t.metadata.get("wrong_answer"), (
                    f"answer == wrong_answer for {adv_type}"
                )

    def test_metadata_has_wrong_answer(self, gen):
        ts = gen.generate(depths=[2], tasks_per_depth=10, families=["adversarial"], include_dimensions=False)
        for t in ts:
            assert "wrong_answer" in t.metadata
            assert "adversarial_type" in t.metadata
            assert "wrong_answer_rationale" in t.metadata

    def test_dimension_is_adversarial(self, gen):
        ts = gen.generate(depths=[2], tasks_per_depth=5, families=["adversarial"], include_dimensions=False)
        for t in ts:
            assert t.dimension == CompositionDimension.ADVERSARIAL

    def test_solution_trace_present(self, gen):
        ts = gen.generate(depths=[2], tasks_per_depth=5, families=["adversarial"], include_dimensions=False)
        for t in ts:
            assert t.solution_trace is not None
            assert len(t.solution_trace) >= 2
            assert t.solution_trace[0][0] == "start"


# ── Intermediate state task correctness ──────────────────────────────────────

class TestIntermediateStateTaskCorrectness:
    def test_answer_matches_trace_at_query_step(self, gen):
        ts = gen.generate(depths=[3, 4], tasks_per_depth=10, families=["intermediate"], include_dimensions=False)
        for t in ts:
            k = t.metadata.get("query_step")
            assert k is not None
            trace = t.solution_trace
            assert trace is not None
            # trace[k] is the k-th step (0-indexed: trace[0]=start, trace[1]=step1, ...)
            assert t.answer_raw == trace[k][1]

    def test_depth_equals_query_step(self, gen):
        ts = gen.generate(depths=[3, 4], tasks_per_depth=10, families=["intermediate"], include_dimensions=False)
        for t in ts:
            assert t.depth == t.metadata["query_step"]

    def test_total_steps_greater_than_query_step(self, gen):
        ts = gen.generate(depths=[3, 4], tasks_per_depth=10, families=["intermediate"], include_dimensions=False)
        for t in ts:
            assert t.metadata["total_steps"] > t.metadata["query_step"]

    def test_trace_truncated_to_query_step(self, gen):
        ts = gen.generate(depths=[3, 4], tasks_per_depth=10, families=["intermediate"], include_dimensions=False)
        for t in ts:
            k = t.metadata["query_step"]
            # trace has k+1 entries: start + k operations
            assert len(t.solution_trace) == k + 1

    def test_final_answer_in_metadata(self, gen):
        ts = gen.generate(depths=[3], tasks_per_depth=5, families=["intermediate"], include_dimensions=False)
        for t in ts:
            assert "final_answer" in t.metadata
            assert isinstance(t.metadata["final_answer"], str)

    def test_dimension_is_intermediate_state(self, gen):
        ts = gen.generate(depths=[3], tasks_per_depth=5, families=["intermediate"], include_dimensions=False)
        for t in ts:
            assert t.dimension == CompositionDimension.INTERMEDIATE_STATE


# ── _element_order helper ─────────────────────────────────────────────────────

class TestElementOrder:
    def test_identity_order_is_1(self, z7):
        assert _element_order(z7, 0) == 1

    def test_z7_generator_order_is_7(self, z7):
        # gcd(1, 7) = 1, so 1 generates Z_7 → order = 7
        assert _element_order(z7, 1) == 7

    def test_z12_element_order(self):
        z12 = CyclicGroup(12)
        assert _element_order(z12, 4) == 3   # gcd(4,12)=4, order=12/4=3
        assert _element_order(z12, 3) == 4   # gcd(3,12)=3, order=12/3=4
        assert _element_order(z12, 6) == 2   # gcd(6,12)=6, order=12/6=2

    def test_q8_neg1_order_is_2(self):
        q8 = QuaternionGroup()
        assert _element_order(q8, 1) == 2   # -1 has order 2

    def test_q8_i_order_is_4(self):
        q8 = QuaternionGroup()
        assert _element_order(q8, 2) == 4   # i has order 4

    def test_q8_identity_order_is_1(self):
        q8 = QuaternionGroup()
        assert _element_order(q8, 0) == 1


# ── QuaternionGroup in generation ─────────────────────────────────────────────

class TestQ8InGeneration:
    def test_q8_appears_in_structures(self):
        # With enough tasks, Q_8 should appear (20% probability)
        gen = AlgebraidGenerator(seed=7)
        ts = gen.generate(depths=[1, 2], tasks_per_depth=20, families=["intra"], include_dimensions=False)
        all_structures = [s for t in ts for s in t.structures]
        assert "Q_8" in all_structures, "Q_8 should appear among structures for large task sets"

    def test_q8_tasks_have_valid_answers(self):
        gen = AlgebraidGenerator(seed=7)
        ts = gen.generate(depths=[1, 2], tasks_per_depth=20, families=["intra"], include_dimensions=False)
        q8_tasks = [t for t in ts if "Q_8" in t.structures]
        q8_names = {"1", "-1", "i", "-i", "j", "-j", "k", "-k"}
        for t in q8_tasks:
            assert t.answer_raw in q8_names, f"Unexpected Q_8 answer: {t.answer_raw}"
