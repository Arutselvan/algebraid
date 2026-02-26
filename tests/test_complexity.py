"""
Tests for algebraic complexity metrics:
  - compute_algebraic_entropy
  - compute_commutativity_distance
  - compute_orbit_complexity
  - compute_structural_interference
  - compute_complexity (combined)
  - compute_conceptual_depth  (new)
  - compute_adversarial_strength  (new)
"""

import pytest

from algebraid.task_model import Task, TaskFamily, CompositionDimension
from algebraid.complexity import (
    compute_algebraic_entropy,
    compute_commutativity_distance,
    compute_orbit_complexity,
    compute_structural_interference,
    compute_complexity,
    compute_conceptual_depth,
    compute_adversarial_strength,
    _parse_order,
    AlgebraicComplexity,
)


# ── _parse_order ──────────────────────────────────────────────────────────────

class TestParseOrder:
    def test_cyclic(self):
        assert _parse_order("Z_7") == 7
        assert _parse_order("Z_12") == 12

    def test_symmetric(self):
        assert _parse_order("S_3") == 6   # 3!
        assert _parse_order("S_4") == 24

    def test_dihedral(self):
        assert _parse_order("D_4") == 8   # 2*4
        assert _parse_order("D_5") == 10

    def test_finite_field(self):
        assert _parse_order("GF(7)") == 7
        assert _parse_order("GF(11)") == 11

    def test_relabeled(self):
        assert _parse_order("Relabeled Z_9") == 9

    def test_quaternion_group(self):
        assert _parse_order("Q_8") == 8

    def test_unknown_returns_zero(self):
        assert _parse_order("Unknown_Group") == 0


# ── compute_algebraic_entropy ─────────────────────────────────────────────────

class TestAlgebraicEntropy:
    def _task(self, structures, family, depth):
        return Task(
            task_id="t", prompt="p", answer="a", answer_raw="a",
            depth=depth, family=family, dimension=CompositionDimension.GENERAL,
            structures=structures, metadata={}, solution_trace=None,
        )

    def test_intra_scales_with_depth(self):
        t1 = self._task(["Z_8"], TaskFamily.INTRA_STRUCTURE, 1)
        t2 = self._task(["Z_8"], TaskFamily.INTRA_STRUCTURE, 2)
        assert compute_algebraic_entropy(t2) == pytest.approx(
            compute_algebraic_entropy(t1) * 2, rel=1e-6
        )

    def test_larger_structure_higher_entropy(self):
        t_small = self._task(["Z_4"], TaskFamily.INTRA_STRUCTURE, 2)
        t_large = self._task(["Z_16"], TaskFamily.INTRA_STRUCTURE, 2)
        assert compute_algebraic_entropy(t_large) > compute_algebraic_entropy(t_small)

    def test_inter_structure_nonzero(self):
        t = self._task(["Z_3", "Z_4"], TaskFamily.INTER_STRUCTURE, 2)
        assert compute_algebraic_entropy(t) > 0

    def test_empty_structures_zero(self):
        t = self._task([], TaskFamily.INTRA_STRUCTURE, 1)
        assert compute_algebraic_entropy(t) == 0.0

    def test_quaternion_group_order_8(self):
        import math
        t = self._task(["Q_8"], TaskFamily.INTRA_STRUCTURE, 2)
        expected = math.log2(8) * 2
        assert compute_algebraic_entropy(t) == pytest.approx(expected)


# ── compute_commutativity_distance ────────────────────────────────────────────

class TestCommutativityDistance:
    def _task(self, structures, family, depth=2):
        return Task(
            task_id="t", prompt="p", answer="a", answer_raw="a",
            depth=depth, family=family, dimension=CompositionDimension.GENERAL,
            structures=structures, metadata={}, solution_trace=None,
        )

    def test_cyclic_zero(self):
        t = self._task(["Z_7"], TaskFamily.INTRA_STRUCTURE, 2)
        assert compute_commutativity_distance(t) == 0.0

    def test_finite_field_zero(self):
        t = self._task(["GF(5)"], TaskFamily.INTRA_STRUCTURE, 2)
        assert compute_commutativity_distance(t) == 0.0

    def test_symmetric_nonzero(self):
        t = self._task(["S_3"], TaskFamily.INTRA_STRUCTURE, 3)
        assert compute_commutativity_distance(t) == 1.0

    def test_dihedral_nonzero(self):
        t = self._task(["D_4"], TaskFamily.INTRA_STRUCTURE, 3)
        assert compute_commutativity_distance(t) == 1.0

    def test_q8_nonzero(self):
        t = self._task(["Q_8"], TaskFamily.INTRA_STRUCTURE, 3)
        assert compute_commutativity_distance(t) == 1.0

    def test_inter_mixed(self):
        # Z_3 (commutative) + S_3 (non-commutative) → 0.5
        t = self._task(["Z_3", "S_3"], TaskFamily.INTER_STRUCTURE)
        d = compute_commutativity_distance(t)
        assert 0.0 < d < 1.0


# ── compute_orbit_complexity ──────────────────────────────────────────────────

class TestOrbitComplexity:
    def test_no_trace_returns_zero(self, simple_task):
        assert compute_orbit_complexity(simple_task) == 0.0

    def test_single_trace_entry(self):
        t = Task(
            task_id="t", prompt="p", answer="3", answer_raw="3",
            depth=1, family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            solution_trace=[("start", "3"), ("inverse", "4")],
        )
        oc = compute_orbit_complexity(t)
        assert 0.0 <= oc <= 1.0

    def test_bounded_between_zero_and_one(self, adversarial_task):
        oc = compute_orbit_complexity(adversarial_task)
        assert 0.0 <= oc <= 1.0

    def test_more_distinct_values_higher_complexity(self):
        # Trace with 4 distinct values vs 2 distinct values
        t_few = Task(
            task_id="t1", prompt="p", answer="3", answer_raw="3",
            depth=3, family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            solution_trace=[("start","3"),("op","3"),("op","3"),("op","3")],
        )
        t_many = Task(
            task_id="t2", prompt="p", answer="6", answer_raw="6",
            depth=3, family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            solution_trace=[("start","1"),("op","2"),("op","4"),("op","6")],
        )
        assert compute_orbit_complexity(t_many) >= compute_orbit_complexity(t_few)


# ── compute_structural_interference ──────────────────────────────────────────

class TestStructuralInterference:
    def _task(self, structures, family=TaskFamily.INTER_STRUCTURE):
        return Task(
            task_id="t", prompt="p", answer="a", answer_raw="a",
            depth=1, family=family, dimension=CompositionDimension.GENERAL,
            structures=structures,
        )

    def test_intra_structure_zero(self):
        t = self._task(["Z_7"], TaskFamily.INTRA_STRUCTURE)
        assert compute_structural_interference(t) == 0.0

    def test_coprime_orders_zero(self):
        # Z_3 × Z_4: gcd(3,4) = 1 → 0 interference
        t = self._task(["Z_3", "Z_4"])
        assert compute_structural_interference(t) == 0.0

    def test_shared_divisors_nonzero(self):
        # Z_4 × Z_4: gcd(4,4) = 4 > 1
        t = self._task(["Z_4", "Z_4"])
        assert compute_structural_interference(t) > 0.0

    def test_bounded(self):
        t = self._task(["Z_6", "Z_4"])
        assert 0.0 <= compute_structural_interference(t) <= 1.0


# ── compute_complexity (combined) ─────────────────────────────────────────────

class TestComputeComplexity:
    def test_returns_algebraic_complexity(self, simple_task):
        c = compute_complexity(simple_task)
        assert isinstance(c, AlgebraicComplexity)

    def test_all_fields_present(self, simple_task):
        c = compute_complexity(simple_task)
        assert hasattr(c, "algebraic_entropy")
        assert hasattr(c, "commutativity_distance")
        assert hasattr(c, "orbit_complexity")
        assert hasattr(c, "structural_interference")

    def test_composite_nonnegative(self, simple_task):
        c = compute_complexity(simple_task)
        assert c.composite() >= 0.0


# ── compute_conceptual_depth ──────────────────────────────────────────────────

class TestConceptualDepth:
    def test_non_conceptual_returns_zero(self, simple_task):
        assert compute_conceptual_depth(simple_task) == 0.0

    def test_identity_is_lowest(self, conceptual_task):
        assert compute_conceptual_depth(conceptual_task) == pytest.approx(0.1)

    def test_element_order_is_highest(self):
        t = Task(
            task_id="t", prompt="p", answer="4", answer_raw="4",
            depth=1, family=TaskFamily.CONCEPTUAL_QUERY,
            dimension=CompositionDimension.GENERAL,
            structures=["Q_8"],
            metadata={"query_subtype": "element_order"},
        )
        assert compute_conceptual_depth(t) == pytest.approx(0.7)

    def test_all_subtypes_bounded(self):
        subtypes = [
            "identity", "structure_order", "inverse_of", "is_abelian",
            "commutativity_check", "is_generator", "element_order",
        ]
        for st in subtypes:
            t = Task(
                task_id="t", prompt="p", answer="x", answer_raw="x",
                depth=1, family=TaskFamily.CONCEPTUAL_QUERY,
                dimension=CompositionDimension.GENERAL,
                structures=["Z_7"],
                metadata={"query_subtype": st},
            )
            d = compute_conceptual_depth(t)
            assert 0.0 < d <= 1.0, f"Depth out of range for subtype {st}: {d}"

    def test_ranking_identity_lt_element_order(self):
        t_id = Task(
            task_id="t1", prompt="p", answer="0", answer_raw="0",
            depth=1, family=TaskFamily.CONCEPTUAL_QUERY,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            metadata={"query_subtype": "identity"},
        )
        t_ord = Task(
            task_id="t2", prompt="p", answer="7", answer_raw="7",
            depth=1, family=TaskFamily.CONCEPTUAL_QUERY,
            dimension=CompositionDimension.GENERAL,
            structures=["Z_7"],
            metadata={"query_subtype": "element_order"},
        )
        assert compute_conceptual_depth(t_id) < compute_conceptual_depth(t_ord)


# ── compute_adversarial_strength ──────────────────────────────────────────────

class TestAdversarialStrength:
    def test_non_adversarial_zero(self, simple_task):
        assert compute_adversarial_strength(simple_task) == 0.0

    def test_double_inverse(self, adversarial_task):
        assert compute_adversarial_strength(adversarial_task) == pytest.approx(0.5)

    def test_commutativity_trap_highest(self):
        t = Task(
            task_id="t", prompt="p", answer="k", answer_raw="k",
            depth=1, family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.ADVERSARIAL,
            structures=["Q_8"],
            metadata={"adversarial_type": "commutativity_trap"},
        )
        assert compute_adversarial_strength(t) == pytest.approx(0.8)

    def test_self_cancelling_lowest(self):
        t = Task(
            task_id="t", prompt="p", answer="3", answer_raw="3",
            depth=2, family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.ADVERSARIAL,
            structures=["Z_7"],
            metadata={"adversarial_type": "self_cancelling"},
        )
        assert compute_adversarial_strength(t) == pytest.approx(0.3)

    def test_ranking_self_cancelling_lt_commutativity_trap(self):
        t_sc = Task(
            task_id="t1", prompt="p", answer="3", answer_raw="3",
            depth=2, family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.ADVERSARIAL,
            structures=["Z_7"],
            metadata={"adversarial_type": "self_cancelling"},
        )
        t_ct = Task(
            task_id="t2", prompt="p", answer="k", answer_raw="k",
            depth=1, family=TaskFamily.INTRA_STRUCTURE,
            dimension=CompositionDimension.ADVERSARIAL,
            structures=["Q_8"],
            metadata={"adversarial_type": "commutativity_trap"},
        )
        assert compute_adversarial_strength(t_sc) < compute_adversarial_strength(t_ct)

    def test_all_types_bounded(self):
        adv_types = ["self_cancelling", "double_inverse", "identity_bait", "commutativity_trap"]
        for adv in adv_types:
            t = Task(
                task_id="t", prompt="p", answer="x", answer_raw="x",
                depth=1, family=TaskFamily.INTRA_STRUCTURE,
                dimension=CompositionDimension.ADVERSARIAL,
                structures=["Z_7"],
                metadata={"adversarial_type": adv},
            )
            s = compute_adversarial_strength(t)
            assert 0.0 < s <= 1.0, f"Strength out of range for type {adv}: {s}"
