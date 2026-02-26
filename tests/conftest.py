"""
Shared fixtures for the ALGEBRAID test suite.
"""

import random
import pytest

from algebraid.primitives import (
    CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField, QuaternionGroup,
)
from algebraid.composers import (
    AlgebraicOperation, ComposedFunction, DirectProduct, make_standard_operations,
)
from algebraid.task_model import Task, TaskFamily, CompositionDimension
from algebraid.generator import AlgebraidGenerator
from algebraid.tasks.verbalizer import Verbalizer


# ── Deterministic RNG ─────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    """A deterministic random.Random instance for reproducible tests."""
    return random.Random(42)


# ── Algebraic primitives ──────────────────────────────────────────────────────

@pytest.fixture
def z7():
    return CyclicGroup(7)

@pytest.fixture
def z12():
    return CyclicGroup(12)

@pytest.fixture
def s3():
    return SymmetricGroup(3)

@pytest.fixture
def s4():
    return SymmetricGroup(4)

@pytest.fixture
def d4():
    return DihedralGroup(4)

@pytest.fixture
def gf7():
    return FiniteField(7)

@pytest.fixture
def q8():
    return QuaternionGroup()


# ── Operations ────────────────────────────────────────────────────────────────

@pytest.fixture
def z7_ops(z7, rng):
    return make_standard_operations(z7, rng)

@pytest.fixture
def s3_ops(s3, rng):
    return make_standard_operations(s3, rng)

@pytest.fixture
def q8_ops(q8, rng):
    return make_standard_operations(q8, rng)


# ── Generator & Verbalizer ────────────────────────────────────────────────────

@pytest.fixture
def gen():
    """AlgebraidGenerator with fixed seed for reproducible task generation."""
    return AlgebraidGenerator(seed=42)

@pytest.fixture
def verbalizer():
    return Verbalizer(seed=42)


# ── Pre-built Task instances ──────────────────────────────────────────────────

def _make_task(**kwargs):
    defaults = dict(
        task_id="AG-test000001",
        prompt="Compute x in Z_7.",
        answer="3",
        answer_raw="3",
        depth=1,
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=CompositionDimension.GENERAL,
        structures=["Z_7"],
        metadata={},
        solution_trace=None,
    )
    defaults.update(kwargs)
    return Task(**defaults)


@pytest.fixture
def simple_task():
    """A minimal intra-structure task."""
    return _make_task()


@pytest.fixture
def conceptual_task():
    return _make_task(
        task_id="AG-conceptual01",
        prompt="What is the identity element of Z_7?",
        answer="0",
        answer_raw="0",
        depth=1,
        family=TaskFamily.CONCEPTUAL_QUERY,
        metadata={"query_subtype": "identity", "structure_is_abelian": True, "structure_order": 7},
    )


@pytest.fixture
def adversarial_task():
    return _make_task(
        task_id="AG-adversarial1",
        prompt="Apply double inverse to x=3 in Z_7.",
        answer="3",
        answer_raw="3",
        depth=2,
        dimension=CompositionDimension.ADVERSARIAL,
        solution_trace=[("start", "3"), ("inverse", "4"), ("inverse", "3")],
        metadata={
            "adversarial_type": "double_inverse",
            "wrong_answer": "4",
            "wrong_answer_rationale": "Forgetting second inverse.",
        },
    )


@pytest.fixture
def intermediate_task():
    return _make_task(
        task_id="AG-intermediate1",
        prompt="What is the value after step 1?",
        answer="4",
        answer_raw="4",
        depth=1,
        dimension=CompositionDimension.INTERMEDIATE_STATE,
        solution_trace=[("start", "3"), ("right_mul_1", "4")],
        metadata={"query_step": 1, "total_steps": 3, "final_answer": "6"},
    )
