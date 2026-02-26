"""
Tests for composers: AlgebraicOperation, ComposedFunction, DirectProduct.
"""

import pytest

from algebraid.primitives import CyclicGroup, SymmetricGroup, QuaternionGroup
from algebraid.composers import (
    AlgebraicOperation, ComposedFunction, DirectProduct, make_standard_operations,
)


# ── AlgebraicOperation ────────────────────────────────────────────────────────

class TestAlgebraicOperation:
    def test_call_with_fixed_args(self, z7):
        op = AlgebraicOperation(
            name="add_3",
            func=lambda x, k: (x + k) % 7,
            arity=1,
            description="add 3 mod 7",
            symbol="+3",
            fixed_args=(3,),
        )
        assert op(1) == 4
        assert op(5) == 1

    def test_repr(self, z7):
        op = AlgebraicOperation(
            name="my_op", func=lambda x: x, arity=1,
            description="identity op", symbol="id", fixed_args=(),
        )
        assert repr(op) == "my_op"


# ── make_standard_operations ──────────────────────────────────────────────────

class TestMakeStandardOperations:
    def test_returns_list(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        assert isinstance(ops, list)
        assert len(ops) > 0

    def test_all_are_algebraic_operations(self, z7, rng):
        for op in make_standard_operations(z7, rng):
            assert isinstance(op, AlgebraicOperation)

    def test_inverse_op_present(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        names = [op.name for op in ops]
        assert "inverse" in names

    def test_inverse_op_correctness(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        inv_op = next(op for op in ops if op.name == "inverse")
        assert inv_op(3) == z7.inverse(3)   # -3 mod 7 = 4

    def test_conj_op_only_for_nonabelian(self, rng):
        # Cyclic groups should not have conjugation
        z5 = CyclicGroup(5)
        ops_z5 = make_standard_operations(z5, rng)
        assert not any("conj" in op.name for op in ops_z5)

        # Symmetric group should have conjugation
        s3 = SymmetricGroup(3)
        ops_s3 = make_standard_operations(s3, rng)
        assert any("conj" in op.name for op in ops_s3)

    def test_q8_has_conj_op(self, q8, rng):
        ops = make_standard_operations(q8, rng)
        assert any("conj" in op.name for op in ops)

    def test_mul_ops_use_valid_elements(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        elems = z7.elements()
        for op in ops:
            if op.fixed_args:
                arg = op.fixed_args[0]
                # For cyclic group, fixed arg should be an int in [0, n)
                if isinstance(arg, int):
                    assert arg in elems, f"Fixed arg {arg} not a valid element"


# ── ComposedFunction ──────────────────────────────────────────────────────────

class TestComposedFunction:
    def test_identity_chain(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        inv_op = next(op for op in ops if op.name == "inverse")
        # Chain of zero ops — not constructable, test length-1 instead
        chain = ComposedFunction([inv_op], z7)
        assert chain(0) == 0  # -0 mod 7 = 0

    def test_single_operation(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        inv_op = next(op for op in ops if op.name == "inverse")
        chain = ComposedFunction([inv_op], z7)
        assert chain(3) == z7.inverse(3)

    def test_chain_depth(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        chosen = [rng.choice(ops) for _ in range(4)]
        chain = ComposedFunction(chosen, z7)
        assert len(chain.operations) == 4

    def test_trace_length(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        chosen = [rng.choice(ops) for _ in range(3)]
        chain = ComposedFunction(chosen, z7)
        x = z7.random_element(rng)
        trace = chain.trace(x)
        assert len(trace) == 4  # start + 3 ops

    def test_trace_starts_with_start(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0]], z7)
        x = 5
        trace = chain.trace(x)
        assert trace[0][0] == "start"
        assert trace[0][1] == x

    def test_trace_final_matches_call(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        chosen = [rng.choice(ops) for _ in range(3)]
        chain = ComposedFunction(chosen, z7)
        x = z7.random_element(rng)
        assert chain(x) == chain.trace(x)[-1][1]

    def test_double_inverse_returns_input(self, z7, rng):
        ops = make_standard_operations(z7, rng)
        inv_op = next(op for op in ops if op.name == "inverse")
        chain = ComposedFunction([inv_op, inv_op], z7)
        for x in z7.elements():
            assert chain(x) == x, f"Double inverse should return {x}, got {chain(x)}"

    def test_q8_chain(self, q8, rng):
        ops = make_standard_operations(q8, rng)
        chosen = [rng.choice(ops) for _ in range(2)]
        chain = ComposedFunction(chosen, q8)
        x = q8.random_element(rng)
        result = chain(x)
        assert 0 <= result <= 7


# ── DirectProduct ─────────────────────────────────────────────────────────────

class TestDirectProduct:
    def test_basic_creation(self):
        G = CyclicGroup(3)
        H = CyclicGroup(4)
        GH = DirectProduct(G, H)
        assert GH.order() == 12  # 3 * 4

    def test_elements_are_tuples(self):
        G = CyclicGroup(2)
        H = CyclicGroup(3)
        GH = DirectProduct(G, H)
        for elem in GH.elements():
            assert isinstance(elem, tuple)
            assert len(elem) == 2

    def test_operation_component_wise(self):
        G = CyclicGroup(3)
        H = CyclicGroup(4)
        GH = DirectProduct(G, H)
        a = (1, 2)
        b = (2, 3)
        result = GH.op(a, b)
        expected = (G.op(1, 2), H.op(2, 3))  # (0, 1)
        assert result == expected

    def test_identity(self):
        G = CyclicGroup(5)
        H = CyclicGroup(7)
        GH = DirectProduct(G, H)
        e = GH.identity()
        assert e == (G.identity(), H.identity())

    def test_inverse(self):
        G = CyclicGroup(5)
        H = CyclicGroup(7)
        GH = DirectProduct(G, H)
        a = (2, 3)
        inv_a = GH.inverse(a)
        assert GH.op(a, inv_a) == GH.identity()

    def test_group_axioms(self):
        G = CyclicGroup(3)
        H = CyclicGroup(4)
        GH = DirectProduct(G, H)
        import random
        rng = random.Random(7)
        elems = GH.elements()
        e = GH.identity()
        for x in elems:
            assert GH.op(e, x) == x
            assert GH.op(x, e) == x
            inv_x = GH.inverse(x)
            assert GH.op(x, inv_x) == e

    def test_nested_product(self):
        G = CyclicGroup(2)
        H = CyclicGroup(3)
        K = CyclicGroup(5)
        GH = DirectProduct(G, H)
        GHK = DirectProduct(GH, K)
        assert GHK.order() == 30  # 2*3*5

    def test_operation_symbol_matches_components_when_same(self):
        """Z_m x Z_n: both use '+', so DirectProduct must also return '+'."""
        GH = DirectProduct(CyclicGroup(3), CyclicGroup(4))
        assert GH.operation_symbol() == "+"

    def test_operation_symbol_falls_back_when_different(self):
        """When G uses '+' and H uses '*' the product must use '.'."""
        G = CyclicGroup(3)        # symbol '+'
        H = SymmetricGroup(3)     # symbol '*'
        GH = DirectProduct(G, H)
        assert GH.operation_symbol() == "."

    def test_description_uses_operation_symbol(self):
        """The outer pair symbol in description must equal operation_symbol()."""
        GH = DirectProduct(CyclicGroup(3), CyclicGroup(4))
        sym = GH.operation_symbol()
        desc = GH.description
        # The description contains a line like "(a1, b1) + (a2, b2) = ..."
        assert f") {sym} (" in desc, (
            f"description uses a different outer symbol than operation_symbol()={sym!r}:\n{desc}"
        )

    def test_description_symbol_not_period_for_cyclic_product(self):
        """Cyclic x Cyclic: description must NOT use '.' as the outer op symbol."""
        GH = DirectProduct(CyclicGroup(5), CyclicGroup(6))
        # '.' would indicate a bug where operation_symbol() hardcoded '.'
        assert ") . (" not in GH.description
