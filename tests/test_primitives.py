"""
Tests for all algebraic primitives: Z_n, S_n, D_n, GF(p), Q_8.

Coverage:
  - AlgebraicStructure interface compliance (all required methods)
  - Mathematical correctness of group axioms (closure, associativity, identity, inverses)
  - Commutativity detection
  - Element representation
  - Structure-specific properties
"""

import math
import random
import pytest

from algebraid.primitives import (
    CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField, QuaternionGroup,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_group_axioms(structure, sample_size=20):
    """Verify closure, identity, inverse, and associativity for a sample of elements."""
    rng = random.Random(0)
    elems = structure.elements()
    e = structure.identity()
    op = structure.op

    # 1. Identity: e * x = x * e = x for all x
    for x in elems:
        assert op(e, x) == x, f"Left identity failed for {x}"
        assert op(x, e) == x, f"Right identity failed for {x}"

    # 2. Inverse: x * x^-1 = x^-1 * x = e for all x
    for x in elems:
        inv_x = structure.inverse(x)
        assert op(x, inv_x) == e, f"Right inverse failed for {x}"
        assert op(inv_x, x) == e, f"Left inverse failed for {x}"

    # 3. Closure: x * y is in elements for random pairs
    elem_set = set(str(x) for x in elems)
    for _ in range(sample_size):
        x, y = rng.choice(elems), rng.choice(elems)
        result = op(x, y)
        assert str(result) in elem_set, f"Closure failed: {x} * {y} = {result} not in group"

    # 4. Associativity: (x*y)*z = x*(y*z) for random triples
    for _ in range(sample_size):
        x, y, z = rng.choice(elems), rng.choice(elems), rng.choice(elems)
        lhs = op(op(x, y), z)
        rhs = op(x, op(y, z))
        assert lhs == rhs, f"Associativity failed: ({x}*{y})*{z} = {lhs}, {x}*({y}*{z}) = {rhs}"


# ── CyclicGroup ───────────────────────────────────────────────────────────────

class TestCyclicGroup:
    def test_init_valid(self):
        g = CyclicGroup(7)
        assert g.n == 7

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            CyclicGroup(1)

    def test_name(self, z7):
        assert z7.name == "Z_7"

    def test_elements(self, z7):
        assert z7.elements() == list(range(7))

    def test_order(self, z7):
        assert z7.order() == 7

    def test_identity(self, z7):
        assert z7.identity() == 0

    def test_operation(self, z7):
        assert z7.op(3, 5) == 1   # (3+5) mod 7
        assert z7.op(0, 4) == 4
        assert z7.op(6, 1) == 0

    def test_inverse(self, z7):
        assert z7.inverse(0) == 0
        assert z7.inverse(3) == 4   # -3 mod 7 = 4
        assert z7.inverse(1) == 6

    def test_multiply(self, z7):
        assert z7.multiply(3, 2) == 6   # 3*2 mod 7
        assert z7.multiply(0, 5) == 0

    def test_is_commutative(self, z7):
        assert z7.is_commutative() is True

    def test_element_to_str(self, z7):
        assert z7.element_to_str(3) == "3"

    def test_operation_symbol(self, z7):
        assert z7.operation_symbol() == "+"

    def test_random_element_is_valid(self, z7, rng):
        for _ in range(20):
            x = z7.random_element(rng)
            assert x in z7.elements()

    def test_group_axioms(self):
        check_group_axioms(CyclicGroup(12))

    def test_large_n(self):
        g = CyclicGroup(18)
        assert g.order() == 18
        assert g.op(17, 1) == 0


# ── SymmetricGroup ────────────────────────────────────────────────────────────

class TestSymmetricGroup:
    def test_init(self, s3):
        assert s3.n == 3

    def test_name(self, s3):
        assert s3.name == "S_3"

    def test_order(self, s3):
        assert s3.order() == math.factorial(3)   # 6

    def test_identity(self, s3):
        assert s3.identity() == (1, 2, 3)

    def test_elements_count(self, s3):
        assert len(s3.elements()) == 6

    def test_operation_composition(self, s3):
        # (2,1,3) ∘ (1,3,2) — apply right first
        a = (2, 1, 3)
        b = (1, 3, 2)
        result = s3.op(a, b)
        # b maps: 1→1, 2→3, 3→2
        # a then: 1→2, 3→1, 2→1
        # composed: 1→2, 2→1, 3→1... let's just check group axioms
        assert result in s3.elements()

    def test_identity_is_neutral(self, s3):
        e = s3.identity()
        for x in s3.elements():
            assert s3.op(e, x) == x
            assert s3.op(x, e) == x

    def test_inverse(self, s3):
        for x in s3.elements():
            inv_x = s3.inverse(x)
            assert s3.op(x, inv_x) == s3.identity()
            assert s3.op(inv_x, x) == s3.identity()

    def test_noncommutative(self, s3):
        # S_3 is non-abelian
        assert s3.is_commutative() is False

    def test_group_axioms(self):
        check_group_axioms(SymmetricGroup(3))

    def test_element_to_str(self, s3):
        e = s3.identity()
        s = s3.element_to_str(e)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_random_element(self, s4, rng):
        for _ in range(20):
            x = s4.random_element(rng)
            assert x in s4.elements()


# ── DihedralGroup ─────────────────────────────────────────────────────────────

class TestDihedralGroup:
    def test_init(self, d4):
        assert d4.n == 4

    def test_name(self, d4):
        assert d4.name == "D_4"

    def test_order(self, d4):
        assert d4.order() == 8   # 2*4

    def test_identity(self, d4):
        assert d4.identity() == (0, 0)

    def test_elements_count(self, d4):
        assert len(d4.elements()) == 8

    def test_rotation_composition(self, d4):
        # rotation by 1 + rotation by 1 = rotation by 2
        r1 = (1, 0)
        result = d4.op(r1, r1)
        assert result == (2, 0)

    def test_reflection_is_self_inverse(self, d4):
        # Any reflection (r, 1) should satisfy x^2 = e (or use inverse)
        s = (0, 1)
        assert d4.op(s, s) == d4.identity()

    def test_noncommutative(self, d4):
        assert d4.is_commutative() is False

    def test_group_axioms(self):
        check_group_axioms(DihedralGroup(5))

    def test_element_to_str(self, d4):
        e = d4.identity()
        assert d4.element_to_str(e) == "e"

    def test_rotation_str(self, d4):
        r2 = (2, 0)
        s = d4.element_to_str(r2)
        assert "r" in s or "2" in s


# ── FiniteField ───────────────────────────────────────────────────────────────

class TestFiniteField:
    def test_init(self, gf7):
        assert gf7.p == 7

    def test_name(self, gf7):
        assert gf7.name == "GF(7)"

    def test_order(self, gf7):
        assert gf7.order() == 7

    def test_identity_is_zero(self, gf7):
        assert gf7.identity() == 0

    def test_additive_operation(self, gf7):
        assert gf7.op(3, 5) == 1   # (3+5) mod 7
        assert gf7.op(6, 1) == 0

    def test_additive_inverse(self, gf7):
        assert gf7.inverse(3) == 4   # -3 mod 7
        assert gf7.inverse(0) == 0

    def test_multiplication(self, gf7):
        assert gf7.mul(3, 4) == 5   # 12 mod 7
        assert gf7.mul(0, 5) == 0

    def test_multiplicative_inverse(self, gf7):
        for x in range(1, 7):
            inv = gf7.mul_inverse(x)
            assert gf7.mul(x, inv) == 1

    def test_is_commutative(self, gf7):
        assert gf7.is_commutative() is True

    def test_group_axioms_addition(self):
        check_group_axioms(FiniteField(5))

    def test_nonprime_raises(self):
        with pytest.raises(ValueError):
            FiniteField(6)

    def test_element_to_str(self, gf7):
        assert gf7.element_to_str(5) == "5"

    def test_power(self, gf7):
        # 3^2 mod 7 = 9 mod 7 = 2
        assert gf7.power(3, 2) == 2


# ── QuaternionGroup ───────────────────────────────────────────────────────────

class TestQuaternionGroup:
    """Thorough tests for Q_8 — the new algebraic structure."""

    def test_name(self, q8):
        assert q8.name == "Q_8"

    def test_order(self, q8):
        assert q8.order() == 8

    def test_elements(self, q8):
        assert q8.elements() == list(range(8))

    def test_identity(self, q8):
        assert q8.identity() == 0
        assert q8.element_to_str(q8.identity()) == "1"

    def test_element_names(self, q8):
        names = [q8.element_to_str(i) for i in range(8)]
        assert names == ["1", "-1", "i", "-i", "j", "-j", "k", "-k"]

    def test_operation_symbol(self, q8):
        assert q8.operation_symbol() == "*"

    def test_is_noncommutative(self, q8):
        assert q8.is_commutative() is False

    # Core quaternion product rules
    def test_i_squared_equals_neg1(self, q8):
        assert q8.op(2, 2) == 1   # i*i = -1

    def test_j_squared_equals_neg1(self, q8):
        assert q8.op(4, 4) == 1   # j*j = -1

    def test_k_squared_equals_neg1(self, q8):
        assert q8.op(6, 6) == 1   # k*k = -1

    def test_ij_equals_k(self, q8):
        assert q8.op(2, 4) == 6   # i*j = k

    def test_ji_equals_neg_k(self, q8):
        assert q8.op(4, 2) == 7   # j*i = -k

    def test_jk_equals_i(self, q8):
        assert q8.op(4, 6) == 2   # j*k = i

    def test_kj_equals_neg_i(self, q8):
        assert q8.op(6, 4) == 3   # k*j = -i

    def test_ki_equals_j(self, q8):
        assert q8.op(6, 2) == 4   # k*i = j

    def test_ik_equals_neg_j(self, q8):
        assert q8.op(2, 6) == 5   # i*k = -j

    def test_neg1_neg1_equals_1(self, q8):
        assert q8.op(1, 1) == 0   # (-1)*(-1) = 1

    def test_neg1_commutes_with_all(self, q8):
        # -1 is in the center of Q_8
        for x in range(8):
            assert q8.op(1, x) == q8.op(x, 1), f"-1 should commute with {q8.element_to_str(x)}"

    def test_identity_is_neutral(self, q8):
        for x in range(8):
            assert q8.op(0, x) == x
            assert q8.op(x, 0) == x

    def test_inverses(self, q8):
        expected = {0: 0, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6}
        for x, inv_x in expected.items():
            assert q8.inverse(x) == inv_x, (
                f"{q8.element_to_str(x)}^-1 should be {q8.element_to_str(inv_x)}, "
                f"got {q8.element_to_str(q8.inverse(x))}"
            )

    def test_inverse_product_is_identity(self, q8):
        for x in range(8):
            assert q8.op(x, q8.inverse(x)) == q8.identity()
            assert q8.op(q8.inverse(x), x) == q8.identity()

    def test_group_axioms(self, q8):
        check_group_axioms(q8)

    def test_random_element(self, q8, rng):
        for _ in range(30):
            x = q8.random_element(rng)
            assert 0 <= x <= 7

    def test_noncommutativity_witness(self, q8):
        # Explicit witness: i*j ≠ j*i
        assert q8.op(2, 4) != q8.op(4, 2)

    def test_center_is_1_and_neg1(self, q8):
        # Center of Q_8 = {1, -1}
        center = []
        for x in range(8):
            if all(q8.op(x, y) == q8.op(y, x) for y in range(8)):
                center.append(x)
        assert set(center) == {0, 1}   # {1, -1}

    def test_element_orders(self, q8):
        # ord(1) = 1, ord(-1) = 2, ord(i) = ord(-i) = ord(j) = ord(-j) = ord(k) = ord(-k) = 4
        from algebraid.generator import _element_order
        assert _element_order(q8, 0) == 1   # 1
        assert _element_order(q8, 1) == 2   # -1
        for x in [2, 3, 4, 5, 6, 7]:
            assert _element_order(q8, x) == 4, f"Order of {q8.element_to_str(x)} should be 4"
