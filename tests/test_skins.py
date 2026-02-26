"""
Tests for the semantic skin system: all 12 skins, SKIN_REGISTRY, and
structure-specific element / operation translations.
"""

import pytest

from algebraid.skins import (
    SemanticSkin, SKIN_REGISTRY,
    # Cyclic skins
    ClockArithmeticSkin, MusicIntervalsSkin, RobotStepsSkin, ColorWheelSkin,
    # Symmetric skins
    DeckOfCardsSkin, SeatingSkin,
    # Dihedral skins
    PolygonSymmetriesSkin, TileFlipSkin,
    # Finite-field skins
    SecretCodesSkin, ModularArithmeticSkin,
    # Quaternion skins
    QuaternionAlgebraSkin, QuaternionRotationSkin,
)
from algebraid.primitives import (
    CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField, QuaternionGroup,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_skins():
    """Flat list of all registered SemanticSkin instances."""
    return [skin for skins in SKIN_REGISTRY.values() for skin in skins]


# ── Registry integrity ────────────────────────────────────────────────────────

class TestSkinRegistry:
    def test_registry_has_all_structure_keys(self):
        expected = {"CyclicGroup", "SymmetricGroup", "DihedralGroup", "FiniteField", "QuaternionGroup"}
        assert expected.issubset(set(SKIN_REGISTRY.keys()))

    def test_cyclic_has_four_skins(self):
        assert len(SKIN_REGISTRY["CyclicGroup"]) == 4

    def test_symmetric_has_two_skins(self):
        assert len(SKIN_REGISTRY["SymmetricGroup"]) == 2

    def test_dihedral_has_two_skins(self):
        assert len(SKIN_REGISTRY["DihedralGroup"]) == 2

    def test_finite_field_has_two_skins(self):
        assert len(SKIN_REGISTRY["FiniteField"]) == 2

    def test_quaternion_has_two_skins(self):
        assert len(SKIN_REGISTRY["QuaternionGroup"]) == 2

    def test_total_skin_count_is_twelve(self):
        total = sum(len(v) for v in SKIN_REGISTRY.values())
        assert total == 12

    def test_all_skins_are_semantic_skin_instances(self):
        for skin in _all_skins():
            assert isinstance(skin, SemanticSkin)

    def test_all_skins_have_nonempty_names(self):
        for skin in _all_skins():
            assert isinstance(skin.name, str) and len(skin.name) > 0


# ── SemanticSkin interface compliance ─────────────────────────────────────────

class TestSkinInterface:
    """Every skin must implement all four abstract methods."""

    def test_clock_arithmetic_implements_interface(self, z7):
        skin = ClockArithmeticSkin()
        assert isinstance(skin.structure_name(z7), str)
        assert isinstance(skin.element_name(3, z7), str)
        assert isinstance(skin.op_description("inverse", (), z7), str)

    def test_all_skins_return_strings_for_cyclic(self, z7):
        for skin in SKIN_REGISTRY["CyclicGroup"]:
            assert isinstance(skin.structure_name(z7), str)
            assert isinstance(skin.element_name(0, z7), str)
            assert isinstance(skin.op_description("inverse", (), z7), str)

    def test_all_skins_return_strings_for_symmetric(self, s3):
        for skin in SKIN_REGISTRY["SymmetricGroup"]:
            assert isinstance(skin.structure_name(s3), str)
            e = s3.identity()
            assert isinstance(skin.element_name(e, s3), str)
            assert isinstance(skin.op_description("inverse", (), s3), str)

    def test_all_skins_return_strings_for_dihedral(self, d4):
        for skin in SKIN_REGISTRY["DihedralGroup"]:
            assert isinstance(skin.structure_name(d4), str)
            e = d4.identity()
            assert isinstance(skin.element_name(e, d4), str)
            assert isinstance(skin.op_description("inverse", (), d4), str)

    def test_all_skins_return_strings_for_finite_field(self, gf7):
        for skin in SKIN_REGISTRY["FiniteField"]:
            assert isinstance(skin.structure_name(gf7), str)
            assert isinstance(skin.element_name(3, gf7), str)
            assert isinstance(skin.op_description("inverse", (), gf7), str)

    def test_all_skins_return_strings_for_quaternion(self, q8):
        for skin in SKIN_REGISTRY["QuaternionGroup"]:
            assert isinstance(skin.structure_name(q8), str)
            assert isinstance(skin.element_name(2, q8), str)  # i
            assert isinstance(skin.op_description("inverse", (), q8), str)


# ── ClockArithmeticSkin ───────────────────────────────────────────────────────

class TestClockArithmeticSkin:
    def setup_method(self):
        self.skin = ClockArithmeticSkin()
        self.z7 = CyclicGroup(7)

    def test_name(self):
        assert "Clock" in self.skin.name

    def test_structure_name_includes_n(self):
        s = self.skin.structure_name(self.z7)
        assert "7" in s

    def test_element_name_includes_position(self):
        assert "3" in self.skin.element_name(3, self.z7)

    def test_inverse_op(self):
        desc = self.skin.op_description("inverse", (), self.z7)
        assert len(desc) > 5

    def test_mul_op_includes_k(self):
        desc = self.skin.op_description("right_mul_3", (3,), self.z7)
        assert "3" in desc

    def test_power_op_includes_k(self):
        desc = self.skin.op_description("power_2", (2,), self.z7)
        assert "2" in desc


# ── MusicIntervalsSkin ────────────────────────────────────────────────────────

class TestMusicIntervalsSkin:
    def setup_method(self):
        self.skin = MusicIntervalsSkin()

    def test_name(self):
        assert "Music" in self.skin.name or "Interval" in self.skin.name

    def test_z12_element_name_is_note(self):
        z12 = CyclicGroup(12)
        # element 0 → "C"
        assert self.skin.element_name(0, z12) == "C"
        # element 7 → "G"
        assert self.skin.element_name(7, z12) == "G"

    def test_non_12_element_name_fallback(self):
        z7 = CyclicGroup(7)
        s = self.skin.element_name(3, z7)
        assert "3" in s

    def test_mul_op_mentions_semitone(self):
        z12 = CyclicGroup(12)
        desc = self.skin.op_description("right_mul_3", (3,), z12)
        assert "semitone" in desc.lower() or "3" in desc


# ── RobotStepsSkin ────────────────────────────────────────────────────────────

class TestRobotStepsSkin:
    def setup_method(self):
        self.skin = RobotStepsSkin()
        self.z7 = CyclicGroup(7)

    def test_name(self):
        assert "Robot" in self.skin.name or "Step" in self.skin.name

    def test_element_name_includes_stop(self):
        s = self.skin.element_name(4, self.z7)
        assert "4" in s

    def test_structure_name_includes_track(self):
        s = self.skin.structure_name(self.z7)
        assert "7" in s

    def test_mul_op(self):
        desc = self.skin.op_description("right_mul_5", (5,), self.z7)
        assert "5" in desc


# ── ColorWheelSkin ────────────────────────────────────────────────────────────

class TestColorWheelSkin:
    def setup_method(self):
        self.skin = ColorWheelSkin()

    def test_name(self):
        assert "Color" in self.skin.name

    def test_z8_element_name_is_color(self):
        z8 = CyclicGroup(8)
        assert self.skin.element_name(0, z8) == "red"
        assert self.skin.element_name(4, z8) == "green"

    def test_non_8_fallback(self):
        z7 = CyclicGroup(7)
        s = self.skin.element_name(3, z7)
        assert "3" in s

    def test_structure_name_includes_hues(self):
        z8 = CyclicGroup(8)
        assert "hue" in self.skin.structure_name(z8).lower() or "8" in self.skin.structure_name(z8)


# ── DeckOfCardsSkin ───────────────────────────────────────────────────────────

class TestDeckOfCardsSkin:
    def setup_method(self):
        self.skin = DeckOfCardsSkin()
        self.s3 = SymmetricGroup(3)

    def test_name(self):
        assert "Card" in self.skin.name or "Deck" in self.skin.name

    def test_element_name_is_card_list(self):
        s = self.skin.element_name((1, 2, 3), self.s3)
        assert "Card" in s

    def test_inverse_op(self):
        desc = self.skin.op_description("inverse", (), self.s3)
        assert "shuffle" in desc.lower() or "inverse" in desc.lower()

    def test_mul_op_with_tuple(self):
        perm = (2, 1, 3)
        desc = self.skin.op_description("right_mul", (perm,), self.s3)
        assert "Card" in desc or "shuffle" in desc.lower()

    def test_power_op(self):
        desc = self.skin.op_description("power_2", (2,), self.s3)
        assert "2" in desc


# ── SeatingSkin ───────────────────────────────────────────────────────────────

class TestSeatingSkin:
    def setup_method(self):
        self.skin = SeatingSkin()
        self.s3 = SymmetricGroup(3)

    def test_name(self):
        assert "Seat" in self.skin.name

    def test_element_name_includes_person(self):
        s = self.skin.element_name((1, 2, 3), self.s3)
        assert "Alice" in s

    def test_structure_name_includes_alice(self):
        s = self.skin.structure_name(self.s3)
        assert "Alice" in s

    def test_inverse_op(self):
        desc = self.skin.op_description("inverse", (), self.s3)
        assert len(desc) > 5


# ── PolygonSymmetriesSkin ─────────────────────────────────────────────────────

class TestPolygonSymmetriesSkin:
    def setup_method(self):
        self.skin = PolygonSymmetriesSkin()
        self.d4 = DihedralGroup(4)

    def test_name(self):
        assert "Polygon" in self.skin.name or "Symmetr" in self.skin.name

    def test_identity_element_name(self):
        s = self.skin.element_name((0, 0), self.d4)
        assert "identity" in s.lower() or "transformation" in s.lower()

    def test_rotation_element_name_includes_degrees(self):
        s = self.skin.element_name((1, 0), self.d4)
        assert "degree" in s.lower() or "rotation" in s.lower()

    def test_reflection_element_name_includes_reflect(self):
        s = self.skin.element_name((0, 1), self.d4)
        assert "reflect" in s.lower()

    def test_structure_name_includes_square(self):
        s = self.skin.structure_name(self.d4)
        assert "square" in s.lower() or "symmetry" in s.lower()

    def test_mul_op_rotation(self):
        desc = self.skin.op_description("right_mul", ((1, 0),), self.d4)
        assert "rotate" in desc.lower() or "degree" in desc.lower()

    def test_mul_op_reflection(self):
        desc = self.skin.op_description("right_mul", ((0, 1),), self.d4)
        assert "reflect" in desc.lower()


# ── TileFlipSkin ──────────────────────────────────────────────────────────────

class TestTileFlipSkin:
    def setup_method(self):
        self.skin = TileFlipSkin()
        self.d4 = DihedralGroup(4)

    def test_name(self):
        assert "Tile" in self.skin.name or "Flip" in self.skin.name

    def test_identity_element_name(self):
        s = self.skin.element_name((0, 0), self.d4)
        assert "original" in s.lower() or "orientation" in s.lower()

    def test_rotation_element_name_includes_clockwise(self):
        s = self.skin.element_name((2, 0), self.d4)
        assert "clockwise" in s.lower() or "2" in s

    def test_flip_element_name_includes_flip(self):
        s = self.skin.element_name((0, 1), self.d4)
        assert "flip" in s.lower()

    def test_inverse_op(self):
        desc = self.skin.op_description("inverse", (), self.d4)
        assert "undo" in desc.lower() or "inverse" in desc.lower()


# ── SecretCodesSkin ───────────────────────────────────────────────────────────

class TestSecretCodesSkin:
    def setup_method(self):
        self.skin = SecretCodesSkin()
        self.gf7 = FiniteField(7)

    def test_name(self):
        assert "Secret" in self.skin.name or "Code" in self.skin.name

    def test_element_name_is_code(self):
        s = self.skin.element_name(5, self.gf7)
        assert "5" in s

    def test_structure_name_includes_p(self):
        s = self.skin.structure_name(self.gf7)
        assert "7" in s

    def test_inverse_op_mentions_decryption(self):
        desc = self.skin.op_description("inverse", (), self.gf7)
        assert "decrypt" in desc.lower() or "inverse" in desc.lower()

    def test_left_mul_op(self):
        desc = self.skin.op_description("left_mul_3", (3,), self.gf7)
        assert "3" in desc

    def test_power_op(self):
        desc = self.skin.op_description("power_2", (2,), self.gf7)
        assert "2" in desc


# ── ModularArithmeticSkin ─────────────────────────────────────────────────────

class TestModularArithmeticSkin:
    def setup_method(self):
        self.skin = ModularArithmeticSkin()
        self.gf7 = FiniteField(7)

    def test_name(self):
        assert "Modular" in self.skin.name or "Arithmetic" in self.skin.name

    def test_element_name_is_plain_number(self):
        assert self.skin.element_name(5, self.gf7) == "5"

    def test_structure_name_includes_modulo(self):
        s = self.skin.structure_name(self.gf7)
        assert "modulo" in s.lower() or "7" in s

    def test_inverse_op_mentions_negate(self):
        desc = self.skin.op_description("inverse", (), self.gf7)
        assert "negate" in desc.lower() or "inverse" in desc.lower()

    def test_right_mul_op_includes_k(self):
        desc = self.skin.op_description("right_mul_4", (4,), self.gf7)
        assert "4" in desc

    def test_power_op_includes_k(self):
        desc = self.skin.op_description("power_3", (3,), self.gf7)
        assert "3" in desc


# ── QuaternionAlgebraSkin ─────────────────────────────────────────────────────

class TestQuaternionAlgebraSkin:
    def setup_method(self):
        self.skin = QuaternionAlgebraSkin()
        self.q8 = QuaternionGroup()

    def test_name(self):
        assert "Quaternion" in self.skin.name

    def test_structure_name_includes_q8(self):
        s = self.skin.structure_name(self.q8)
        assert "Q_8" in s or "quaternion" in s.lower()

    def test_element_names(self):
        expected = ["1", "-1", "i", "-i", "j", "-j", "k", "-k"]
        for idx, name in enumerate(expected):
            assert self.skin.element_name(idx, self.q8) == name

    def test_inverse_op(self):
        desc = self.skin.op_description("inverse", (), self.q8)
        assert "inverse" in desc.lower() or "quaternion" in desc.lower()

    def test_left_mul_op_includes_element_name(self):
        desc = self.skin.op_description("left_mul_2", (2,), self.q8)
        assert "i" in desc   # element 2 → "i"

    def test_right_mul_op_includes_element_name(self):
        desc = self.skin.op_description("right_mul_4", (4,), self.q8)
        assert "j" in desc   # element 4 → "j"

    def test_conj_op_includes_element_name(self):
        desc = self.skin.op_description("conj_6", (6,), self.q8)
        assert "k" in desc   # element 6 → "k"

    def test_power_op_includes_k(self):
        desc = self.skin.op_description("power_3", (3,), self.q8)
        assert "3" in desc

    def test_unknown_op_fallback(self):
        desc = self.skin.op_description("unknown_op", (), self.q8)
        assert isinstance(desc, str) and len(desc) > 0


# ── QuaternionRotationSkin ────────────────────────────────────────────────────

class TestQuaternionRotationSkin:
    def setup_method(self):
        self.skin = QuaternionRotationSkin()
        self.q8 = QuaternionGroup()

    def test_name(self):
        assert "Quaternion" in self.skin.name or "Rotation" in self.skin.name

    def test_structure_name_includes_rotation(self):
        s = self.skin.structure_name(self.q8)
        assert "rotation" in s.lower() or "3D" in s

    def test_element_0_is_identity_rotation(self):
        s = self.skin.element_name(0, self.q8)
        assert "identity" in s.lower()

    def test_element_1_is_full_flip(self):
        s = self.skin.element_name(1, self.q8)
        assert "flip" in s.lower()

    def test_element_2_is_x_rotation(self):
        s = self.skin.element_name(2, self.q8)
        assert "x" in s.lower()

    def test_element_4_is_y_rotation(self):
        s = self.skin.element_name(4, self.q8)
        assert "y" in s.lower()

    def test_element_6_is_z_rotation(self):
        s = self.skin.element_name(6, self.q8)
        assert "z" in s.lower()

    def test_inverse_op_mentions_reverse(self):
        desc = self.skin.op_description("inverse", (), self.q8)
        assert "reverse" in desc.lower() or "inverse" in desc.lower()

    def test_left_mul_op_mentions_left(self):
        desc = self.skin.op_description("left_mul_2", (2,), self.q8)
        assert "left" in desc.lower() or "x" in desc.lower()

    def test_right_mul_op_mentions_right(self):
        desc = self.skin.op_description("right_mul_4", (4,), self.q8)
        assert "right" in desc.lower() or "y" in desc.lower()

    def test_all_eight_element_names_are_distinct(self):
        names = [self.skin.element_name(i, self.q8) for i in range(8)]
        assert len(set(names)) == 8

    def test_power_op_includes_count(self):
        desc = self.skin.op_description("power_3", (3,), self.q8)
        assert "3" in desc


# ── Cross-skin: no unresolved placeholders ────────────────────────────────────

class TestNoUnresolvedPlaceholders:
    """Ensure no skin accidentally leaves {placeholder} tokens in output."""

    import re

    def _check(self, text: str, context: str):
        import re
        assert not re.search(r'\{[a-zA-Z_]+\}', text), (
            f"Unresolved placeholder in {context}: {text!r}"
        )

    def test_cyclic_skins_no_placeholders(self, z7):
        for skin in SKIN_REGISTRY["CyclicGroup"]:
            self._check(skin.structure_name(z7), f"{skin.name}.structure_name")
            for elem in z7.elements():
                self._check(skin.element_name(elem, z7), f"{skin.name}.element_name({elem})")

    def test_quaternion_skins_no_placeholders(self, q8):
        for skin in SKIN_REGISTRY["QuaternionGroup"]:
            self._check(skin.structure_name(q8), f"{skin.name}.structure_name")
            for idx in range(8):
                self._check(skin.element_name(idx, q8), f"{skin.name}.element_name({idx})")

    def test_dihedral_skins_no_placeholders(self, d4):
        for skin in SKIN_REGISTRY["DihedralGroup"]:
            self._check(skin.structure_name(d4), f"{skin.name}.structure_name")
            for elem in d4.elements():
                self._check(skin.element_name(elem, d4), f"{skin.name}.element_name({elem})")
