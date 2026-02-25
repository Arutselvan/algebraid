"""
Semantic skin system for ALGEBRAID.

A *skin* wraps an abstract algebraic structure in a coherent real-world
narrative so that the same mathematical task can be presented in many
surface forms.  Each skin translates three things:

    1. The structure itself  — e.g. Z_12 becomes "a 12-hour clock".
    2. Individual elements   — e.g. 3 becomes "3 o'clock".
    3. Operations            — e.g. right_mul_5 becomes "advance by 5 hours".

Ten skins are provided, covering cyclic groups, symmetric groups,
dihedral groups, and finite fields.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from .primitives.base import AlgebraicStructure


# ── Abstract base ───────────────────────────────────────────────────────────

class SemanticSkin(ABC):
    """Base class for all semantic skins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable skin name."""

    @abstractmethod
    def structure_name(self, structure: AlgebraicStructure) -> str:
        """Narrative name for the algebraic structure."""

    @abstractmethod
    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        """Narrative name for a single element."""

    @abstractmethod
    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        """Narrative description of an operation step."""


# ═══════════════════════════════════════════════════════════════════════════
# Cyclic-group skins  (Z_n — additive)
#
# Semantics reminder
#   op(a, b)    = a + b mod n
#   inverse(a)  = −a mod n
#   power_k(x)  = k·x mod n   (scalar multiplication, NOT x + k)
# ═══════════════════════════════════════════════════════════════════════════

class ClockArithmeticSkin(SemanticSkin):
    """Z_n presented as positions on an n-hour clock."""

    @property
    def name(self) -> str:
        return "Clock Arithmetic"

    def structure_name(self, structure):
        return f"a {structure.n}-position clock (positions 0 to {structure.n - 1})"

    def element_name(self, element, structure):
        return f"position {element}"

    def op_description(self, op_name, fixed_args, structure):
        n = structure.n
        if op_name == "inverse":
            return "move to the additive inverse (the position that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"conjugate by {k} (compute {k} + x + ({n} - {k}) mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"advance the clock by {k} position{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current position by {k} (mod {n})"
        return "advance by one position"


class MusicIntervalsSkin(SemanticSkin):
    """Z_n presented as tones on a chromatic scale."""

    _NOTES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    @property
    def name(self) -> str:
        return "Musical Intervals"

    def structure_name(self, structure):
        return f"a {structure.n}-tone musical scale"

    def element_name(self, element, structure):
        if structure.n == 12:
            return self._NOTES_12[element % 12]
        return f"tone {element}"

    def op_description(self, op_name, fixed_args, structure):
        n = structure.n
        if op_name == "inverse":
            return "find the additive inverse tone (the tone that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply interval conjugation by {k} (mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"ascend by {k} semitone{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current tone number by {k} (mod {n})"
        return "ascend by one semitone"


class RobotStepsSkin(SemanticSkin):
    """Z_n presented as stops on a circular track."""

    @property
    def name(self) -> str:
        return "Robot Steps"

    def structure_name(self, structure):
        return f"a circular track with {structure.n} stops (numbered 0 to {structure.n - 1})"

    def element_name(self, element, structure):
        return f"stop {element}"

    def op_description(self, op_name, fixed_args, structure):
        n = structure.n
        if op_name == "inverse":
            return "move to the additive inverse stop (the stop that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply track conjugation by {k} stop{'s' if k != 1 else ''} (mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"move forward by {k} stop{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current stop number by {k} (mod {n})"
        return "move forward one stop"


class ColorWheelSkin(SemanticSkin):
    """Z_n presented as hues on a color wheel."""

    _COLORS_8 = ["red", "orange", "yellow", "chartreuse", "green", "teal", "blue", "violet"]

    @property
    def name(self) -> str:
        return "Color Wheel"

    def structure_name(self, structure):
        return f"a color wheel with {structure.n} hues"

    def element_name(self, element, structure):
        if structure.n == 8:
            return self._COLORS_8[element % 8]
        return f"hue {element}"

    def op_description(self, op_name, fixed_args, structure):
        n = structure.n
        if op_name == "inverse":
            return "find the additive inverse hue (the hue that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply color conjugation by {k} hue{'s' if k != 1 else ''} (mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"rotate the wheel forward by {k} hue{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current hue number by {k} (mod {n})"
        return "rotate forward by one hue"


# ═══════════════════════════════════════════════════════════════════════════
# Symmetric-group skins  (S_n — permutation composition)
#
# Elements are 1-indexed tuples, e.g. S_3 identity = (1, 2, 3).
#   op(a, b)    = a composed with b
#   power_k(x)  = x composed with itself k times
# ═══════════════════════════════════════════════════════════════════════════

class DeckOfCardsSkin(SemanticSkin):
    """S_n presented as shuffles of a deck of n cards."""

    @property
    def name(self) -> str:
        return "Deck of Cards"

    def structure_name(self, structure):
        return f"a deck of {structure.n} cards (Card 1 through Card {structure.n})"

    def element_name(self, element, structure):
        if isinstance(element, (list, tuple)):
            return "[" + ", ".join(f"Card {i}" for i in element) + "]"
        return str(element)

    def op_description(self, op_name, fixed_args, structure):
        if op_name == "inverse":
            return "undo the last shuffle (find the inverse permutation)"
        if "conj" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                label = "[" + ", ".join(f"Card {i}" for i in perm) + "]"
                return f"conjugate by the shuffle {label}"
            return f"conjugate by the shuffle {perm}"
        if "mul" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                label = "[" + ", ".join(f"Card {i}" for i in perm) + "]"
                return f"apply the shuffle {label} (compose on the right)"
            return f"apply the shuffle {perm}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return (
                f"compose the current arrangement with itself "
                f"{k} time{'s' if k != 1 else ''} "
                f"(apply it as a shuffle {k} time{'s' if k != 1 else ''})"
            )
        return "apply a shuffle"


class SeatingSkin(SemanticSkin):
    """S_n presented as seating arrangements of named people."""

    _NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]

    def _person(self, i: int) -> str:
        """Map a 1-indexed element value to a person name."""
        idx = i - 1
        if 0 <= idx < len(self._NAMES):
            return self._NAMES[idx]
        return f"Person {i}"

    @property
    def name(self) -> str:
        return "Seating Arrangements"

    def structure_name(self, structure):
        names = [self._person(i) for i in range(1, structure.n + 1)]
        return f"the possible seating arrangements of {', '.join(names)}"

    def element_name(self, element, structure):
        if isinstance(element, (list, tuple)):
            return "[" + ", ".join(self._person(i) for i in element) + "]"
        return str(element)

    def op_description(self, op_name, fixed_args, structure):
        if op_name == "inverse":
            return "reverse the last rearrangement (find the inverse permutation)"
        if "conj" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                label = "[" + ", ".join(self._person(i) for i in perm) + "]"
                return f"conjugate by the rearrangement {label}"
            return f"conjugate by the rearrangement {perm}"
        if "mul" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                label = "[" + ", ".join(self._person(i) for i in perm) + "]"
                return f"rearrange seats to {label} (compose on the right)"
            return f"rearrange seats to {perm}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return (
                f"apply the current seating rearrangement "
                f"{k} time{'s' if k != 1 else ''} in succession"
            )
        return "rearrange the seats"


# ═══════════════════════════════════════════════════════════════════════════
# Dihedral-group skins  (D_n — symmetries of a regular n-gon)
#
# Elements are (rotation_index, reflection_flag) pairs.
#   power_k(x) = x composed with itself k times in D_n
# ═══════════════════════════════════════════════════════════════════════════

class PolygonSymmetriesSkin(SemanticSkin):
    """D_n presented as symmetries of a regular polygon."""

    _POLYGON = {
        3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon",
        7: "heptagon", 8: "octagon", 9: "nonagon", 10: "decagon",
    }

    @property
    def name(self) -> str:
        return "Polygon Symmetries"

    def structure_name(self, structure):
        poly = self._POLYGON.get(structure.n, f"{structure.n}-gon")
        return f"the symmetry group of a regular {poly}"

    def element_name(self, element, structure):
        r, s = element
        angle = round(360 * r / structure.n)
        if s == 0:
            return "no transformation (identity)" if r == 0 else f"rotation by {angle} degrees"
        return "reflection (across the primary axis)" if r == 0 else f"reflection, then rotation by {angle} degrees"

    def op_description(self, op_name, fixed_args, structure):
        if op_name == "inverse":
            return "undo the last symmetry (find the inverse transformation)"
        if "conj" in op_name and fixed_args:
            r, s = fixed_args[0]
            angle = round(360 * r / structure.n)
            if s == 0:
                return f"conjugate by rotation of {angle} degrees"
            return f"conjugate by (reflection, then rotation of {angle} degrees)"
        if "mul" in op_name and fixed_args:
            r, s = fixed_args[0]
            angle = round(360 * r / structure.n)
            if s == 0:
                return "apply the identity (no change)" if r == 0 else f"rotate by {angle} degrees"
            return "reflect across the primary axis" if r == 0 else f"reflect, then rotate by {angle} degrees"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"compose the current transformation with itself {k} time{'s' if k != 1 else ''}"
        return "apply a symmetry"


class TileFlipSkin(SemanticSkin):
    """D_n presented as orientations of a regular tile."""

    @property
    def name(self) -> str:
        return "Tile Flips and Rotations"

    def structure_name(self, structure):
        return f"the {2 * structure.n} orientations of a regular {structure.n}-sided tile"

    def element_name(self, element, structure):
        r, s = element
        if s == 0:
            return "original orientation" if r == 0 else f"rotated {r} notch{'es' if r != 1 else ''} clockwise"
        return "flipped (no rotation)" if r == 0 else f"flipped, then rotated {r} notch{'es' if r != 1 else ''} clockwise"

    def op_description(self, op_name, fixed_args, structure):
        if op_name == "inverse":
            return "undo the last move (find the inverse orientation)"
        if "conj" in op_name and fixed_args:
            r, s = fixed_args[0]
            if s == 0:
                return f"conjugate by rotating {r} notch{'es' if r != 1 else ''} clockwise"
            return f"conjugate by (flip, then rotate {r} notch{'es' if r != 1 else ''} clockwise)"
        if "mul" in op_name and fixed_args:
            r, s = fixed_args[0]
            if s == 0:
                return "apply the identity (no change)" if r == 0 else f"rotate {r} notch{'es' if r != 1 else ''} clockwise"
            return "flip the tile" if r == 0 else f"flip, then rotate {r} notch{'es' if r != 1 else ''} clockwise"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the current tile orientation {k} time{'s' if k != 1 else ''} in succession"
        return "apply a tile move"


# ═══════════════════════════════════════════════════════════════════════════
# Finite-field skins  (GF(p) — additive group)
#
# Same scalar-multiplication semantics as Z_n for power_k.
# ═══════════════════════════════════════════════════════════════════════════

class SecretCodesSkin(SemanticSkin):
    """GF(p) presented as a secret-code system with p symbols."""

    @property
    def name(self) -> str:
        return "Secret Codes"

    def structure_name(self, structure):
        return f"a secret code system with {structure.p} symbols (0 to {structure.p - 1})"

    def element_name(self, element, structure):
        return f"code {element}"

    def op_description(self, op_name, fixed_args, structure):
        p = structure.p
        if op_name == "inverse":
            return f"find the decryption key (additive inverse mod {p})"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply code conjugation by {k} (compute {k} + x + ({p} - {k}) mod {p})"
        if "left_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"shift the code by adding {k} (mod {p})"
        if "right_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"combine the code with {k} by adding (mod {p})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current code value by {k} (mod {p})"
        return "apply a code transformation"


class ModularArithmeticSkin(SemanticSkin):
    """GF(p) presented as plain modular arithmetic."""

    @property
    def name(self) -> str:
        return "Modular Arithmetic"

    def structure_name(self, structure):
        return f"arithmetic modulo {structure.p}"

    def element_name(self, element, structure):
        return str(element)

    def op_description(self, op_name, fixed_args, structure):
        p = structure.p
        if op_name == "inverse":
            return f"take the additive inverse (negate mod {p})"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"conjugate by {k} (compute {k} + x + ({p} - {k}) mod {p})"
        if "left_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"add {k} (mod {p})"
        if "right_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"add {k} to the result (mod {p})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply by {k} (mod {p})"
        return "apply a modular operation"


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

SKIN_REGISTRY: Dict[str, List[SemanticSkin]] = {
    "CyclicGroup": [
        ClockArithmeticSkin(),
        MusicIntervalsSkin(),
        RobotStepsSkin(),
        ColorWheelSkin(),
    ],
    "SymmetricGroup": [
        DeckOfCardsSkin(),
        SeatingSkin(),
    ],
    "DihedralGroup": [
        PolygonSymmetriesSkin(),
        TileFlipSkin(),
    ],
    "FiniteField": [
        SecretCodesSkin(),
        ModularArithmeticSkin(),
    ],
}
