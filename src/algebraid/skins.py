"""
ALGEBRAID Semantic Skin System.

Provides coherent, real-world narrative layers (skins) that can be applied to
abstract algebraic tasks. Each skin translates:
  - The structure name into a real-world domain (e.g. "a 12-hour clock")
  - Elements into domain objects (e.g. "3 o'clock")
  - Operations into domain actions (e.g. "move forward by 5 hours")

Skins are designed to remain coherent when operations are chained and nested,
so the model reads a flowing narrative rather than abstract symbols.

IMPORTANT: The power_N operation computes x ∗ x ∗ ... ∗ x (N times), which
for additive groups (Z_n) means N*x mod n (scalar multiplication), NOT x+N.
All skin descriptions must accurately reflect this distinction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .primitives.base import AlgebraicStructure


class SemanticSkin(ABC):
    """Abstract base class for a semantic narrative layer."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the skin, e.g., 'Clock Arithmetic'."""
        pass

    @abstractmethod
    def structure_name(self, structure: AlgebraicStructure) -> str:
        """Narrative name for the structure, e.g., 'a 12-hour clock'."""
        pass

    @abstractmethod
    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        """Narrative name for an element, e.g., '3 o\'clock'."""
        pass

    @abstractmethod
    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        """Narrative description for an operation, e.g., 'move forward by 2 hours'."""
        pass


# ===========================================================================
# CyclicGroup Skins
# ===========================================================================

class ClockArithmeticSkin(SemanticSkin):
    """Z_n as a clock face with n positions (0 to n-1).

    Key semantics for Z_n (additive group):
    - op(a, b) = a + b mod n  →  "advance by b"
    - inverse(a) = -a mod n   →  "move backward"
    - power_k(x) = k*x mod n  →  "multiply the current position by k"
    """

    @property
    def name(self) -> str:
        return "Clock Arithmetic"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a {structure.n}-position clock (positions 0 to {structure.n - 1})"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return f"position {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        n = structure.n
        if op_name == "inverse":
            return "move to the position that is the additive inverse (i.e., the position that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the conjugation by {k} (compute {k} + x + ({n} - {k}) mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"advance the clock by {k} position{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current position by {k} (mod {n})"
        return "advance by one position"


class MusicIntervalsSkin(SemanticSkin):
    """Z_n as a chromatic musical scale with n tones.

    Key semantics for Z_n (additive group):
    - op(a, b) = a + b mod n  →  "ascend by b semitones"
    - power_k(x) = k*x mod n  →  "multiply the tone number by k"
    """

    NOTES_12 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    @property
    def name(self) -> str:
        return "Musical Intervals"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a {structure.n}-tone musical scale"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        if structure.n == 12:
            return self.NOTES_12[element % 12]
        return f"tone {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        n = structure.n
        if op_name == "inverse":
            return "find the tone that is the additive inverse (the tone that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the interval transformation by {k} (conjugation mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"ascend by {k} semitone{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current tone number by {k} (mod {n})"
        return "ascend by one semitone"


class RobotStepsSkin(SemanticSkin):
    """Z_n as a robot moving in n discrete positions around a circular track.

    Key semantics for Z_n (additive group):
    - op(a, b) = a + b mod n  →  "move forward by b stops"
    - power_k(x) = k*x mod n  →  "multiply the stop number by k"
    """

    @property
    def name(self) -> str:
        return "Robot Steps"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a circular track with {structure.n} equally-spaced stops (numbered 0 to {structure.n - 1})"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return f"stop {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        n = structure.n
        if op_name == "inverse":
            return "move to the stop that is the additive inverse (the stop that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the track transformation by {k} stop{'s' if k != 1 else ''} (conjugation mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"move forward by {k} stop{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current stop number by {k} (mod {n})"
        return "move forward one stop"


class ColorWheelSkin(SemanticSkin):
    """Z_n as positions on a color wheel.

    Key semantics for Z_n (additive group):
    - op(a, b) = a + b mod n  →  "rotate by b hues"
    - power_k(x) = k*x mod n  →  "multiply the hue number by k"
    """

    COLORS_8 = ["red", "orange", "yellow", "chartreuse", "green", "teal", "blue", "violet"]

    @property
    def name(self) -> str:
        return "Color Wheel"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a color wheel with {structure.n} hues"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        if structure.n == 8:
            return self.COLORS_8[element % 8]
        return f"hue {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        n = structure.n
        if op_name == "inverse":
            return "find the hue that is the additive inverse (the hue that sums to 0)"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the color transformation by {k} hue{'s' if k != 1 else ''} (conjugation mod {n})"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"rotate the wheel forward by {k} hue{'s' if k != 1 else ''} (add {k} mod {n})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply the current hue number by {k} (mod {n})"
        return "rotate forward by one hue"


# ===========================================================================
# SymmetricGroup Skins
# ===========================================================================

class DeckOfCardsSkin(SemanticSkin):
    """S_n as rearrangements of n cards.

    IMPORTANT: S_n elements are 1-indexed tuples, e.g. (1,2,3) for the identity
    of S_3. Card names map 1→Card 1, 2→Card 2, etc. to avoid off-by-one confusion.

    Key semantics for S_n (composition):
    - op(a, b) = a ∘ b  →  "apply shuffle b, then shuffle a"
    - power_k(x) = x ∘ x ∘ ... ∘ x (k times)  →  "apply the current arrangement as a shuffle k times"
    """

    def _card_label(self, i: int, n: int) -> str:
        """Map a 1-indexed position to a card name."""
        # Use simple numbered cards for clarity
        return f"Card {i}"

    @property
    def name(self) -> str:
        return "Deck of Cards"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a deck of {structure.n} cards (Card 1 through Card {structure.n})"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        if isinstance(element, (list, tuple)):
            labels = [self._card_label(i, structure.n) for i in element]
            return "[" + ", ".join(labels) + "]"
        return str(element)

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "undo the last shuffle (find the inverse permutation)"
        if "conj" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                labels = [self._card_label(i, structure.n) for i in perm]
                return f"conjugate by the shuffle [{', '.join(labels)}]"
            return f"conjugate by the shuffle {perm}"
        if "mul" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                labels = [self._card_label(i, structure.n) for i in perm]
                return f"apply the shuffle [{', '.join(labels)}] (compose on the right)"
            return f"apply the shuffle {perm}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"compose the current arrangement with itself {k} time{'s' if k != 1 else ''} (apply it as a shuffle {k} time{'s' if k != 1 else ''})"
        return "apply a shuffle"


class SeatingSkin(SemanticSkin):
    """S_n as seating arrangements of n people.

    IMPORTANT: S_n elements are 1-indexed. Person names map 1→Alice, 2→Bob, etc.

    Key semantics for S_n (composition):
    - power_k(x) = x^k  →  "apply the current rearrangement k times in succession"
    """

    NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]

    def _person_name(self, i: int) -> str:
        """Map a 1-indexed position to a person name."""
        idx = i - 1  # Convert 1-indexed to 0-indexed
        if 0 <= idx < len(self.NAMES):
            return self.NAMES[idx]
        return f"Person {i}"

    @property
    def name(self) -> str:
        return "Seating Arrangements"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        names = [self._person_name(i) for i in range(1, structure.n + 1)]
        return f"the possible seating arrangements of {', '.join(names)}"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        if isinstance(element, (list, tuple)):
            labels = [self._person_name(i) for i in element]
            return "[" + ", ".join(labels) + "]"
        return str(element)

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "reverse the last rearrangement (find the inverse permutation)"
        if "conj" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                labels = [self._person_name(i) for i in perm]
                return f"conjugate by the rearrangement [{', '.join(labels)}]"
            return f"conjugate by the rearrangement {perm}"
        if "mul" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                labels = [self._person_name(i) for i in perm]
                return f"rearrange seats to [{', '.join(labels)}] (compose on the right)"
            return f"rearrange seats to {perm}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the current seating rearrangement {k} time{'s' if k != 1 else ''} in succession"
        return "rearrange the seats"


# ===========================================================================
# DihedralGroup Skins
# ===========================================================================

class PolygonSymmetriesSkin(SemanticSkin):
    """D_n as symmetries of a regular n-gon.

    Key semantics for D_n:
    - Elements are (r, s) where r=rotation index, s=reflection flag
    - power_k(x) = x * x * ... * x (k times in D_n)  →  "apply the current symmetry k times"
    """

    POLYGON_NAMES = {
        3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon",
        7: "heptagon", 8: "octagon", 9: "nonagon", 10: "decagon",
    }

    @property
    def name(self) -> str:
        return "Polygon Symmetries"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        poly = self.POLYGON_NAMES.get(structure.n, f"{structure.n}-gon")
        return f"the symmetry group of a regular {poly}"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        r, s = element
        angle = round(360 * r / structure.n)
        if s == 0:
            if r == 0:
                return "no transformation (identity)"
            return f"rotation by {angle} degrees"
        if r == 0:
            return "reflection (across the primary axis)"
        return f"reflection, then rotation by {angle} degrees"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
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
                if r == 0:
                    return "apply the identity (no change)"
                return f"rotate by {angle} degrees"
            if r == 0:
                return "reflect across the primary axis"
            return f"reflect, then rotate by {angle} degrees"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"compose the current transformation with itself {k} time{'s' if k != 1 else ''}"
        return "apply a symmetry"


class TileFlipSkin(SemanticSkin):
    """D_n as flipping and rotating a tile.

    Key semantics for D_n:
    - power_k(x) = x^k in D_n  →  "apply the current tile orientation k times"
    """

    @property
    def name(self) -> str:
        return "Tile Flips and Rotations"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"the {2 * structure.n} ways to orient a regular tile with {structure.n} sides"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        r, s = element
        steps = r
        if s == 0:
            if r == 0:
                return "original orientation"
            return f"rotated {steps} notch{'es' if steps != 1 else ''} clockwise"
        if r == 0:
            return "flipped (no rotation)"
        return f"flipped, then rotated {steps} notch{'es' if steps != 1 else ''} clockwise"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
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
                if r == 0:
                    return "apply the identity (no change)"
                return f"rotate {r} notch{'es' if r != 1 else ''} clockwise"
            if r == 0:
                return "flip the tile"
            return f"flip the tile, then rotate {r} notch{'es' if r != 1 else ''} clockwise"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the current tile orientation {k} time{'s' if k != 1 else ''} in succession"
        return "apply a tile move"


# ===========================================================================
# FiniteField Skins
# ===========================================================================

class SecretCodesSkin(SemanticSkin):
    """GF(p) as a secret code system with p symbols.

    Key semantics for GF(p) (additive group):
    - op(a, b) = a + b mod p  →  "shift code by adding b"
    - power_k(x) = k*x mod p  →  "multiply the code value by k"
    """

    @property
    def name(self) -> str:
        return "Secret Codes"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a secret code system with {structure.p} symbols (0 to {structure.p - 1})"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return f"code {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
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
    """GF(p) as modular arithmetic with p residues.

    Key semantics for GF(p) (additive group):
    - power_k(x) = k*x mod p  →  "multiply by k"
    """

    @property
    def name(self) -> str:
        return "Modular Arithmetic"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"arithmetic modulo {structure.p}"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return str(element)

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        p = structure.p
        if op_name == "inverse":
            return f"take the additive inverse (negate mod {p})"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply conjugation by {k} (compute {k} + x + ({p} - {k}) mod {p})"
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


# ===========================================================================
# Skin Registry
# ===========================================================================

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
