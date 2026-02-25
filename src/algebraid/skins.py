"""
ALGEBRAID Semantic Skin System.

Provides coherent, real-world narrative layers (skins) that can be applied to
abstract algebraic tasks. Each skin translates:
  - The structure name into a real-world domain (e.g. "a 12-hour clock")
  - Elements into domain objects (e.g. "3 o'clock")
  - Operations into domain actions (e.g. "move forward by 5 hours")

Skins are designed to remain coherent when operations are chained and nested,
so the model reads a flowing narrative rather than abstract symbols.
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
    """Z_n as a clock face with n positions (0 to n-1)."""

    @property
    def name(self) -> str:
        return "Clock Arithmetic"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a {structure.n}-position clock (positions 0 to {structure.n - 1})"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return f"position {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "move backward by the same number of steps"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the conjugation by {k} step{'s' if k != 1 else ''} (wrap around)"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"advance the clock by {k} step{'s' if k != 1 else ''}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"advance by one step, {k} times in a row"
        return "advance by one step"


class MusicIntervalsSkin(SemanticSkin):
    """Z_n as a chromatic musical scale with n tones."""

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
        if op_name == "inverse":
            return "descend by the same interval"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the interval transformation by {k} (conjugation)"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"ascend by {k} semitone{'s' if k != 1 else ''}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"ascend by one semitone, {k} times"
        return "ascend by one semitone"


class RobotStepsSkin(SemanticSkin):
    """Z_n as a robot moving in n discrete positions around a circular track."""

    @property
    def name(self) -> str:
        return "Robot Steps"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a circular track with {structure.n} equally-spaced stops (numbered 0 to {structure.n - 1})"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return f"stop {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "move backward by the same number of stops"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the track transformation by {k} stop{'s' if k != 1 else ''} (wrap around)"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"move forward by {k} stop{'s' if k != 1 else ''}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"move forward one stop, {k} times"
        return "move forward one stop"


class ColorWheelSkin(SemanticSkin):
    """Z_n as positions on a color wheel."""

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
        if op_name == "inverse":
            return "rotate the wheel backward by the same amount"
        if "conj" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the color transformation by {k} hue{'s' if k != 1 else ''} (conjugation)"
        if "mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"rotate the wheel forward by {k} hue{'s' if k != 1 else ''}"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"rotate forward by one hue, {k} times"
        return "rotate forward by one hue"


# ===========================================================================
# SymmetricGroup Skins
# ===========================================================================

class DeckOfCardsSkin(SemanticSkin):
    """S_n as rearrangements of n cards."""

    CARD_NAMES = {
        0: "Ace", 1: "Two", 2: "Three", 3: "Four", 4: "Five",
        5: "Six", 6: "Seven", 7: "Eight", 8: "Nine", 9: "Ten",
        10: "Jack", 11: "Queen", 12: "King",
    }

    @property
    def name(self) -> str:
        return "Deck of Cards"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a deck of {structure.n} cards"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        if isinstance(element, (list, tuple)):
            labels = [self.CARD_NAMES.get(i, str(i)) for i in element]
            return "[" + ", ".join(labels) + "]"
        return str(element)

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "undo the last shuffle"
        if "mul" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                labels = [self.CARD_NAMES.get(i, str(i)) for i in perm]
                return f"apply the shuffle [{', '.join(labels)}]"
        return "apply a shuffle"


class SeatingSkin(SemanticSkin):
    """S_n as seating arrangements of n people."""

    NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]

    @property
    def name(self) -> str:
        return "Seating Arrangements"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        names = self.NAMES[:structure.n]
        return f"the possible seating arrangements of {', '.join(names)}"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        if isinstance(element, (list, tuple)):
            names = self.NAMES
            labels = [names[i] if i < len(names) else f"Person {i}" for i in element]
            return "[" + ", ".join(labels) + "]"
        return str(element)

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "reverse the last rearrangement"
        if "mul" in op_name and fixed_args:
            perm = fixed_args[0]
            if isinstance(perm, (list, tuple)):
                names = self.NAMES
                labels = [names[i] if i < len(names) else f"Person {i}" for i in perm]
                return f"rearrange seats to [{', '.join(labels)}]"
        return "rearrange the seats"


# ===========================================================================
# DihedralGroup Skins
# ===========================================================================

class PolygonSymmetriesSkin(SemanticSkin):
    """D_n as symmetries of a regular n-gon."""

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
            return f"rotation by {angle}°"
        return f"reflection, then rotation by {angle}°"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "undo the last symmetry"
        if "mul" in op_name and fixed_args:
            r, s = fixed_args[0]
            angle = round(360 * r / structure.n)
            if s == 0:
                return f"rotate by {angle}°"
            return f"reflect, then rotate by {angle}°"
        return "apply a symmetry"


class TileFlipSkin(SemanticSkin):
    """D_n as flipping and rotating a tile."""

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
        return f"flipped, then rotated {steps} notch{'es' if steps != 1 else ''} clockwise"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "undo the last move"
        if "mul" in op_name and fixed_args:
            r, s = fixed_args[0]
            if s == 0:
                return f"rotate {r} notch{'es' if r != 1 else ''} clockwise"
            return f"flip the tile, then rotate {r} notch{'es' if r != 1 else ''} clockwise"
        return "apply a tile move"


# ===========================================================================
# FiniteField Skins
# ===========================================================================

class SecretCodesSkin(SemanticSkin):
    """GF(p) as a secret code system with p symbols."""

    @property
    def name(self) -> str:
        return "Secret Codes"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"a secret code system with {structure.p} symbols (0 to {structure.p - 1})"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return f"code {element}"

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "find the decryption key (additive inverse)"
        if "left_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"shift the code by adding {k} (mod {structure.p})"
        if "right_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"combine the code with {k} (mod {structure.p})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"apply the code {k} times"
        return "apply a code transformation"


class ModularArithmeticSkin(SemanticSkin):
    """GF(p) as modular arithmetic with p residues."""

    @property
    def name(self) -> str:
        return "Modular Arithmetic"

    def structure_name(self, structure: AlgebraicStructure) -> str:
        return f"arithmetic modulo {structure.p}"

    def element_name(self, element: Any, structure: AlgebraicStructure) -> str:
        return str(element)

    def op_description(self, op_name: str, fixed_args: Tuple, structure: AlgebraicStructure) -> str:
        if op_name == "inverse":
            return "take the additive inverse (negate mod p)"
        if "left_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"add {k} (mod {structure.p})"
        if "right_mul" in op_name and fixed_args:
            k = fixed_args[0]
            return f"add {k} to the result (mod {structure.p})"
        if "power" in op_name and fixed_args:
            k = fixed_args[0]
            return f"multiply by {k} (mod {structure.p})"
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
