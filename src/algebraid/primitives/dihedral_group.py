"""
Dihedral Group D_n - the group of symmetries of a regular n-gon.

D_n has 2n elements: n rotations and n reflections. It is non-commutative
for n >= 3, and is naturally a semidirect product of Z_n and Z_2.

Elements are represented as (r, s) where:
- r in {0, 1, ..., n-1} is the rotation index
- s in {0, 1} indicates whether a reflection is applied (0 = no, 1 = yes)

The group operation follows: (r1, s1) * (r2, s2) = (r1 + (-1)^s1 * r2 mod n, s1 XOR s2)
"""

from typing import Any, List, Tuple, Optional, TYPE_CHECKING
import random
from .base import AlgebraicStructure

if TYPE_CHECKING:
    from random import Random
    from types import ModuleType

DihedralElement = Tuple[int, int]  # (rotation, reflection)

class DihedralGroup(AlgebraicStructure):
    """Dihedral group D_n of symmetries of a regular n-gon."""

    def __init__(self, n: int) -> None:
        if n < 3:
            raise ValueError("n must be >= 3 for a non-trivial dihedral group")
        self.n = n

    @property
    def name(self) -> str:
        return f"D_{self.n}"

    @property
    def description(self) -> str:
        return f"The dihedral group D_{self.n}, the group of symmetries of a regular {self.n}-gon. It has {2 * self.n} elements: {self.n} rotations and {self.n} reflections. Elements are written as (r, s) where r is the rotation index r in {{0, ..., {self.n - 1}}} and s is 0 (no reflection) or 1 (reflection)."

    @property
    def short_description(self) -> str:
        return f"(symmetries of a {self.n}-gon)"

    def elements(self) -> List[DihedralElement]:
        return [(r, s) for r in range(self.n) for s in range(2)]

    def random_element(self, rng: Optional[random.Random] = None) -> DihedralElement:
        r: "Random | ModuleType" = rng or random
        return (r.randint(0, self.n - 1), r.randint(0, 1))

    def identity(self) -> DihedralElement:
        return (0, 0)

    def op(self, a: DihedralElement, b: DihedralElement) -> DihedralElement:
        """
        Group operation:
        (r1, s1) * (r2, s2) = (r1 + (-1)^s1 * r2 mod n, s1 XOR s2)
        """
        r1, s1 = a
        r2, s2 = b
        if s1 == 0:  # No reflection: just add rotations
            new_r: int = (r1 + r2) % self.n
        else:
            new_r: int = (r1 - r2) % self.n
        new_s: int = s1 ^ s2
        return (new_r, new_s)

    def inverse(self, a: DihedralElement) -> DihedralElement:
        """
        Inverse:
        If s = 0: (r, 0)^(-1) = (-r mod n, 0)
        If s = 1: (r, 1)^(-1) = (r, 1)  (reflections are self-inverse)
        """
        r, s = a
        if s == 0:
            return ((-r) % self.n, 0)
        else:
            return (r, 1)

    def element_to_str(self, a: DihedralElement) -> str:
        r, s = a
        if r == 0 and s == 0:
            return "e"  # identity
        if s == 0:
            return f"r^{r}"
        if r == 0:
            return f"s"
        return f"r^{r}s"

    def operation_symbol(self) -> str:
        return "*"

    def is_commutative(self) -> bool:
        return False  # D_n is non-commutative for n >= 3

