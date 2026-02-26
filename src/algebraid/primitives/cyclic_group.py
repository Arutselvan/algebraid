"""
Cyclic Group Z_n - the integers modulo n under addition.

This is the simplest and most fundamental finite group, serving as the
building block for many ALGEBRAID tasks.
"""

from typing import Any, List, Optional, TYPE_CHECKING
import random
from .base import AlgebraicStructure

if TYPE_CHECKING:
    from random import Random
    from types import ModuleType

class CyclicGroup(AlgebraicStructure):
    """The cyclic group Z_n = {0, 1, ..., n-1} under addition mod n."""

    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError("Order n must be >= 2")
        self.n = n

    @property
    def name(self) -> str:
        return f"Z_{self.n}"

    @property
    def description(self) -> str:
        return f"The cyclic group of order {self.n}, consisting of integers {{0, 1, ..., {self.n - 1}}} under addition modulo {self.n}."

    @property
    def short_description(self) -> str:
        return f"(the integers mod {self.n})"

    def elements(self) -> List[int]:
        return list(range(self.n))

    def random_element(self, rng: Optional[random.Random] = None) -> int:
        r: "Random | ModuleType" = rng or random
        return r.randint(0, self.n - 1)

    def identity(self) -> int:
        return 0

    def op(self, a: int, b: int) -> int:
        """Addition modulo n."""
        return (a + b) % self.n

    def inverse(self, a: int) -> int:
        """Additive inverse: -a mod n."""
        return (-a) % self.n

    def multiply(self, k: int, a: int) -> int:
        """Scalar multiplication: k * a mod n."""
        return (k * a) % self.n

    def element_to_str(self, a: int) -> str:
        return str(a)

    def operation_symbol(self) -> str:
        return "+"

    def is_commutative(self) -> bool:
        return True  # Always commutative

