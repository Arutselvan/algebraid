"""
Symmetric Group S_n - the group of all permutations of n elements.
"""

from typing import Any, List, Tuple, Optional
import random
import itertools
from .base import AlgebraicStructure

Permutation = Tuple[int, ...]


class SymmetricGroup(AlgebraicStructure):
    """The symmetric group S_n on n letters."""

    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError("n must be >= 2 for a non-trivial symmetric group")
        self.n = n
        self._elements: Optional[List[Permutation]] = None

    @property
    def name(self) -> str:
        return f"S_{self.n}"

    @property
    def description(self) -> str:
        return (
            f"The symmetric group S_{self.n}, the group of all permutations of "
            f"{{1, ..., {self.n}}}. Elements are written in one-line notation as tuples."
        )

    def elements(self) -> List[Permutation]:
        if self._elements is None:
            self._elements = list(itertools.permutations(range(1, self.n + 1)))
        return self._elements

    def random_element(self, rng: Optional[random.Random] = None) -> Permutation:
        r = rng or random
        p = list(range(1, self.n + 1))
        r.shuffle(p)
        return tuple(p)

    def identity(self) -> Permutation:
        return tuple(range(1, self.n + 1))

    def op(self, a: Permutation, b: Permutation) -> Permutation:
        """Composition of permutations: apply b then a (i.e., a∘b means apply b first)."""
        return tuple(a[i - 1] for i in b)

    def inverse(self, a: Permutation) -> Permutation:
        """Inverse of a permutation."""
        res = [0] * self.n
        for i, p in enumerate(a):
            res[p - 1] = i + 1
        return tuple(res)

    def element_to_str(self, a: Permutation) -> str:
        """Convert permutation to a tuple string, e.g. (2, 1, 3)."""
        return str(a)

    def operation_symbol(self) -> str:
        return "∘"

    def is_commutative(self) -> bool:
        return self.n < 3
