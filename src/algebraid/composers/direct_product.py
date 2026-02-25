"""
Direct product G x H of two algebraic structures.

Elements are pairs (g, h) with component-wise operations.
"""

from typing import Any, List, Tuple, Optional
import random
from ..primitives.base import AlgebraicStructure

class DirectProduct(AlgebraicStructure):
    """
    Direct product G x H of two algebraic structures.

    Elements are tuples (g, h) where g ∈ G and h ∈ H.
    Operations are applied component-wise.
    """

    def __init__(self, G: AlgebraicStructure, H: AlgebraicStructure) -> None:
        self.G = G
        self.H = H

    @property
    def name(self) -> str:
        return f"{self.G.name} x {self.H.name}"

    @property
    def description(self) -> str:
        return (
            f"The direct product of {self.G.name} and {self.H.name}. "
            f"Elements are pairs (a, b) where a ∈ {self.G.name} and b ∈ {self.H.name}. "
            f"The operation is applied component-wise: "
            f"(a1, b1) {self.G.operation_symbol()} (a2, b2) = "
            f"(a1 {self.G.operation_symbol()} a2, b1 {self.H.operation_symbol()} b2)."
        )

    @property
    def short_description(self) -> str:
        return f"(product of {self.G.name} and {self.H.name})"

    def elements(self) -> List[Tuple[Any, Any]]:
        return [(g, h) for g in self.G.elements() for h in self.H.elements()]

    def random_element(self, rng: Optional[random.Random] = None) -> Tuple[Any, Any]:
        return (self.G.random_element(rng), self.H.random_element(rng))

    def identity(self) -> Tuple[Any, Any]:
        return (self.G.identity(), self.H.identity())

    def op(self, a: Tuple[Any, Any], b: Tuple[Any, Any]) -> Tuple[Any, Any]:
        """Component-wise operation."""
        return (self.G.op(a[0], b[0]), self.H.op(a[1], b[1]))

    def inverse(self, a: Tuple[Any, Any]) -> Tuple[Any, Any]:
        """Component-wise inverse."""
        return (self.G.inverse(a[0]), self.H.inverse(a[1]))

    def element_to_str(self, a: Tuple[Any, Any]) -> str:
        return f"({self.G.element_to_str(a[0])}, {self.H.element_to_str(a[1])})"

    def operation_symbol(self) -> str:
        return "."

    def is_commutative(self) -> bool:
        return self.G.is_commutative() and self.H.is_commutative()

    @property
    def composition_depth(self) -> int:
        """Track how deeply composed this structure is."""
        g_depth: int | Any = getattr(self.G, 'composition_depth', 0)
        h_depth: int | Any = getattr(self.H, 'composition_depth', 0)
        return max(g_depth, h_depth) + 1
