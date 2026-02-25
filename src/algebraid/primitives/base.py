"""
Base classes for algebraic structures used in ALGEBRAID.

All algebraic structures inherit from AlgebraicStructure and must implement
a standard interface for element generation, operation execution, and
human-readable representation.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict, Optional
import random

class AlgebraicStructure(ABC):
    """Abstract base class for all algebraic structures."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the structure, e.g. 'Z_7'."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural language description of the structure."""
        pass

    @abstractmethod
    def elements(self) -> List[Any]:
        """Return all elements of the structure (for finite structures)."""
        pass

    @abstractmethod
    def random_element(self, rng: Optional[random.Random] = None) -> Any:
        """Return a uniformly random element."""
        pass

    @abstractmethod
    def identity(self) -> Any:
        """Return the identity element."""
        pass

    @abstractmethod
    def op(self, a: Any, b: Any) -> Any:
        """Apply the primary binary operation."""
        pass

    @abstractmethod
    def inverse(self, a: Any) -> Any:
        """Return the inverse of an element."""
        pass

    @abstractmethod
    def element_to_str(self, a: Any) -> str:
        """Human-readable string for an element."""
        pass

    @abstractmethod
    def operation_symbol(self) -> str:
        """Symbol for the primary operation, e.g. '+' or '*'."""
        pass

    def op_chain(self, *args: Any) -> Any:
        """Apply the operation left-to-right across multiple elements."""
        result = args[0]
        for a in args[1:]:
            result = self.op(result, a)
        return result

    def is_commutative(self) -> bool:
        """Check if the operation is commutative (by sampling)."""
        elems: list[Any] = self.elements()
        for _ in range(min(20, len(elems) ** 2)):
            a = random.choice(elems)
            b = random.choice(elems)
            if self.op(a, b) != self.op(b, a):
                return False
        return True

    def order(self) -> int:
        """Return the order (number of elements) of the structure."""
        return len(self.elements())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
