"""
Finite Field GF(p) - the field of integers modulo a prime p.

A field supports both addition and multiplication, along with their inverses.
This structure enables richer compositional tasks involving two interacting
operations (ring/field axioms).
"""

from typing import Any, List, Optional, TYPE_CHECKING
import random
from .base import AlgebraicStructure

if TYPE_CHECKING:
    from random import Random
    from types import ModuleType

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _mod_inverse(a: int, p: int) -> int:
    """Compute modular multiplicative inverse using extended Euclidean algorithm."""
    if a % p == 0:
        raise ValueError(f"{a} has no multiplicative inverse mod {p}")
    g, x, _ = _extended_gcd(a % p, p)
    if g != 1:
        raise ValueError(f"{a} has no multiplicative inverse mod {p}")
    return x % p

def _extended_gcd(a: int, b: int) -> tuple:
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x

class FiniteField(AlgebraicStructure):
    """The finite field GF(p) for a prime p."""

    def __init__(self, p: int) -> None:
        if not _is_prime(p):
            raise ValueError(f"{p} is not a prime number")
        self.p = p

    @property
    def name(self) -> str:
        return f"GF({self.p})"

    @property
    def description(self) -> str:
        return (
            f"The finite field GF({self.p}), consisting of integers "
            f"{{0, 1, ..., {self.p - 1}}} with addition and multiplication "
            f"modulo {self.p}."
        )

    def elements(self) -> List[int]:
        return list(range(self.p))

    def nonzero_elements(self) -> List[int]:
        return list(range(1, self.p))

    def random_element(self, rng: Optional[random.Random] = None) -> int:
        r: "Random | ModuleType" = rng or random
        return r.randint(0, self.p - 1)

    def random_nonzero(self, rng: Optional[random.Random] = None) -> int:
        r: "Random | ModuleType" = rng or random
        return r.randint(1, self.p - 1)

    def identity(self) -> int:
        """Additive identity."""
        return 0

    def multiplicative_identity(self) -> int:
        return 1

    def op(self, a: int, b: int) -> int:
        """Addition modulo p."""
        return (a + b) % self.p

    def mul(self, a: int, b: int) -> int:
        """Multiplication modulo p."""
        return (a * b) % self.p

    def inverse(self, a: int) -> int:
        """Additive inverse: -a mod p."""
        return (-a) % self.p

    def mul_inverse(self, a: int) -> int:
        """Multiplicative inverse of a nonzero element."""
        return _mod_inverse(a, self.p)

    def power(self, a: int, b: int) -> int:
        """Compute a^b mod p."""
        return pow(a, b, self.p)

    def subtract(self, a: int, b: int) -> int:
        """Subtraction: a - b mod p."""
        return (a - b) % self.p

    def divide(self, a: int, b: int) -> int:
        """Division: a / b = a * b^(-1) mod p."""
        return (a * self.mul_inverse(b)) % self.p

    def element_to_str(self, a: int) -> str:
        return str(a)

    def operation_symbol(self) -> str:
        return "+"

    def is_commutative(self) -> bool:
        return True

