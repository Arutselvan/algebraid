"""
Quaternion Group Q_8 - the 8-element non-abelian group of unit quaternions.

Elements: {1, -1, i, -i, j, -j, k, -k}
Multiplication rules: i^2=j^2=k^2=-1, ij=k, ji=-k, jk=i, kj=-i, ki=j, ik=-j

Internal representation: integer indices 0-7
    0: "1",  1: "-1",  2: "i",  3: "-i",  4: "j",  5: "-j",  6: "k",  7: "-k"
"""

from typing import Any, List, Optional
import random
from .base import AlgebraicStructure

# Human-readable names for each element index
_NAMES = ["1", "-1", "i", "-i", "j", "-j", "k", "-k"]

# Precomputed multiplication table: MUL_TABLE[a][b] = a * b (as index)
# Derived from quaternion algebra: i^2=j^2=k^2=ijk=-1
_MUL_TABLE = [
    # a=0  (+1, identity): 1*b = b
    [0, 1, 2, 3, 4, 5, 6, 7],
    # a=1  (-1): (-1)*b negates sign
    [1, 0, 3, 2, 5, 4, 7, 6],
    # a=2  (+i): i*[1,-1,i,-i,j,-j,k,-k] = [i,-i,-1,1,k,-k,-j,j]
    [2, 3, 1, 0, 6, 7, 5, 4],
    # a=3  (-i): (-i)*b = negation of row 2
    [3, 2, 0, 1, 7, 6, 4, 5],
    # a=4  (+j): j*[1,-1,i,-i,j,-j,k,-k] = [j,-j,-k,k,-1,1,i,-i]
    [4, 5, 7, 6, 1, 0, 2, 3],
    # a=5  (-j): (-j)*b = negation of row 4
    [5, 4, 6, 7, 0, 1, 3, 2],
    # a=6  (+k): k*[1,-1,i,-i,j,-j,k,-k] = [k,-k,j,-j,-i,i,-1,1]
    [6, 7, 4, 5, 3, 2, 1, 0],
    # a=7  (-k): (-k)*b = negation of row 6
    [7, 6, 5, 4, 2, 3, 0, 1],
]

# Inverse table: INV_TABLE[a] = a^{-1}
# 1^-1=1, (-1)^-1=-1, i^-1=-i, (-i)^-1=i, j^-1=-j, (-j)^-1=j, k^-1=-k, (-k)^-1=k
_INV_TABLE = [0, 1, 3, 2, 5, 4, 7, 6]


class QuaternionGroup(AlgebraicStructure):
    """The quaternion group Q_8 = {+/-1, +/-i, +/-j, +/-k} under quaternion multiplication.

    This is the unique non-abelian group of order 8 with a single element of
    order 2 (namely -1). It is non-abelian: e.g. i*j = k but j*i = -k.
    Every non-identity element has order 4, except -1 which has order 2.
    """

    @property
    def name(self) -> str:
        return "Q_8"

    @property
    def description(self) -> str:
        return (
            "The quaternion group of order 8, with elements {1, -1, i, -i, j, -j, k, -k} "
            "under quaternion multiplication (i^2=j^2=k^2=-1, ij=k, ji=-k)."
        )

    @property
    def short_description(self) -> str:
        return "(the quaternion group: +/-1, +/-i, +/-j, +/-k)"

    def elements(self) -> List[int]:
        return list(range(8))

    def random_element(self, rng: Optional[random.Random] = None) -> int:
        r = rng or random
        return r.randint(0, 7)

    def identity(self) -> int:
        return 0  # represents "1"

    def op(self, a: int, b: int) -> int:
        """Quaternion multiplication."""
        return _MUL_TABLE[a][b]

    def inverse(self, a: int) -> int:
        """Quaternion inverse."""
        return _INV_TABLE[a]

    def element_to_str(self, a: int) -> str:
        return _NAMES[a]

    def operation_symbol(self) -> str:
        return "*"

    def is_commutative(self) -> bool:
        return False  # Q_8 is non-abelian
