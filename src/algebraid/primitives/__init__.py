"""Algebraic primitives: Z_n, S_n, D_n, GF(p), and Q_8."""

from .cyclic_group import CyclicGroup
from .symmetric_group import SymmetricGroup
from .dihedral_group import DihedralGroup
from .finite_field import FiniteField
from .quaternion_group import QuaternionGroup
from .base import AlgebraicStructure

__all__ = [
    "CyclicGroup",
    "SymmetricGroup",
    "DihedralGroup",
    "FiniteField",
    "QuaternionGroup",
    "AlgebraicStructure",
]
