"""
Function Composition Composer - builds f(g(h(...))) chains.

This composer takes a base algebraic structure and a set of unary operations
on that structure (e.g., inverse, left-multiplication by a constant) and
builds new, more complex operations by chaining them together.
"""

from typing import Any, List, Tuple, Optional, Callable
import random
from dataclasses import dataclass, field
from ..primitives.base import AlgebraicStructure


@dataclass
class AlgebraicOperation:
    """A single named operation on an algebraic structure."""
    name: str
    func: Callable[..., Any]
    arity: int
    description: str
    symbol: str
    fixed_args: Tuple[Any, ...] = field(default_factory=tuple)

    def __call__(self, *args: Any) -> Any:
        return self.func(*args, *self.fixed_args)

    def __repr__(self) -> str:
        return self.name


def make_standard_operations(
    structure: AlgebraicStructure, rng: Optional[random.Random] = None
) -> List[AlgebraicOperation]:
    """
    Generate a standard set of unary operations for a given structure.
    """
    r: random.Random = rng or random.Random()
    ops: List[AlgebraicOperation] = []

    # 1. Inverse
    ops.append(
        AlgebraicOperation(
            name="inverse",
            func=structure.inverse,
            arity=1,
            description="take the inverse",
            symbol="^-1",
        )
    )

    # 2. Left-multiply by fixed element
    fixed_elem_l = structure.random_element(rng)
    ops.append(
        AlgebraicOperation(
            name=f"left_mul_{structure.element_to_str(fixed_elem_l)}",
            func=lambda x, c: structure.op(c, x),
            arity=1,
            description=f"left-multiply by {structure.element_to_str(fixed_elem_l)}",
            symbol=f"{structure.element_to_str(fixed_elem_l)} *",
            fixed_args=(fixed_elem_l,),
        )
    )

    # 3. Right-multiply by fixed element
    fixed_elem_r = structure.random_element(rng)
    ops.append(
        AlgebraicOperation(
            name=f"right_mul_{structure.element_to_str(fixed_elem_r)}",
            func=structure.op,
            arity=1,
            description=f"right-multiply by {structure.element_to_str(fixed_elem_r)}",
            symbol=f"* {structure.element_to_str(fixed_elem_r)}",
            fixed_args=(fixed_elem_r,),
        )
    )

    # 4. Conjugate by fixed element
    fixed_elem_c = structure.random_element(rng)
    ops.append(
        AlgebraicOperation(
            name=f"conj_{structure.element_to_str(fixed_elem_c)}",
            func=lambda x, c: structure.op_chain(c, x, structure.inverse(c)),
            arity=1,
            description=f"conjugate by {structure.element_to_str(fixed_elem_c)}",
            symbol=f"{structure.element_to_str(fixed_elem_c)} * _ * {structure.element_to_str(fixed_elem_c)}^-1",
            fixed_args=(fixed_elem_c,),
        )
    )

    # 5. Power (squaring, cubing)
    power = r.choice([2, 3])
    ops.append(
        AlgebraicOperation(
            name=f"power_{power}",
            func=lambda x, p: structure.op_chain(*([x] * p)),
            arity=1,
            description=f"raise to the power of {power}",
            symbol=f"^{power}",
            fixed_args=(power,),
        )
    )

    return ops


class ComposedFunction:
    """A function created by composing a list of AlgebraicOperations."""

    def __init__(self, operations: List[AlgebraicOperation], structure: AlgebraicStructure) -> None:
        self.operations = operations
        self.structure = structure

    @property
    def name(self) -> str:
        return " -> ".join(op.name for op in self.operations)

    @property
    def description(self) -> str:
        return " then ".join(op.description for op in self.operations)

    def __call__(self, x: Any) -> Any:
        """Execute the function chain: op_n(...op_2(op_1(x)))...)."""
        result = x
        for op in self.operations:
            result = op(result)
        return result

    def trace(self, x: Any) -> List[Tuple[str, Any]]:
        """Execute the chain and record intermediate results."""
        trace_log: List[Tuple[str, Any]] = [("start", x)]
        result = x
        for op in self.operations:
            result = op(result)
            trace_log.append((op.name, result))
        return trace_log

    def __repr__(self) -> str:
        return f"ComposedFunction({self.name})"
