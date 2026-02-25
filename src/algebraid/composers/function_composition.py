# Function Composition Composer - builds f(g(h(...))) chains.

"""
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
            description=r.choice(["take the inverse", "find the inverse", "compute the inverse"]),
            symbol="^-1",
        )
    )

    # 2. Left-multiply by fixed element
    fixed_elem_l = structure.random_element(rng)
    elem_str_l = structure.element_to_str(fixed_elem_l)
    ops.append(
        AlgebraicOperation(
            name=f"left_mul_{elem_str_l}",
            func=lambda x, c: structure.op(c, x),
            arity=1,
            description=r.choice([
                f"combine {elem_str_l} with the current value (in that order)",
                f"apply {elem_str_l} before the current value",
                f"prepend {elem_str_l} using the group operation",
            ]),
            symbol=f"{elem_str_l} *",
            fixed_args=(fixed_elem_l,),
        )
    )

    # 3. Right-multiply by fixed element
    fixed_elem_r = structure.random_element(rng)
    elem_str_r = structure.element_to_str(fixed_elem_r)
    ops.append(
        AlgebraicOperation(
            name=f"right_mul_{elem_str_r}",
            func=structure.op,
            arity=1,
            description=r.choice([
                f"combine the current value with {elem_str_r} (in that order)",
                f"apply {elem_str_r} after the current value",
                f"append {elem_str_r} using the group operation",
            ]),
            symbol=f"* {elem_str_r}",
            fixed_args=(fixed_elem_r,),
        )
    )

    # 4. Conjugate by fixed element
    fixed_elem_c = structure.random_element(rng)
    elem_str_c = structure.element_to_str(fixed_elem_c)
    ops.append(
        AlgebraicOperation(
            name=f"conj_{elem_str_c}",
            func=lambda x, c: structure.op_chain(c, x, structure.inverse(c)),
            arity=1,
            description=r.choice([
                f"sandwich the current value between {elem_str_c} and its inverse",
                f"apply the transformation {elem_str_c} * x * {elem_str_c}\u207b\u00b9",
                f"wrap the current value with {elem_str_c} on both sides",
            ]),
            symbol=f"{elem_str_c} * _ * {elem_str_c}^-1",
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
            description=r.choice([
                f"raise to the power of {power}",
                f"apply the power {power}",
                f"repeat the operation {power} times",
            ]),
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
