# Function Composition Composer - builds f(g(h(...))) chains.

"""
This composer takes a base algebraic structure and a set of unary operations
on that structure (e.g., inverse, left-multiplication by a constant) and
builds new, more complex operations by chaining them together.

IMPORTANT SEMANTICS:
- For additive groups (Z_n, GF(p)): power_k(x) = x + x + ... + x (k times) = k*x mod n
  This is SCALAR MULTIPLICATION, not iterated addition of 1.
- For non-abelian groups (S_n, D_n): power_k(x) = x * x * ... * x (k times)
  This is repeated self-composition.
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

    Operations generated:
    1. inverse: x → x^(-1)
    2. left_mul_c: x → c * x  (for a random fixed element c)
    3. right_mul_c: x → x * c  (for a random fixed element c)
    4. conj_c: x → c * x * c^(-1)  (only for non-abelian groups)
    5. power_k: x → x^k = x * x * ... * x (k times)
       - For Z_n: this is k*x mod n (scalar multiplication)
       - For S_n: this is k-fold permutation composition
       - For D_n: this is k-fold symmetry composition
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
                f"combine {elem_str_l} with the current value (left operation: {elem_str_l} {structure.operation_symbol()} x)",
                f"apply {elem_str_l} on the left ({elem_str_l} {structure.operation_symbol()} x)",
                f"compute {elem_str_l} {structure.operation_symbol()} x",
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
                f"combine the current value with {elem_str_r} (right operation: x {structure.operation_symbol()} {elem_str_r})",
                f"apply {elem_str_r} on the right (x {structure.operation_symbol()} {elem_str_r})",
                f"compute x {structure.operation_symbol()} {elem_str_r}",
            ]),
            symbol=f"* {elem_str_r}",
            fixed_args=(fixed_elem_r,),
        )
    )

    # 4. Conjugate by fixed element — only meaningful for non-abelian structures
    from ..primitives.cyclic_group import CyclicGroup as _CG
    from ..primitives.finite_field import FiniteField as _FF
    if not isinstance(structure, (_CG, _FF)):
        fixed_elem_c = structure.random_element(rng)
        elem_str_c = structure.element_to_str(fixed_elem_c)
        ops.append(
            AlgebraicOperation(
                name=f"conj_{elem_str_c}",
                func=lambda x, c: structure.op_chain(c, x, structure.inverse(c)),
                arity=1,
                description=r.choice([
                    f"conjugate by {elem_str_c}: compute {elem_str_c} * x * {elem_str_c}\u207b\u00b9",
                    f"apply the conjugation {elem_str_c} * x * {elem_str_c}\u207b\u00b9",
                ]),
                symbol=f"{elem_str_c} * _ * {elem_str_c}^-1",
                fixed_args=(fixed_elem_c,),
            )
        )

    # 5. Power (squaring, cubing) — x^k = x * x * ... * x (k times)
    power = r.choice([2, 3])
    # Determine accurate description based on structure type
    if isinstance(structure, _CG):
        power_desc = r.choice([
            f"multiply the current value by {power} (compute {power}*x mod {structure.n})",
            f"scalar multiply by {power}: compute {power} times the current value (mod {structure.n})",
        ])
    elif isinstance(structure, _FF):
        power_desc = r.choice([
            f"multiply the current value by {power} (compute {power}*x mod {structure.p})",
            f"scalar multiply by {power}: compute {power} times the current value (mod {structure.p})",
        ])
    else:
        power_desc = r.choice([
            f"compose the current element with itself {power} times (compute x^{power})",
            f"apply the current element {power} times in succession (x^{power})",
        ])
    ops.append(
        AlgebraicOperation(
            name=f"power_{power}",
            func=lambda x, p: structure.op_chain(*([x] * p)),
            arity=1,
            description=power_desc,
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
