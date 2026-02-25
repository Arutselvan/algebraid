"""
Function-composition composer.

Builds complex unary operations by chaining simpler ones:
f(g(h(x))).  Each base operation is an ``AlgebraicOperation`` wrapping a
single algebraic primitive (inverse, left/right multiplication, conjugation,
or scalar power).  A ``ComposedFunction`` chains several of these into a
single callable with an optional execution trace.
"""

from typing import Any, Callable, List, Optional, Tuple
import random
from dataclasses import dataclass, field

from ..primitives.base import AlgebraicStructure


@dataclass
class AlgebraicOperation:
    """A named unary operation on an algebraic structure."""

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


# ── Standard operation factory ──────────────────────────────────────────────

def make_standard_operations(
    structure: AlgebraicStructure,
    rng: Optional[random.Random] = None,
) -> List[AlgebraicOperation]:
    """Return a standard set of unary operations for *structure*.

    Operations
    ----------
    inverse        x  ->  x^{-1}
    left_mul_c     x  ->  c * x
    right_mul_c    x  ->  x * c
    conj_c         x  ->  c * x * c^{-1}   (non-abelian groups only)
    power_k        x  ->  x * x * ... * x   (k times)
                   For additive groups this equals k*x mod n.
    """
    r = rng or random.Random()
    ops: List[AlgebraicOperation] = []

    # inverse
    ops.append(AlgebraicOperation(
        name="inverse",
        func=structure.inverse,
        arity=1,
        description=r.choice(["take the inverse", "find the inverse", "compute the inverse"]),
        symbol="^-1",
    ))

    # left multiplication by a fixed element
    c_l = structure.random_element(rng)
    s_l = structure.element_to_str(c_l)
    ops.append(AlgebraicOperation(
        name=f"left_mul_{s_l}",
        func=lambda x, c: structure.op(c, x),
        arity=1,
        description=r.choice([
            f"combine {s_l} with the current value (left: {s_l} {structure.operation_symbol()} x)",
            f"apply {s_l} on the left ({s_l} {structure.operation_symbol()} x)",
            f"compute {s_l} {structure.operation_symbol()} x",
        ]),
        symbol=f"{s_l} *",
        fixed_args=(c_l,),
    ))

    # right multiplication by a fixed element
    c_r = structure.random_element(rng)
    s_r = structure.element_to_str(c_r)
    ops.append(AlgebraicOperation(
        name=f"right_mul_{s_r}",
        func=structure.op,
        arity=1,
        description=r.choice([
            f"combine the current value with {s_r} (right: x {structure.operation_symbol()} {s_r})",
            f"apply {s_r} on the right (x {structure.operation_symbol()} {s_r})",
            f"compute x {structure.operation_symbol()} {s_r}",
        ]),
        symbol=f"* {s_r}",
        fixed_args=(c_r,),
    ))

    # conjugation (non-abelian groups only)
    from ..primitives.cyclic_group import CyclicGroup as _CG
    from ..primitives.finite_field import FiniteField as _FF

    if not isinstance(structure, (_CG, _FF)):
        c_c = structure.random_element(rng)
        s_c = structure.element_to_str(c_c)
        ops.append(AlgebraicOperation(
            name=f"conj_{s_c}",
            func=lambda x, c: structure.op_chain(c, x, structure.inverse(c)),
            arity=1,
            description=r.choice([
                f"conjugate by {s_c}: compute {s_c} * x * {s_c}\u207b\u00b9",
                f"apply conjugation {s_c} * x * {s_c}\u207b\u00b9",
            ]),
            symbol=f"{s_c} * _ * {s_c}^-1",
            fixed_args=(c_c,),
        ))

    # power (squaring or cubing)
    k = r.choice([2, 3])

    if isinstance(structure, _CG):
        mod = structure.n
        desc = r.choice([
            f"multiply the current value by {k} (compute {k}*x mod {mod})",
            f"scalar-multiply by {k} (compute {k} times the current value, mod {mod})",
        ])
    elif isinstance(structure, _FF):
        mod = structure.p
        desc = r.choice([
            f"multiply the current value by {k} (compute {k}*x mod {mod})",
            f"scalar-multiply by {k} (compute {k} times the current value, mod {mod})",
        ])
    else:
        desc = r.choice([
            f"compose the current element with itself {k} times (compute x^{k})",
            f"apply the current element {k} times in succession (x^{k})",
        ])

    ops.append(AlgebraicOperation(
        name=f"power_{k}",
        func=lambda x, p: structure.op_chain(*([x] * p)),
        arity=1,
        description=desc,
        symbol=f"^{k}",
        fixed_args=(k,),
    ))

    return ops


# ── Composed function ───────────────────────────────────────────────────────

class ComposedFunction:
    """A pipeline of ``AlgebraicOperation`` objects applied in sequence."""

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
        result = x
        for op in self.operations:
            result = op(result)
        return result

    def trace(self, x: Any) -> List[Tuple[str, Any]]:
        """Execute the chain and return ``[(op_name, intermediate_value), ...]``."""
        log: List[Tuple[str, Any]] = [("start", x)]
        result = x
        for op in self.operations:
            result = op(result)
            log.append((op.name, result))
        return log

    def __repr__(self) -> str:
        return f"ComposedFunction({self.name})"
