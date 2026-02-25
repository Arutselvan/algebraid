"""
ALGEBRAID Algebraic Complexity Metrics.

Four native metrics that only ALGEBRAID can compute because of its algebraic foundation:
- H_alg: Algebraic Entropy - information-theoretic difficulty based on group order
- D_comm: Commutativity Distance - fraction of consecutive op pairs that are order-dependent
- O_c: Orbit Complexity - fraction of the group's elements visited by the computation trace
- I_s: Structural Interference - shared divisors between component group orders in direct products
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple, NamedTuple

from .task_model import Task, TaskFamily


class AlgebraicComplexity(NamedTuple):
    """Full complexity profile for a single ALGEBRAID task."""
    algebraic_entropy: float
    commutativity_distance: float
    orbit_complexity: float
    structural_interference: float

    def composite(self) -> float:
        """Composite score (geometric mean of the four components)."""
        vals = [
            max(self.algebraic_entropy, 1e-9),
            max(self.commutativity_distance, 1e-9),
            max(self.orbit_complexity, 1e-9),
            max(self.structural_interference, 1e-9),
        ]
        return (vals[0] * vals[1] * vals[2] * vals[3]) ** 0.25


def _parse_order(s_name: str) -> int:
    """Parse the group order from a structure name string."""
    if m := re.match(r"Z_(\d+)", s_name):
        return int(m.group(1))
    if m := re.match(r"S_(\d+)", s_name):
        return math.factorial(int(m.group(1)))
    if m := re.match(r"D_(\d+)", s_name):
        return 2 * int(m.group(1))
    if m := re.match(r"GF\((\d+)\)", s_name):
        return int(m.group(1))
    # Handle relabeled structures
    if m := re.match(r"Relabeled Z_(\d+)", s_name):
        return int(m.group(1))
    return 0


def _is_commutative_name(s_name: str) -> bool:
    """Determine if a structure is commutative from its name."""
    return bool(re.match(r"Z_\d+", s_name) or re.match(r"GF\(\d+\)", s_name))


def compute_algebraic_entropy(task: Task) -> float:
    """H_alg = sum of log2(|G_i|) over non-trivial operations.

    Measures the information-theoretic difficulty: how large is the search
    space that the model must navigate? For intra-structure tasks, this is
    log2(|G|) * depth. For inter-structure tasks, it's the sum of logs of
    component group orders.
    """
    if not task.structures:
        return 0.0

    if task.family == TaskFamily.INTER_STRUCTURE:
        total = sum(_parse_order(s) for s in task.structures)
        return math.log2(total) if total > 1 else 0.0

    order = _parse_order(task.structures[0])
    if order <= 1:
        return 0.0

    # Scale by depth (number of operations)
    return math.log2(order) * task.depth


def compute_commutativity_distance(task: Task) -> float:
    """D_comm = fraction of consecutive operation pairs where order matters.

    For commutative groups, all pairs commute (D_comm = 0).
    For non-commutative groups, we estimate based on the structure.
    For inter-structure tasks with mixed commutativity, we compute the
    fraction of non-commutative components.
    """
    if not task.structures:
        return 0.0

    if task.family == TaskFamily.INTER_STRUCTURE:
        non_comm = sum(1 for s in task.structures if not _is_commutative_name(s))
        return non_comm / len(task.structures) if task.structures else 0.0

    # For intra-structure tasks with a chain of ops
    is_comm = _is_commutative_name(task.structures[0])
    if is_comm:
        return 0.0

    # Non-commutative structure: estimate based on depth
    # At depth 1, there are 0 consecutive pairs; at depth d, there are d-1 pairs
    if task.depth <= 1:
        return 0.0
    # In a non-commutative group, we conservatively assume all pairs are order-dependent
    return 1.0


def compute_orbit_complexity(task: Task) -> float:
    """O_c = |distinct intermediate values| / |G|.

    Measures how much of the group's structure the computation visits.
    A higher value means the model must track more of the group.
    Requires a solution_trace to be meaningful.
    """
    if not task.solution_trace or len(task.solution_trace) < 2:
        return 0.0
    if not task.structures:
        return 0.0

    order = _parse_order(task.structures[0])
    if order <= 0:
        return 0.0

    # Count distinct intermediate element values
    try:
        distinct = set()
        for _, val in task.solution_trace:
            # Handle tuples (for direct products) and simple values
            if isinstance(val, (list, tuple)):
                distinct.add(str(val))
            else:
                distinct.add(val)
        return min(len(distinct) / order, 1.0)
    except Exception:
        return 0.0


def compute_structural_interference(task: Task) -> float:
    """I_s = shared divisors between component group orders.

    For direct-product (inter-structure) tasks, measures whether the component
    groups share divisors, which could cause the model to confuse elements
    across components.

    I_s = 1 - (1 / gcd(|G|, |H|)) for two-component products.
    Higher value = more potential for confusion.
    """
    if task.family != TaskFamily.INTER_STRUCTURE or len(task.structures) < 2:
        return 0.0

    orders = [_parse_order(s) for s in task.structures if _parse_order(s) > 0]
    if len(orders) < 2:
        return 0.0

    g = math.gcd(orders[0], orders[1])
    # Normalize: gcd=1 means no interference (0.0), gcd=order means full overlap (1.0)
    max_possible_gcd = min(orders[0], orders[1])
    return (g - 1) / (max_possible_gcd - 1) if max_possible_gcd > 1 else 0.0


def compute_complexity(task: Task) -> AlgebraicComplexity:
    """Compute the full algebraic complexity profile for a task."""
    return AlgebraicComplexity(
        algebraic_entropy=compute_algebraic_entropy(task),
        commutativity_distance=compute_commutativity_distance(task),
        orbit_complexity=compute_orbit_complexity(task),
        structural_interference=compute_structural_interference(task),
    )
