"""
Train/test split utilities for generalisation studies.

Addresses paper Blocker 4 (No Generalisation Study) by providing
principled splits that let researchers test compositional generalisation
along four axes:

  split_by_depth()         Train depth <= k, test depth >= k+gap (default gap=2)
  split_by_commutativity() Train on abelian (Z_n, GF(p)), test on non-abelian
  split_by_structure()     Train on structure-name prefixes, test on others
  split_by_family()        Train on task families, test on others

Each function returns a (train: TaskSet, test: TaskSet) pair.

Example research questions these splits answer
-----------------------------------------------
  "Does the model generalise to longer chains unseen during training?"
      -> split_by_depth(task_set, train_max_depth=2)

  "Does abelian training transfer to non-abelian structures?"
      -> split_by_commutativity(task_set)

  "Does training on Z_5 and Z_7 transfer to Z_11?"
      -> split_by_structure(ts, train_prefixes=["Z_5","Z_7"], test_prefixes=["Z_11"])

  "Does intra-structure training transfer to inter-structure tasks?"
      -> split_by_family(ts, ["intra"], ["inter"])
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .task_model import Task, TaskSet


# -- Helpers ------------------------------------------------------------------

#: Name prefixes whose structures are provably abelian.
_ABELIAN_PREFIXES: Set[str] = {"Z_", "GF("}

#: Name prefixes / full names whose structures are non-abelian.
_NONABELIAN_PREFIXES: Set[str] = {"S_", "D_", "Q_8"}


def _is_abelian_name(name: str) -> bool:
    return any(name.startswith(p) for p in _ABELIAN_PREFIXES)


def _family_value(task: Task) -> str:
    return task.family.value if hasattr(task.family, "value") else str(task.family)


def _dimension_value(task: Task) -> str:
    return task.dimension.value if hasattr(task.dimension, "value") else str(task.dimension)


def _make_pair(
    train_tasks: List[Task],
    test_tasks: List[Task],
    base_name: str,
) -> Tuple[TaskSet, TaskSet]:
    train = TaskSet(
        tasks=train_tasks,
        name=f"{base_name}_train",
        description=f"Train split of '{base_name}' ({len(train_tasks)} tasks)",
    )
    test = TaskSet(
        tasks=test_tasks,
        name=f"{base_name}_test",
        description=f"Test split of '{base_name}' ({len(test_tasks)} tasks)",
    )
    return train, test


# -- 1. Depth split -----------------------------------------------------------

def split_by_depth(
    task_set: TaskSet,
    train_max_depth: int,
    test_min_depth: Optional[int] = None,
) -> Tuple[TaskSet, TaskSet]:
    """
    Split by composition depth.

    Train : depth <= train_max_depth
    Test  : depth >= test_min_depth  (default: train_max_depth + 2)

    The gap between the two thresholds acts as an extrapolation buffer.
    Tasks at depth train_max_depth+1 are excluded from both sets so the
    test distribution strictly exceeds the training distribution.

    Parameters
    ----------
    task_set          Source TaskSet.
    train_max_depth   Maximum depth included in the training set.
    test_min_depth    Minimum depth included in the test set.
                      Defaults to train_max_depth + 2.
    """
    if test_min_depth is None:
        test_min_depth = train_max_depth + 2

    train_tasks = [t for t in task_set if t.depth <= train_max_depth]
    test_tasks = [t for t in task_set if t.depth >= test_min_depth]
    return _make_pair(train_tasks, test_tasks, task_set.name)


# -- 2. Commutativity split ---------------------------------------------------

def split_by_commutativity(task_set: TaskSet) -> Tuple[TaskSet, TaskSet]:
    """
    Split by commutativity of the primary algebraic structure.

    Train (abelian)     : tasks on Z_n or GF(p)
    Test  (non-abelian) : tasks on S_n, D_n, or Q_8

    Tasks involving direct products are placed in the test set because
    they may include non-abelian factors.  Tasks with no structure info
    go to the test set (conservative assignment).

    Research question
    -----------------
    "Does learning on commutative structures transfer to non-commutative ones?"
    """
    def _is_abelian_task(task: Task) -> bool:
        if not task.structures:
            return False
        return all(_is_abelian_name(s) for s in task.structures)

    train_tasks = [t for t in task_set if _is_abelian_task(t)]
    test_tasks = [t for t in task_set if not _is_abelian_task(t)]
    return _make_pair(train_tasks, test_tasks, task_set.name)


# -- 3. Structure-name split --------------------------------------------------

def split_by_structure(
    task_set: TaskSet,
    train_prefixes: List[str],
    test_prefixes: List[str],
) -> Tuple[TaskSet, TaskSet]:
    """
    Split by structure name prefix (or exact name).

    Tasks whose structures match *train_prefixes* go to train;
    those matching *test_prefixes* go to test.
    Tasks matching neither list are excluded from both.

    Parameters
    ----------
    train_prefixes  List of name prefixes / exact names for training structures.
                    E.g. ["Z_5", "Z_7"]  or  ["Z_"]  (all cyclic groups).
    test_prefixes   List of name prefixes / exact names for test structures.
                    E.g. ["Z_11"]  or  ["S_", "D_", "Q_8"].

    Research questions
    ------------------
    "Train on Z_5, Z_7 -> test on Z_11":
        split_by_structure(ts, ["Z_5","Z_7"], ["Z_11"])

    "Train on cyclic -> test on symmetric/dihedral/quaternion":
        split_by_structure(ts, ["Z_"], ["S_","D_","Q_8"])
    """
    def _matches(task: Task, prefixes: List[str]) -> bool:
        if not task.structures:
            return False
        return all(
            any(s.startswith(p) or s == p for p in prefixes)
            for s in task.structures
        )

    train_tasks = [t for t in task_set if _matches(t, train_prefixes)]
    test_tasks = [t for t in task_set if _matches(t, test_prefixes)]
    return _make_pair(train_tasks, test_tasks, task_set.name)


# -- 4. Task-family split -----------------------------------------------------

#: Short-hand keys accepted in train_families / test_families.
_FAMILY_SHORT: dict = {
    "intra":        "intra-structure composition",
    "inter":        "inter-structure composition",
    "field":        "field arithmetic",
    "rule":         "rule induction",
    "conceptual":   "conceptual query",
    # adversarial and intermediate share the intra-structure family value;
    # use the dimension key to identify them.
}

_DIMENSION_SHORT: dict = {
    "adversarial":    "adversarial",
    "intermediate":   "intermediate_state",
}


def split_by_family(
    task_set: TaskSet,
    train_families: List[str],
    test_families: List[str],
) -> Tuple[TaskSet, TaskSet]:
    """
    Split by task family (or compositionality dimension).

    Accepted keys: "intra", "inter", "field", "rule", "conceptual",
                   "adversarial", "intermediate"
    or the full family value strings from TaskFamily enum.

    Research question
    -----------------
    "Train on intra-structure chains -> test on inter-structure chains":
        split_by_family(ts, ["intra"], ["inter"])

    "Train on non-adversarial -> test on adversarial traps":
        split_by_family(ts, ["intra","inter","field","rule","conceptual"],
                            ["adversarial"])
    """
    def _expand(names: List[str]) -> Tuple[Set[str], Set[str]]:
        """Return (family_values, dimension_values) sets."""
        fam_vals: Set[str] = set()
        dim_vals: Set[str] = set()
        for n in names:
            if n in _DIMENSION_SHORT:
                dim_vals.add(_DIMENSION_SHORT[n])
            else:
                fam_vals.add(_FAMILY_SHORT.get(n, n))
        return fam_vals, dim_vals

    train_fams, train_dims = _expand(train_families)
    test_fams, test_dims = _expand(test_families)

    def _in_group(task: Task, fams: Set[str], dims: Set[str]) -> bool:
        fv = _family_value(task)
        dv = _dimension_value(task)
        return fv in fams or dv in dims

    train_tasks = [t for t in task_set if _in_group(t, train_fams, train_dims)]
    test_tasks = [t for t in task_set if _in_group(t, test_fams, test_dims)]
    return _make_pair(train_tasks, test_tasks, task_set.name)


# -- Convenience --------------------------------------------------------------

def split_summary(train: TaskSet, test: TaskSet) -> str:
    """Return a compact multi-line split summary string."""
    def _depth_range(ts: TaskSet) -> str:
        if not ts:
            return "-"
        depths = sorted({t.depth for t in ts})
        return f"{depths[0]}-{depths[-1]}" if len(depths) > 1 else str(depths[0])

    def _structure_counts(ts: TaskSet) -> str:
        from collections import Counter
        cnt: Counter = Counter()
        for t in ts:
            for s in (t.structures or []):
                cnt[s] += 1
        top = cnt.most_common(3)
        return ", ".join(f"{s}x{n}" for s, n in top) or "-"

    lines = [
        "Split summary",
        f"  Train : {len(train):>5} tasks | depths {_depth_range(train)}"
        f" | top structures: {_structure_counts(train)}",
        f"  Test  : {len(test):>5} tasks | depths {_depth_range(test)}"
        f" | top structures: {_structure_counts(test)}",
    ]
    return "\n".join(lines)
