"""
COMPEVAL Task Generator.

Procedurally generates tasks in four families:
- Intra-structure: chain d operations on one group element
- Inter-structure: apply operations to tuples across d+1 nested groups
- Field arithmetic: evaluate a depth-d expression tree in GF(p)
- Rule induction: infer a hidden d-step chain from input-output examples
"""

import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .primitives import CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField
from .primitives.base import AlgebraicStructure
from .composers import DirectProduct, AlgebraicOperation, ComposedFunction, make_standard_operations
from .task_model import Task, TaskFamily, CompositionDimension, TaskSet
from .tasks.verbalizer import Verbalizer


def _task_id(seed, family, depth, idx):
    raw = f"{seed}-{family}-{depth}-{idx}"
    return "CE-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]


def _random_cyclic(rng, min_n=3, max_n=18) -> CyclicGroup:
    return CyclicGroup(rng.randint(min_n, max_n))


def _random_symmetric(rng, min_n=3, max_n=5) -> SymmetricGroup:
    return SymmetricGroup(rng.randint(min_n, max_n))


def _random_dihedral(rng, min_n=3, max_n=8) -> DihedralGroup:
    return DihedralGroup(rng.randint(min_n, max_n))


def _random_field(rng) -> FiniteField:
    return FiniteField(rng.choice(SMALL_PRIMES[1:]))


def _random_structure(rng) -> AlgebraicStructure:
    choice = rng.randint(0, 3)
    if choice == 0:
        return _random_cyclic(rng)
    elif choice == 1:
        return _random_symmetric(rng)
    elif choice == 2:
        return _random_dihedral(rng)
    else:
        return _random_field(rng)


def _build_field_expression(field, depth, rng):
    if depth == 0:
        val = field.random_element(rng)
        return str(val), val

    op = rng.choice(["+", "*"])
    left_depth = rng.randint(0, depth - 1)
    right_depth = depth - 1 - left_depth

    left_expr, left_val = _build_field_expression(field, left_depth, rng)
    right_expr, right_val = _build_field_expression(field, right_depth, rng)

    if op == "+":
        val = field.op(left_val, right_val)
    else:
        val = field.mul(left_val, right_val)

    return f"({left_expr} {op} {right_expr})", val


def _generate_intra_structure_task(rng, depth, idx, seed, verbalizer, dimension=CompositionDimension.GENERAL):
    structure = _random_structure(rng)
    all_ops = make_standard_operations(structure, rng)
    chosen_ops = [rng.choice(all_ops) for _ in range(depth)]
    chain = ComposedFunction(chosen_ops, structure)
    x = structure.random_element(rng)
    answer_raw = chain(x)
    answer_str = structure.element_to_str(answer_raw)
    trace = chain.trace(x)
    prompt = verbalizer.verbalize_intra(structure, chain, x)
    return Task(
        task_id=_task_id(seed, "intra", depth, idx),
        prompt=prompt,
        answer=answer_str,
        answer_raw=answer_str,  # store as string for JSON serialization
        depth=depth,
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=dimension,
        structures=[structure.name],
        metadata={"ops": [op.name for op in chosen_ops]},
        solution_trace=[(op_name, structure.element_to_str(val)) for op_name, val in trace],
    )


def _generate_inter_structure_task(rng, depth, idx, seed, verbalizer, dimension=CompositionDimension.GENERAL):
    structures = [_random_cyclic(rng, 2, 7) for _ in range(depth + 1)]
    composed = structures[0]
    for s in structures[1:]:
        composed = DirectProduct(composed, s)

    op_type = rng.choice(["op", "inverse", "op_then_inverse"])
    a = composed.random_element(rng)

    if op_type == "inverse":
        answer_raw = composed.inverse(a)
        prompt = verbalizer.verbalize_inter(composed, a, op_type="inverse")
    else:
        b = composed.random_element(rng)
        if op_type == "op":
            answer_raw = composed.op(a, b)
            prompt = verbalizer.verbalize_inter(composed, a, b)
        else:
            c = composed.op(a, b)
            answer_raw = composed.inverse(c)
            prompt = verbalizer.verbalize_inter(composed, a, b, op_type="op_then_inverse")

    return Task(
        task_id=_task_id(seed, "inter", depth, idx),
        prompt=prompt,
        answer=composed.element_to_str(answer_raw),
        answer_raw=composed.element_to_str(answer_raw),
        depth=depth,
        family=TaskFamily.INTER_STRUCTURE,
        dimension=dimension,
        structures=[s.name for s in structures],
    )


def _generate_field_arithmetic_task(rng, depth, idx, seed, verbalizer, dimension=CompositionDimension.GENERAL):
    field = _random_field(rng)
    expr_str, answer_raw = _build_field_expression(field, depth, rng)
    prompt = verbalizer.verbalize_field(field, expr_str)
    return Task(
        task_id=_task_id(seed, "field", depth, idx),
        prompt=prompt,
        answer=str(answer_raw),
        answer_raw=str(answer_raw),
        depth=depth,
        family=TaskFamily.FIELD_ARITHMETIC,
        dimension=dimension,
        structures=[field.name],
    )


def _generate_rule_induction_task(rng, depth, idx, seed, verbalizer, dimension=CompositionDimension.GENERAL):
    num_examples = depth + 3
    structure = _random_cyclic(rng, min_n=num_examples + 5, max_n=num_examples + 15)
    all_ops = make_standard_operations(structure, rng)
    chosen_ops = [rng.choice(all_ops) for _ in range(depth)]
    chain = ComposedFunction(chosen_ops, structure)

    inputs = rng.sample(structure.elements(), num_examples + 1)
    examples = [(i, chain(i)) for i in inputs[:num_examples]]
    test_input = inputs[num_examples]
    answer_raw = chain(test_input)

    prompt = verbalizer.verbalize_rule(structure, examples, test_input)

    return Task(
        task_id=_task_id(seed, "rule", depth, idx),
        prompt=prompt,
        answer=structure.element_to_str(answer_raw),
        answer_raw=structure.element_to_str(answer_raw),
        depth=depth,
        family=TaskFamily.RULE_INDUCTION,
        dimension=dimension,
        structures=[structure.name],
    )


def _generate_systematicity_pair(rng, depth, idx, seed, verbalizer):
    structure = _random_cyclic(rng, 5, 13)
    all_ops = make_standard_operations(structure, rng)
    if len(all_ops) < 4:
        return []
    f, g, h, k = rng.sample(all_ops, 4)

    x1 = structure.random_element(rng)
    chain1 = ComposedFunction([f, k], structure)
    answer1_raw = chain1(x1)
    task1 = Task(
        task_id=_task_id(seed, "syst", depth, f"{idx}-1"),
        prompt=verbalizer.verbalize_intra(structure, chain1, x1),
        answer=structure.element_to_str(answer1_raw),
        answer_raw=structure.element_to_str(answer1_raw),
        depth=depth,
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=CompositionDimension.SYSTEMATICITY,
        structures=[structure.name],
    )

    x2 = structure.random_element(rng)
    chain2 = ComposedFunction([h, g], structure)
    answer2_raw = chain2(x2)
    task2 = Task(
        task_id=_task_id(seed, "syst", depth, f"{idx}-2"),
        prompt=verbalizer.verbalize_intra(structure, chain2, x2),
        answer=structure.element_to_str(answer2_raw),
        answer_raw=structure.element_to_str(answer2_raw),
        depth=depth,
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=CompositionDimension.SYSTEMATICITY,
        structures=[structure.name],
    )
    return [task1, task2]


def _generate_substitutivity_task(rng, depth, idx, seed, verbalizer):
    n = rng.randint(5, 10)
    structure = CyclicGroup(n)
    labels = rng.sample(list("abcdefghijklmnopqrstuvwxyz"), n)
    label_map = {i: labels[i] for i in range(n)}

    all_ops = make_standard_operations(structure, rng)
    chosen_ops = [rng.choice(all_ops) for _ in range(depth)]
    chain = ComposedFunction(chosen_ops, structure)
    x = structure.random_element(rng)
    answer_raw = chain(x)

    prompt = f"Consider an algebraic structure with elements {{{', '.join(labels)}}}.\n"
    prompt += f"Starting with the element x = {label_map[x]}, perform the following operations in order:\n"
    for i, op in enumerate(chosen_ops, 1):
        prompt += f"Step {i}: Apply the operation '{op.name}'.\n"
    prompt += "\nWhat is the final result? Give only the label."

    return Task(
        task_id=_task_id(seed, "subst", depth, idx),
        prompt=prompt,
        answer=label_map[answer_raw],
        answer_raw=label_map[answer_raw],
        depth=depth,
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=CompositionDimension.SUBSTITUTIVITY,
        structures=[f"Relabeled Z_{n}"],
    )


def _generate_overgeneralization_task(rng, depth, idx, seed, verbalizer):
    structure = rng.choice([SymmetricGroup(3), DihedralGroup(4)])
    elements = structure.elements()
    attempts = 0
    a, b = rng.sample(elements, 2)
    while structure.op(a, b) == structure.op(b, a) and attempts < 50:
        a, b = rng.sample(elements, 2)
        attempts += 1

    answer_raw = structure.op(a, b)
    wrong_answer = structure.op(b, a)

    prompt = (
        f"Consider the algebraic structure {structure.name}: {structure.description}\n\n"
        f"Compute {structure.element_to_str(a)} {structure.operation_symbol()} {structure.element_to_str(b)}."
    )

    return Task(
        task_id=_task_id(seed, "overgen", depth, idx),
        prompt=prompt,
        answer=structure.element_to_str(answer_raw),
        answer_raw=structure.element_to_str(answer_raw),
        depth=depth,
        family=TaskFamily.INTER_STRUCTURE,
        dimension=CompositionDimension.OVERGENERALIZATION,
        structures=[structure.name],
        metadata={"wrong_answer": structure.element_to_str(wrong_answer)},
    )


class CompEvalGenerator:
    """Generates COMPEVAL benchmark task sets."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.verbalizer = Verbalizer(seed)

    def generate(
        self,
        depths: Optional[List[int]] = None,
        tasks_per_depth: int = 10,
        families: Optional[List[str]] = None,
        include_dimensions: bool = True,
    ) -> TaskSet:
        if depths is None:
            depths = [1, 2, 3, 4, 5]
        if families is None:
            families = ["intra", "inter", "field", "rule"]

        all_tasks: List[Task] = []
        blk_counter = 0

        family_generators = {
            "intra": _generate_intra_structure_task,
            "inter": _generate_inter_structure_task,
            "field": _generate_field_arithmetic_task,
            "rule": _generate_rule_induction_task,
        }

        for depth in depths:
            for family_name in families:
                gen_func = family_generators.get(family_name)
                if gen_func is None:
                    continue
                for i in range(tasks_per_depth):
                    try:
                        task = gen_func(self.rng, depth, blk_counter, self.seed, self.verbalizer)
                        all_tasks.append(task)
                        blk_counter += 1
                    except Exception:
                        continue

        if include_dimensions:
            for depth in [2, 3]:
                for i in range(tasks_per_depth // 2):
                    try:
                        syst_tasks = _generate_systematicity_pair(self.rng, depth, blk_counter, self.seed, self.verbalizer)
                        all_tasks.extend(syst_tasks)
                        blk_counter += len(syst_tasks)
                    except Exception:
                        continue

                    try:
                        subst_task = _generate_substitutivity_task(self.rng, depth, blk_counter, self.seed, self.verbalizer)
                        all_tasks.append(subst_task)
                        blk_counter += 1
                    except Exception:
                        continue

                    try:
                        overgen_task = _generate_overgeneralization_task(self.rng, depth, blk_counter, self.seed, self.verbalizer)
                        all_tasks.append(overgen_task)
                        blk_counter += 1
                    except Exception:
                        continue

        return TaskSet(
            tasks=all_tasks,
            name=f"compeval_seed{self.seed}",
            description=(
                f"COMPEVAL benchmark generated with seed={self.seed}. "
                f"Depths: {depths}, families: {families}, "
                f"tasks_per_depth: {tasks_per_depth}."
            ),
            metadata={
                "seed": self.seed,
                "depths": depths,
                "families": families,
                "tasks_per_depth": tasks_per_depth,
            },
        )

    def generate_productivity_suite(
        self,
        max_depth: int = 20,
        tasks_per_depth: int = 10,
    ) -> TaskSet:
        """Generate a productivity-focused suite with increasing depth."""
        tasks: List[Task] = []
        blk = 0
        for depth in range(1, max_depth + 1):
            for i in range(tasks_per_depth):
                try:
                    task = _generate_intra_structure_task(
                        self.rng, depth, blk, self.seed, self.verbalizer,
                        dimension=CompositionDimension.PRODUCTIVITY,
                    )
                    tasks.append(task)
                    blk += 1
                except Exception:
                    continue

        return TaskSet(
            tasks=tasks,
            name=f"compeval_productivity_seed{self.seed}",
            description=f"COMPEVAL productivity suite (depths 1-{max_depth}).",
            metadata={"seed": self.seed, "max_depth": max_depth},
        )
