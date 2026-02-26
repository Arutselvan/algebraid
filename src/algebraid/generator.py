"""
Compositional algebraic task generator.

Produces tasks in seven families: intra-structure composition,
inter-structure composition, field arithmetic, rule induction,
conceptual queries, adversarial reversible chains, and intermediate-state queries.
"""

import math
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .primitives import CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField, QuaternionGroup
from .skins import SKIN_REGISTRY, SemanticSkin
from .primitives.base import AlgebraicStructure
from .composers import DirectProduct, AlgebraicOperation, ComposedFunction, make_standard_operations
from .task_model import Task, TaskFamily, CompositionDimension, TaskSet
from .tasks.verbalizer import Verbalizer


def _task_id(seed, family, depth, idx):
    raw = f"{seed}-{family}-{depth}-{idx}"
    return "AG-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


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
    choice = rng.randint(0, 4)
    if choice == 0:
        return _random_cyclic(rng)
    elif choice == 1:
        return _random_symmetric(rng)
    elif choice == 2:
        return _random_dihedral(rng)
    elif choice == 3:
        return _random_field(rng)
    else:
        return QuaternionGroup()


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


def _generate_intra_structure_task(rng, depth, idx, seed, verbalizer, dimension=CompositionDimension.GENERAL, use_skins=True):
    structure = _random_structure(rng)
    skin = rng.choice(SKIN_REGISTRY[structure.__class__.__name__]) if use_skins else None
    all_ops = make_standard_operations(structure, rng)
    chosen_ops = [rng.choice(all_ops) for _ in range(depth)]
    chain = ComposedFunction(chosen_ops, structure)
    x = structure.random_element(rng)
    answer_raw = chain(x)
    answer_str = structure.element_to_str(answer_raw)
    answer_display = skin.element_name(answer_raw, structure) if skin else answer_str
    trace = chain.trace(x)
    prompt = verbalizer.verbalize_intra(structure, chain, x, skin=skin)
    return Task(
        task_id=_task_id(seed, "intra", depth, idx),
        prompt=prompt,
        answer=answer_display,
        answer_raw=answer_str,
        depth=depth,
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=dimension,
        structures=[structure.name],
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


def _generate_rule_induction_task(rng, depth, idx, seed, verbalizer, dimension=CompositionDimension.GENERAL, use_skins=True):
    num_examples = depth + 3
    structure = _random_cyclic(rng, min_n=num_examples + 5, max_n=num_examples + 15)
    all_ops = make_standard_operations(structure, rng)
    chosen_ops = [rng.choice(all_ops) for _ in range(depth)]
    chain = ComposedFunction(chosen_ops, structure)

    inputs = rng.sample(structure.elements(), num_examples + 1)
    examples = [(i, chain(i)) for i in inputs[:num_examples]]
    test_input = inputs[num_examples]
    answer_raw = chain(test_input)

    skin = rng.choice(SKIN_REGISTRY[structure.__class__.__name__]) if use_skins else None
    prompt = verbalizer.verbalize_rule(structure, examples, test_input, skin=skin)
    answer_str = structure.element_to_str(answer_raw)
    answer_display = skin.element_name(answer_raw, structure) if skin else answer_str

    return Task(
        task_id=_task_id(seed, "rule", depth, idx),
        prompt=prompt,
        answer=answer_display,
        answer_raw=answer_str,
        depth=depth,
        family=TaskFamily.RULE_INDUCTION,
        dimension=dimension,
        structures=[structure.name],
    )


def _element_order(structure: AlgebraicStructure, x: Any) -> int:
    """Compute the order of element x in structure (smallest k>=1 where x^k = identity)."""
    e = structure.identity()
    current = x
    for k in range(1, structure.order() + 1):
        if current == e:
            return k
        current = structure.op(current, x)
    return structure.order()  # fallback (should not occur for finite groups)


def _generate_conceptual_task(rng, depth, idx, seed, verbalizer):
    """Generate a conceptual query task (identity, order, commutativity, etc.)."""
    # Query subtypes weighted: is_generator only for CyclicGroup
    general_subtypes = [
        "identity", "element_order", "commutativity_check",
        "structure_order", "is_abelian", "inverse_of",
    ]
    subtype = rng.choice(general_subtypes + ["is_generator"])

    if subtype == "is_generator":
        structure: AlgebraicStructure = _random_cyclic(rng, min_n=4, max_n=15)
    else:
        structure = _random_structure(rng)

    x = structure.random_element(rng)
    a = structure.random_element(rng)
    b = structure.random_element(rng)

    if subtype == "identity":
        answer_raw = structure.element_to_str(structure.identity())
        prompt = verbalizer.verbalize_conceptual(structure, "identity")
    elif subtype == "element_order":
        ord_x = _element_order(structure, x)
        answer_raw = str(ord_x)
        prompt = verbalizer.verbalize_conceptual(structure, "element_order", x=x)
    elif subtype == "commutativity_check":
        # Ensure non-trivial pair for non-abelian structures
        if not structure.is_commutative():
            elements = structure.elements()
            attempts = 0
            while structure.op(a, b) == structure.op(b, a) and attempts < 30:
                a, b = rng.choice(elements), rng.choice(elements)
                attempts += 1
        answer_raw = "yes" if structure.op(a, b) == structure.op(b, a) else "no"
        prompt = verbalizer.verbalize_conceptual(structure, "commutativity_check", a=a, b=b)
    elif subtype == "structure_order":
        answer_raw = str(structure.order())
        prompt = verbalizer.verbalize_conceptual(structure, "structure_order")
    elif subtype == "is_abelian":
        answer_raw = "yes" if structure.is_commutative() else "no"
        prompt = verbalizer.verbalize_conceptual(structure, "is_abelian")
    elif subtype == "inverse_of":
        answer_raw = structure.element_to_str(structure.inverse(x))
        prompt = verbalizer.verbalize_conceptual(structure, "inverse_of", x=x)
    else:  # is_generator (CyclicGroup only)
        n = structure.n  # type: ignore[attr-defined]
        answer_raw = "yes" if math.gcd(x, n) == 1 else "no"
        prompt = verbalizer.verbalize_conceptual(structure, "is_generator", x=x)

    return Task(
        task_id=_task_id(seed, "conceptual", depth, idx),
        prompt=prompt,
        answer=answer_raw,
        answer_raw=answer_raw,
        depth=1,
        family=TaskFamily.CONCEPTUAL_QUERY,
        dimension=CompositionDimension.GENERAL,
        structures=[structure.name],
        metadata={
            "query_subtype": subtype,
            "structure_is_abelian": structure.is_commutative(),
            "structure_order": structure.order(),
        },
        solution_trace=None,
    )


def _generate_adversarial_task(rng, depth, idx, seed, verbalizer, use_skins=True):
    """Generate an adversarial reversible-chain task designed to trap common errors."""
    adv_type = rng.choice(["double_inverse", "self_cancelling", "commutativity_trap", "identity_bait"])

    if adv_type == "double_inverse":
        structure = _random_structure(rng)
        skin = rng.choice(SKIN_REGISTRY[structure.__class__.__name__]) if use_skins else None
        x = structure.random_element(rng)
        all_ops = make_standard_operations(structure, rng)
        inv_ops = [op for op in all_ops if op.name == "inverse"]
        if not inv_ops:
            raise ValueError("No inverse operation found")
        inv_op = inv_ops[0]
        chain = ComposedFunction([inv_op, inv_op], structure)
        answer_raw_val = chain(x)
        wrong_val = structure.inverse(x)
        trace = chain.trace(x)
        wrong_str = structure.element_to_str(wrong_val)
        rationale = (
            f"Applying inverse once gives {wrong_str}; "
            "a model that stops early misses the second inverse which restores the original."
        )
        prompt = verbalizer.verbalize_intra(structure, chain, x, skin=skin)
        # Inject adversarial annotation
        prompt = prompt.rstrip() + "\nNote: each operation is applied exactly as listed."

    elif adv_type == "self_cancelling":
        structure = _random_cyclic(rng, min_n=4, max_n=15)
        skin = rng.choice(SKIN_REGISTRY["CyclicGroup"]) if use_skins else None
        x = structure.random_element(rng)
        n = structure.n
        c = rng.randint(1, n - 1)
        neg_c = (-c) % n
        all_ops = make_standard_operations(structure, rng)
        # Build custom left_mul_c and left_mul_neg_c
        op_add = AlgebraicOperation(
            name=f"left_mul_{c}",
            func=lambda val, k: structure.op(structure.multiply(1, k) if hasattr(structure, 'multiply') else k, val),
            arity=1,
            description=f"add {c} (mod {n})",
            symbol=f"+{c}",
            fixed_args=(c,),
        )
        op_sub = AlgebraicOperation(
            name=f"left_mul_{neg_c}",
            func=lambda val, k: structure.op(structure.multiply(1, k) if hasattr(structure, 'multiply') else k, val),
            arity=1,
            description=f"add {neg_c} (mod {n})",
            symbol=f"+{neg_c}",
            fixed_args=(neg_c,),
        )
        # Use right_mul ops for clean semantics: (x + c) then (x + n - c) = x
        op_add_r = AlgebraicOperation(
            name=f"right_mul_{c}",
            func=lambda val, k: (val + k) % n,
            arity=1,
            description=f"add {c} (mod {n})",
            symbol=f"+{c}",
            fixed_args=(c,),
        )
        op_sub_r = AlgebraicOperation(
            name=f"right_mul_{neg_c}",
            func=lambda val, k: (val + k) % n,
            arity=1,
            description=f"add {neg_c} (mod {n})",
            symbol=f"+{neg_c}",
            fixed_args=(neg_c,),
        )
        chain = ComposedFunction([op_add_r, op_sub_r], structure)
        answer_raw_val = chain(x)
        wrong_val = (x + c) % n
        trace = chain.trace(x)
        wrong_str = str(wrong_val)
        rationale = (
            f"After adding {c} the value is {wrong_str}; "
            f"a model that ignores the second step (+{neg_c}) stops there."
        )
        prompt = verbalizer.verbalize_intra(structure, chain, x, skin=skin)
        prompt = prompt.rstrip() + "\nApply all operations in order, without simplifying intermediate steps."

    elif adv_type == "commutativity_trap":
        # Non-abelian structure: find a, b where a*b != b*a
        struct_choice = rng.randint(0, 2)
        if struct_choice == 0:
            structure = _random_symmetric(rng)
        elif struct_choice == 1:
            structure = _random_dihedral(rng)
        else:
            structure = QuaternionGroup()
        skin = rng.choice(SKIN_REGISTRY[structure.__class__.__name__]) if use_skins else None
        elements = structure.elements()
        a_elem = rng.choice(elements)
        b_elem = rng.choice(elements)
        attempts = 0
        while structure.op(a_elem, b_elem) == structure.op(b_elem, a_elem) and attempts < 50:
            a_elem, b_elem = rng.choice(elements), rng.choice(elements)
            attempts += 1
        # Build chain: start with a, right-multiply by b
        b_op = AlgebraicOperation(
            name=f"right_mul_{structure.element_to_str(b_elem)}",
            func=lambda val, c: structure.op(val, c),
            arity=1,
            description=f"right-multiply by {structure.element_to_str(b_elem)}",
            symbol=f"* {structure.element_to_str(b_elem)}",
            fixed_args=(b_elem,),
        )
        chain = ComposedFunction([b_op], structure)
        x = a_elem
        answer_raw_val = chain(x)
        wrong_val = structure.op(b_elem, a_elem)
        trace = chain.trace(x)
        wrong_str = structure.element_to_str(wrong_val)
        rationale = (
            f"The commuted result {wrong_str} is different; "
            "a model that assumes commutativity will swap the operands."
        )
        prompt = verbalizer.verbalize_intra(structure, chain, x, skin=skin)
        prompt = prompt.rstrip() + "\nNote: order of operands matters in this structure."

    else:  # identity_bait
        structure = _random_cyclic(rng, min_n=4, max_n=15)
        skin = rng.choice(SKIN_REGISTRY["CyclicGroup"]) if use_skins else None
        x = structure.random_element(rng)
        n = structure.n
        neg_x = (-x) % n
        # Chain: add neg_x to x -> result is 0 (identity)
        op_cancel = AlgebraicOperation(
            name=f"right_mul_{neg_x}",
            func=lambda val, k: (val + k) % n,
            arity=1,
            description=f"add {neg_x} (mod {n})",
            symbol=f"+{neg_x}",
            fixed_args=(neg_x,),
        )
        chain = ComposedFunction([op_cancel], structure)
        answer_raw_val = chain(x)  # = 0, the identity
        wrong_val = x
        trace = chain.trace(x)
        wrong_str = str(wrong_val)
        rationale = (
            f"The starting value {wrong_str} is not the answer; "
            "a model confused by the identity element may return the input unchanged."
        )
        prompt = verbalizer.verbalize_intra(structure, chain, x, skin=skin)
        prompt = prompt.rstrip() + "\nCompute the result step by step."

    answer_str = structure.element_to_str(answer_raw_val)
    answer_display = answer_str
    if use_skins and skin:
        answer_display = skin.element_name(answer_raw_val, structure)

    return Task(
        task_id=_task_id(seed, "adversarial", depth, idx),
        prompt=prompt,
        answer=answer_display,
        answer_raw=answer_str,
        depth=len(chain.operations),
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=CompositionDimension.ADVERSARIAL,
        structures=[structure.name],
        metadata={
            "adversarial_type": adv_type,
            "wrong_answer": wrong_str,
            "wrong_answer_rationale": rationale,
        },
        solution_trace=[(op_name, structure.element_to_str(val)) for op_name, val in trace],
    )


def _generate_intermediate_state_task(rng, depth, idx, seed, verbalizer, use_skins=True):
    """Generate a task that asks for the value at an intermediate step in a chain."""
    if depth < 2:
        depth = 2
    structure = _random_structure(rng)
    skin = rng.choice(SKIN_REGISTRY[structure.__class__.__name__]) if use_skins else None
    all_ops = make_standard_operations(structure, rng)
    chosen_ops = [rng.choice(all_ops) for _ in range(depth)]
    chain = ComposedFunction(chosen_ops, structure)
    x = structure.random_element(rng)

    # Full trace: [("start", x), (op1, v1), ..., (opN, vN)]
    full_trace = chain.trace(x)

    # Query step k in [1, depth-1]
    k = rng.randint(1, depth - 1)

    answer_raw_val = full_trace[k][1]  # value after step k
    answer_str = structure.element_to_str(answer_raw_val)
    answer_display = skin.element_name(answer_raw_val, structure) if skin else answer_str

    final_val = full_trace[depth][1]
    final_str = structure.element_to_str(final_val)

    prompt = verbalizer.verbalize_intermediate_state(structure, chain, x, k, skin=skin)

    return Task(
        task_id=_task_id(seed, "intermediate", k, idx),
        prompt=prompt,
        answer=answer_display,
        answer_raw=answer_str,
        depth=k,
        family=TaskFamily.INTRA_STRUCTURE,
        dimension=CompositionDimension.INTERMEDIATE_STATE,
        structures=[structure.name],
        metadata={
            "query_step": k,
            "total_steps": depth,
            "final_answer": final_str,
        },
        solution_trace=[(op_name, structure.element_to_str(val)) for op_name, val in full_trace[:k + 1]],
    )


class AlgebraidGenerator:
    """Generates ALGEBRAID benchmark task sets."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.verbalizer = Verbalizer(seed)

    def generate(
        self,
        depths: Optional[List[int]] = None,
        tasks_per_depth: int = 10,
        families: Optional[List[str]] = None,
        use_skins: bool = True,
    ) -> TaskSet:
        if depths is None:
            depths = [1, 2, 3, 4, 5]
        if families is None:
            families = ["intra", "inter", "field", "rule", "conceptual", "adversarial", "intermediate"]

        all_tasks: List[Task] = []
        blk_counter = 0

        family_generators = {
            "intra": _generate_intra_structure_task,
            "inter": _generate_inter_structure_task,
            "field": _generate_field_arithmetic_task,
            "rule": _generate_rule_induction_task,
            "conceptual": _generate_conceptual_task,
            "adversarial": _generate_adversarial_task,
            "intermediate": _generate_intermediate_state_task,
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

        return TaskSet(
            tasks=all_tasks,
            name=f"algebraid_s{self.seed}",
            description=(
                f"ALGEBRAID benchmark generated with seed={self.seed}. "
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
            name=f"algebraid_productivity_s{self.seed}",
            description=f"ALGEBRAID productivity suite (depths 1-{max_depth}).",
            metadata={"seed": self.seed, "max_depth": max_depth},
        )
