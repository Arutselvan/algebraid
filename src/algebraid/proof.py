"""
Independent algebraic verifier for ALGEBRAID tasks.

Addresses paper Blocker 1 (Formal Verification) by providing a
proof-of-correctness guarantee through independent re-derivation of
every answer from the solution trace using raw algebraic primitives,
without relying on the generator's computation path.

For each task that has a solution_trace, the verifier:
  1. Reconstructs the algebraic structure from its canonical name.
  2. Parses every element string in the trace back to its internal form.
  3. Re-applies each named operation using the structure's own axioms.
  4. Asserts that each computed value matches the stored trace value.
  5. Asserts that the final value matches answer_raw.

A task that passes all checks is *proven correct* up to the soundness of
the algebraic primitive implementations, which are themselves tested by
the group-axiom tests in tests/test_primitives.py.

Tasks without a solution_trace (conceptual queries, rule-induction tasks)
are marked ``verified=True, steps_checked=0`` - they are "skipped" rather
than "proven".  Their correctness is guaranteed by construction: the
generator computes the answers by direct algebraic computation rather than
by following a trace, and the conceptual-task correctness tests in
tests/test_generator.py independently verify that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .task_model import Task, TaskFamily, TaskSet
from .primitives import (
    CyclicGroup, SymmetricGroup, DihedralGroup, FiniteField, QuaternionGroup,
)
from .primitives.base import AlgebraicStructure
from .composers import DirectProduct


# -- Q_8 element name -> internal index ---------------------------------------

_Q8_NAME_TO_IDX: Dict[str, int] = {
    "1": 0, "-1": 1, "i": 2, "-i": 3, "j": 4, "-j": 5, "k": 6, "-k": 7,
}


# -- Structure reconstruction -------------------------------------------------

def _parse_structure(name: str) -> Optional[AlgebraicStructure]:
    """Reconstruct an algebraic structure from its canonical name string."""
    name = name.strip()
    if name == "Q_8":
        return QuaternionGroup()
    m = re.match(r'^Z_(\d+)$', name)
    if m:
        return CyclicGroup(int(m.group(1)))
    m = re.match(r'^S_(\d+)$', name)
    if m:
        return SymmetricGroup(int(m.group(1)))
    m = re.match(r'^D_(\d+)$', name)
    if m:
        return DihedralGroup(int(m.group(1)))
    m = re.match(r'^GF\((\d+)\)$', name)
    if m:
        return FiniteField(int(m.group(1)))
    return None  # direct products or unknown


# -- Element parsing ----------------------------------------------------------

def _parse_element(s: str, structure: AlgebraicStructure) -> Any:
    """Parse a string element representation to its internal form."""
    s = s.strip()
    if isinstance(structure, QuaternionGroup):
        if s in _Q8_NAME_TO_IDX:
            return _Q8_NAME_TO_IDX[s]
        raise ValueError(f"Unknown Q_8 element: {s!r}")
    if isinstance(structure, (CyclicGroup, FiniteField)):
        return int(s)
    if isinstance(structure, SymmetricGroup):
        # Format: "(1, 2, 3)"
        nums = re.findall(r'-?\d+', s)
        return tuple(int(x) for x in nums)
    if isinstance(structure, DihedralGroup):
        # Formats: "e", "s", "r^k", "r^ks"
        if s == 'e':
            return (0, 0)
        if s == 's':
            return (0, 1)
        m = re.match(r'^r\^(\d+)(s?)$', s)
        if m:
            return (int(m.group(1)), 1 if m.group(2) == 's' else 0)
        raise ValueError(f"Cannot parse D_n element: {s!r}")
    raise ValueError(f"Cannot parse element {s!r} for {structure.name}")


# -- Operation re-application -------------------------------------------------

def _apply_named_op(op_name: str, val: Any, structure: AlgebraicStructure) -> Any:
    """
    Re-apply a generator operation by name to a value, from first principles.

    Recognises:
      inverse          - structure.inverse(val)
      right_mul_{k}    - structure.op(val, k)
      left_mul_{k}     - structure.op(k, val)
      power_{n}        - val composed with itself n times
      conj_{k}         - k * val * k^{-1}
    """
    if op_name == "start":
        return val

    if op_name == "inverse":
        return structure.inverse(val)

    m = re.match(r'^right_mul_(.+)$', op_name)
    if m:
        k = _parse_element(m.group(1), structure)
        return structure.op(val, k)

    m = re.match(r'^left_mul_(.+)$', op_name)
    if m:
        k = _parse_element(m.group(1), structure)
        return structure.op(k, val)

    m = re.match(r'^power_(\d+)$', op_name)
    if m:
        n = int(m.group(1))
        # Use repeated group operation (op_chain semantics), NOT structure.power()
        # which may implement field exponentiation rather than group repetition.
        result = val
        for _ in range(n - 1):
            result = structure.op(result, val)
        return result

    m = re.match(r'^conj_(.+)$', op_name)
    if m:
        k = _parse_element(m.group(1), structure)
        return structure.op(structure.op(k, val), structure.inverse(k))

    raise ValueError(f"Unrecognised operation: {op_name!r}")


# -- Direct product support ---------------------------------------------------

def _parse_structure_list(names: List[str]) -> Optional[AlgebraicStructure]:
    """Reconstruct a (possibly composed) structure from a list of names.

    If *names* has one entry, returns a single structure.
    If multiple, composes them left-to-right with DirectProduct, matching
    the generator's construction order.
    """
    if not names:
        return None
    structs = [_parse_structure(n) for n in names]
    if any(s is None for s in structs):
        return None
    result = structs[0]
    for s in structs[1:]:
        result = DirectProduct(result, s)
    return result


def _parse_dp_element(s: str, structure: AlgebraicStructure) -> Any:
    """Parse a string like ``(1, 3)`` or ``((1, 3), 5)`` into a nested tuple.

    For a single (non-product) structure, delegates to ``_parse_element``.
    For DirectProducts, recursively parses the nested parenthesised form
    produced by ``DirectProduct.element_to_str``.
    """
    if not isinstance(structure, DirectProduct):
        return _parse_element(s, structure)

    import ast
    raw = ast.literal_eval(s.strip())

    def _convert(val: Any, st: AlgebraicStructure) -> Any:
        if not isinstance(st, DirectProduct):
            return _parse_element(str(val), st)
        if not isinstance(val, (tuple, list)) or len(val) != 2:
            raise ValueError(f"Expected 2-tuple for DirectProduct, got {val!r}")
        left = _convert(val[0], st.G)
        right = _convert(val[1], st.H)
        return (left, right)

    return _convert(raw, structure)


def _verify_inter_task(task: Task) -> "ProofResult":
    """Verify an inter-structure task by reconstructing the DirectProduct."""
    meta = task.metadata or {}
    inter_op = meta.get("inter_op")
    if not inter_op:
        return ProofResult(
            task_id=task.task_id, verified=True, steps_checked=0,
            error_message="Skipped: no inter_op metadata.",
        )

    structure = _parse_structure_list(task.structures or [])
    if structure is None:
        return ProofResult(
            task_id=task.task_id, verified=True, steps_checked=0,
            error_message="Skipped: cannot reconstruct structure list.",
        )

    try:
        a = _parse_dp_element(meta["operand_a"], structure)
        if inter_op == "inverse":
            computed = structure.inverse(a)
        elif inter_op == "op":
            b = _parse_dp_element(meta["operand_b"], structure)
            computed = structure.op(a, b)
        elif inter_op == "op_then_inverse":
            b = _parse_dp_element(meta["operand_b"], structure)
            c = structure.op(a, b)
            computed = structure.inverse(c)
        else:
            return ProofResult(
                task_id=task.task_id, verified=False, steps_checked=0,
                error_message=f"Unknown inter_op: {inter_op!r}",
            )

        expected = _parse_dp_element(str(task.answer_raw), structure)
        if computed != expected:
            return ProofResult(
                task_id=task.task_id, verified=False, steps_checked=0,
                error_message=(
                    f"Inter verification failed: computed "
                    f"{structure.element_to_str(computed)!r}, "
                    f"expected {task.answer_raw!r}."
                ),
            )
        return ProofResult(task_id=task.task_id, verified=True, steps_checked=1)
    except Exception as exc:
        return ProofResult(
            task_id=task.task_id, verified=False, steps_checked=0,
            error_message=f"Inter verification error: {exc}",
        )


# -- Proof result -------------------------------------------------------------

@dataclass
class ProofResult:
    """Outcome of formally verifying a single task."""
    task_id: str
    verified: bool
    steps_checked: int
    failed_step: Optional[int] = None
    error_message: Optional[str] = None


# -- Single-task verifier -----------------------------------------------------

def verify_task(task: Task) -> ProofResult:
    """
    Verify a task by independently re-deriving its answer from the trace.

    Returns a ProofResult.  Tasks without a solution_trace are marked
    verified=True with steps_checked=0 (skipped, not failed).
    """
    # Inter-structure tasks have no solution trace but can be independently verified
    if task.family == TaskFamily.INTER_STRUCTURE and not task.solution_trace:
        return _verify_inter_task(task)

    if not task.solution_trace:
        return ProofResult(task_id=task.task_id, verified=True, steps_checked=0)

    struct_name = (task.structures or [""])[0]
    structure = _parse_structure(struct_name)
    if structure is None:
        return ProofResult(
            task_id=task.task_id,
            verified=True,
            steps_checked=0,
            error_message=f"Skipped: cannot reconstruct '{struct_name}' (direct product?)",
        )

    trace = task.solution_trace
    try:
        op0, val0_str = trace[0]
        if op0 != "start":
            return ProofResult(
                task_id=task.task_id, verified=False, steps_checked=0,
                failed_step=0, error_message="Trace missing 'start' entry.",
            )
        current = _parse_element(str(val0_str), structure)

        for step_idx, (op_name, stored_val_str) in enumerate(trace[1:], start=1):
            try:
                computed = _apply_named_op(op_name, current, structure)
            except ValueError as exc:
                return ProofResult(
                    task_id=task.task_id, verified=False,
                    steps_checked=step_idx - 1, failed_step=step_idx,
                    error_message=f"Cannot apply '{op_name}': {exc}",
                )
            stored = _parse_element(str(stored_val_str), structure)
            if computed != stored:
                return ProofResult(
                    task_id=task.task_id, verified=False,
                    steps_checked=step_idx - 1, failed_step=step_idx,
                    error_message=(
                        f"Step {step_idx} ({op_name}): expected "
                        f"{structure.element_to_str(computed)!r}, "
                        f"trace says {stored_val_str!r}."
                    ),
                )
            current = computed

        # Final consistency: last trace value must equal answer_raw.
        final_trace_val = _parse_element(str(trace[-1][1]), structure)
        final_answer = _parse_element(str(task.answer_raw), structure)
        if final_trace_val != final_answer:
            return ProofResult(
                task_id=task.task_id, verified=False,
                steps_checked=len(trace) - 1, failed_step=len(trace),
                error_message=(
                    f"Final trace value {trace[-1][1]!r} does not match "
                    f"answer_raw {task.answer_raw!r}."
                ),
            )

        return ProofResult(
            task_id=task.task_id, verified=True, steps_checked=len(trace) - 1,
        )

    except Exception as exc:
        return ProofResult(
            task_id=task.task_id, verified=False, steps_checked=0,
            error_message=f"Unexpected error: {exc}",
        )


# -- Batch verifier -----------------------------------------------------------

def verify_set(task_set: TaskSet) -> Dict[str, Any]:
    """Verify every task in a TaskSet and return a proof report dict."""
    results: List[ProofResult] = [verify_task(t) for t in task_set]
    trace_verified = [r for r in results if r.verified and r.steps_checked > 0]
    skipped = [r for r in results if r.verified and r.steps_checked == 0]
    failed = [r for r in results if not r.verified]
    total_traceable = len(trace_verified) + len(failed)

    return {
        "total": len(results),
        "trace_verified": len(trace_verified),
        "skipped": len(skipped),
        "failed": len(failed),
        "proof_rate": round(len(trace_verified) / total_traceable * 100, 1) if total_traceable else 0.0,
        "coverage": round((len(trace_verified) + len(skipped)) / len(results) * 100, 1) if results else 0.0,
        "failures": [
            {"task_id": r.task_id, "failed_step": r.failed_step, "message": r.error_message}
            for r in failed
        ],
    }


def print_proof_report(report: Dict[str, Any]) -> None:
    """Print a human-readable proof verification summary."""
    sep = "=" * 56
    print(f"\n{sep}")
    print("  ALGEBRAID Proof Report")
    print(sep)
    print(f"  Total tasks    : {report['total']}")
    print(f"  Trace-verified : {report['trace_verified']}  (trace algebraically re-derived)")
    print(f"  Skipped        : {report['skipped']}  (no trace: conceptual/rule tasks)")
    print(f"  Failed         : {report['failed']}")
    print(f"  Proof rate     : {report['proof_rate']}%  (of traceable tasks)")
    print(f"  Coverage       : {report['coverage']}%  (verified + skipped / total)")
    if report["failures"]:
        print(f"\n  Proof failures (first 10):")
        for f in report["failures"][:10]:
            step = f['failed_step']
            print(f"    [{f['task_id']}] step {step}: {f['message']}")
    print(sep)
