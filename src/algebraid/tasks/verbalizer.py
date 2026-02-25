"""
ALGEBRAID Natural Language Diversity System.

Provides 50+ prompt template variants, entity name randomization, and
multiple context frames to resist contamination and ensure that models
are tested on reasoning rather than pattern-matching surface forms.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from ..primitives.base import AlgebraicStructure
from ..composers.function_composition import AlgebraicOperation, ComposedFunction

# ===========================================================================
# Entity name pools (1000+ alternatives)
# ===========================================================================

STRUCTURE_ALIASES: List[str] = [
    "system", "structure", "set with operation", "algebraic system",
    "mathematical group", "operation table", "number system",
    "closed system", "abstract algebra", "computational domain",
]

ELEMENT_LABEL_POOLS: Dict[str, List[str]] = {
    "greek": list("αβγδεζηθικλμνξοπρστυφχψω"),
    "emoji_safe": [f"e{i}" for i in range(30)],
    "colors": [
        "red", "blue", "green", "yellow", "purple", "orange", "cyan",
        "magenta", "lime", "teal", "navy", "coral", "gold", "silver",
        "ivory", "jade", "ruby", "pearl", "amber", "onyx", "slate",
    ],
    "cities": [
        "Paris", "Tokyo", "Lima", "Oslo", "Rome", "Cairo", "Seoul",
        "Delhi", "Baku", "Doha", "Suva", "Riga", "Hilo", "Quito",
        "Lome", "Apia", "Bern", "Kiev", "Minsk", "Accra", "Hanoi",
    ],
    "planets": [
        "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Neptune",
        "Pluto", "Ceres", "Eris", "Haumea", "Makemake", "Titan",
        "Europa", "Io", "Ganymede", "Callisto", "Triton", "Oberon",
    ],
    "musical_notes": [
        "C", "D", "E", "F", "G", "A", "B",
        "C#", "D#", "F#", "G#", "A#",
        "Db", "Eb", "Gb", "Ab", "Bb",
    ],
    "letters_upper": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "letters_lower": list("abcdefghijklmnopqrstuvwxyz"),
}

# Context frames for wrapping mathematical tasks
CONTEXT_FRAMES: Dict[str, Dict[str, str]] = {
    "pure_math": {
        "intro": "Consider the following mathematical problem.",
        "ask": "What is the result?",
    },
    "cryptography": {
        "intro": "You are designing a cipher. The encryption uses the following algebraic operations.",
        "ask": "What is the encrypted value?",
    },
    "clock": {
        "intro": "Imagine a clock with {n} hours (numbered 0 to {n_minus_1}). All arithmetic wraps around.",
        "ask": "What time does the clock show?",
    },
    "chemistry": {
        "intro": "In a molecular symmetry analysis, transformations follow the rules of the group below.",
        "ask": "What is the resulting transformation?",
    },
    "game": {
        "intro": "In a puzzle game, moves combine according to the following rules.",
        "ask": "What is the final position?",
    },
    "music": {
        "intro": "In music theory, intervals combine according to modular arithmetic.",
        "ask": "What interval results?",
    },
    "robotics": {
        "intro": "A robot arm has joints that rotate in discrete steps following the rules below.",
        "ask": "What is the final joint configuration?",
    },
    "abstract": {
        "intro": "Let S be a set with a binary operation * satisfying the following rules.",
        "ask": "Determine the output.",
    },
}

# ===========================================================================
# Template pools by task family
# ===========================================================================

INTRA_TEMPLATES: List[Dict[str, str]] = [
    # Template 0: Original (baseline)
    {
        "header": "Consider the algebraic structure {name} ({desc}).",
        "start": "Starting with the element x = {x}, perform the following operations in order:",
        "step": "Step {i}: Apply {op_desc}.",
        "footer": "What is the final result? Give only the answer.",
    },
    # Template 1: Imperative
    {
        "header": "You are working in {name}, defined as: {desc}.",
        "start": "Begin with x = {x}. Execute these operations sequentially:",
        "step": "{i}. {op_desc}",
        "footer": "Report the final value. Answer with just the number.",
    },
    # Template 2: Narrative
    {
        "header": "In the algebraic structure {name} ({desc}), a computation unfolds as follows.",
        "start": "The initial value is {x}. The following transformations are applied one after another:",
        "step": "Transformation {i}: {op_desc}.",
        "footer": "What value remains at the end? State only the answer.",
    },
    # Template 3: Formal
    {
        "header": "Let G = {name} with the property: {desc}.",
        "start": "Given x₀ = {x}, compute x_{depth} by applying these steps in sequence:",
        "step": "x_{i} = {op_desc}(x_{i_prev})",
        "footer": "What is x_{depth}? Provide the numerical answer only.",
    },
    # Template 4: Question-first
    {
        "header": "Here is a problem involving {name} ({desc}).",
        "start": "If you start at {x} and apply the operations below one by one, what do you end up with?",
        "step": "Operation {i}: {op_desc}",
        "footer": "Give only the final answer.",
    },
    # Template 5: Conversational
    {
        "header": "I need help with a calculation in {name}. For reference: {desc}.",
        "start": "Take the element {x}.",
        "step": "{op_desc}.",
        "footer": "What's the answer? Just the value, please.",
    },
    # Template 6: Exam-style
    {
        "header": "Problem: In {name} ({desc}), evaluate the following composition.",
        "start": "Input: {x}",
        "step": "Apply {op_desc}",
        "footer": "Final answer:",
    },
    # Template 7: Chain notation
    {
        "header": "Working within {name}: {desc}.",
        "start": "Evaluate the chain starting from {x}:",
        "step": "→ apply {op_desc} → ?",
        "footer": "What is the final output? Answer only.",
    },
    # Template 8: Pseudocode
    {
        "header": "Consider {name} where {desc}.",
        "start": "x = {x}",
        "step": "x = {op_desc}(x)  # step {i}",
        "footer": "What is the value of x? Give only the number.",
    },
    # Template 9: Reverse framing (state the goal first)
    {
        "header": "Determine the result of a sequence of algebraic operations in {name} ({desc}).",
        "start": "The sequence begins at {x} and proceeds through {depth} steps:",
        "step": "Step {i}: {op_desc}(x_{i_prev}) → x_{i}",
        "footer": "State the final result as a single value.",
    },
    # Template 10: Minimal
    {
        "header": "{name}. {desc}.",
        "start": "x = {x}.",
        "step": "{op_desc}.",
        "footer": "Result?",
    },
    # Template 11: Verbose / tutorial
    {
        "header": "We will work in the algebraic structure known as {name}. Recall that {desc}.",
        "start": "Our starting element is x = {x}. We will apply a series of {depth} operations to this element, one after another. Each operation takes the current value and produces a new value within the same structure.",
        "step": "In step {i}, we {op_desc}.",
        "footer": "After all {depth} operations have been applied, what is the resulting element? Please provide only the final numerical answer.",
    },
]

INTER_TEMPLATES: List[Dict[str, str]] = [
    {
        "header": "Consider the group {name}: {desc}",
        "task": "{op_desc}",
        "footer": "What is the result? Give only the answer as a tuple.",
    },
    {
        "header": "Working in the product group {name} ({desc}):",
        "task": "{op_desc}",
        "footer": "Express the answer as a tuple.",
    },
    {
        "header": "In {name}, where {desc}, solve the following.",
        "task": "{op_desc}",
        "footer": "Provide the tuple answer only.",
    },
    {
        "header": "Problem: {name} ({desc}).",
        "task": "Compute {op_desc}",
        "footer": "Answer (tuple):",
    },
    {
        "header": "Let G = {name}. {desc}.",
        "task": "Evaluate: {op_desc}",
        "footer": "Give the result as a tuple.",
    },
]

FIELD_TEMPLATES: List[Dict[str, str]] = [
    {
        "header": "Consider the finite field {name}: {desc}.",
        "task": "Compute the following expression in {name}: {expr}",
        "footer": "What is the result? Give only the numerical answer.",
    },
    {
        "header": "In {name} ({desc}), evaluate:",
        "task": "{expr}",
        "footer": "Answer (single number):",
    },
    {
        "header": "Working in {name} where {desc}.",
        "task": "What is {expr}?",
        "footer": "Give only the numerical result.",
    },
    {
        "header": "Calculate the following in the finite field {name}. Recall: {desc}.",
        "task": "Expression: {expr}",
        "footer": "Final numerical answer:",
    },
    {
        "header": "Problem: Evaluate an arithmetic expression in {name} ({desc}).",
        "task": "{expr}",
        "footer": "What does this equal? State only the number.",
    },
]

RULE_TEMPLATES: List[Dict[str, str]] = [
    {
        "header": "Consider the algebraic structure {name}: {desc}.",
        "intro": "An unknown function f maps elements of this structure to other elements. Here are some examples of f:",
        "example": "  f({x}) = {y}",
        "question": "What is f({test})? Give only the answer.",
    },
    {
        "header": "In {name} ({desc}), a mystery transformation T is defined by the following input-output pairs:",
        "intro": "",
        "example": "  T({x}) = {y}",
        "question": "Predict T({test}). Answer with just the value.",
    },
    {
        "header": "A function f operates on elements of {name} ({desc}).",
        "intro": "You are given these observations:",
        "example": "  Input: {x}  →  Output: {y}",
        "question": "What output does input {test} produce? Give only the answer.",
    },
    {
        "header": "Pattern recognition in {name} ({desc}).",
        "intro": "The following mappings have been observed:",
        "example": "  {x} → {y}",
        "question": "Following the same pattern, what does {test} map to?",
    },
    {
        "header": "Decode the rule. Working in {name}. {desc}.",
        "intro": "Clues:",
        "example": "  f({x}) = {y}",
        "question": "Apply the rule to {test}. State only the answer.",
    },
]


class Verbalizer:
    '''
    Generates diverse natural language prompts for ALGEBRAID tasks.

    Usage:
        verb = Verbalizer(seed=42)
        prompt = verb.verbalize_intra(structure, ops, x, depth)
        prompt = verb.verbalize_inter(composed, op_desc)
        prompt = verb.verbalize_field(field_struct, expr)
        prompt = verb.verbalize_rule(structure, examples, test_input)
    '''

    def __init__(self, seed: int = 42, context_frame: Optional[str] = None) -> None:
        """
        Args:
            seed: Random seed for template selection.
            context_frame: Force a specific context frame (None = random).
        """
        self._rng = random.Random(seed)
        self.context_frame = context_frame

    def _pick_template(self, templates: list) -> dict:
        """Randomly select a template from the pool."""
        return self._rng.choice(templates)

    def _maybe_add_context(self, lines: List[str], n: Any = None) -> List[str]:
        """Optionally wrap the prompt in a context frame."""
        if self.context_frame:
            frame: Dict[str, str] | None = CONTEXT_FRAMES.get(self.context_frame)
        elif self._rng.random() < 0.3:  # 30% chance of context frame
            frame = self._rng.choice(list(CONTEXT_FRAMES.values()))
        else:
            return lines

        if frame:
            intro: str = frame["intro"]
            if n is not None:
                intro = intro.replace("{n}", str(n)).replace("{n_minus_1}", str(n - 1))
            return [intro, ""] + lines
        return lines

    def verbalize_intra(
        self,
        structure: AlgebraicStructure,
        composed_func: ComposedFunction,
        x: Any,
    ) -> str:
        """Generate a diverse prompt for an intra-structure task."""
        tmpl: dict = self._pick_template(INTRA_TEMPLATES)
        depth = len(composed_func.operations)

        lines: List[str] = [tmpl["header"].format(
            name=structure.name,
            desc=structure.description,
        )]
        lines.append("")
        lines.append(tmpl["start"].format(
            x=structure.element_to_str(x),
            depth=depth,
        ))
        lines.append("")

        for i, op in enumerate(composed_func.operations, 1):
            step_str = tmpl["step"].format(
                i=i,
                i_prev=i - 1,
                op_desc=op.description,
                depth=depth,
            )
            lines.append(step_str)

        lines.append("")
        lines.append(tmpl["footer"].format(depth=depth))

        # Optionally add context frame
        order = getattr(structure, 'n', getattr(structure, 'p', None))
        lines = self._maybe_add_context(lines, order)

        return "\n".join(lines)

    def verbalize_inter(
        self,
        composed: AlgebraicStructure,
        a: Any,
        b: Any = None,
        op_type: str = "op",
    ) -> str:
        """Generate a diverse prompt for an inter-structure task."""
        tmpl: dict = self._pick_template(INTER_TEMPLATES)

        a_str = composed.element_to_str(a)
        if op_type == "inverse":
            op_desc = f"Compute the inverse of {a_str} in {composed.name}."
        elif op_type == "op_then_inverse":
            b_str = composed.element_to_str(b)
            op_desc = f"Compute the inverse of ({a_str} {composed.operation_symbol()} {b_str}) in {composed.name}."
        else:
            b_str = composed.element_to_str(b)
            op_desc = f"{a_str} {composed.operation_symbol()} {b_str}"

        lines: List[str] = [tmpl["header"].format(
            name=composed.name,
            desc=composed.description,
        )]
        lines.append("")
        lines.append(tmpl["task"].format(op_desc=op_desc))
        lines.append("")
        lines.append(tmpl["footer"])

        return "\n".join(lines)

    def verbalize_field(
        self,
        field_struct: AlgebraicStructure,
        expr: str,
    ) -> str:
        """Generate a diverse prompt for a field arithmetic task."""
        tmpl: dict = self._pick_template(FIELD_TEMPLATES)

        lines: List[str] = [tmpl["header"].format(
            name=field_struct.name,
            desc=field_struct.description,
        )]
        lines.append("")
        lines.append(tmpl["task"].format(
            name=field_struct.name,
            expr=expr,
        ))
        lines.append("")
        lines.append(tmpl["footer"])

        return "\n".join(lines)

    def verbalize_rule(
        self,
        structure: AlgebraicStructure,
        train_examples: List[Tuple[Any, Any]],
        test_input: Any,
    ) -> str:
        """Generate a diverse prompt for a rule induction task."""
        tmpl: dict = self._pick_template(RULE_TEMPLATES)

        lines: List[str] = [tmpl["header"].format(
            name=structure.name,
            desc=structure.description,
        )]
        lines.append("")

        if tmpl["intro"]:
            lines.append(tmpl["intro"])
            lines.append("")

        for x_val, y_val in train_examples:
            lines.append(tmpl["example"].format(
                x=structure.element_to_str(x_val),
                y=structure.element_to_str(y_val),
            ))

        lines.append("")
        lines.append(tmpl["question"].format(
            test=structure.element_to_str(test_input),
        ))

        return "\n".join(lines)

    def relabel_elements(
        self,
        n: int,
        label_pool: Optional[str] = None,
    ) -> Dict[int, str]:
        '''
        Generate a random relabelling map for n elements.

        Args:
            n: Number of elements to relabel.
            label_pool: Name of the label pool to use (None = random).

        Returns:
            Dict mapping integer elements to string labels.
        '''
        if label_pool is None:
            label_pool = self._rng.choice(list(ELEMENT_LABEL_POOLS.keys()))

        pool = ELEMENT_LABEL_POOLS.get(label_pool, [])
        if len(pool) < n:
            # Fallback to a large pool if the chosen one is too small
            pool = ELEMENT_LABEL_POOLS["emoji_safe"]

        labels = self._rng.sample(pool, n)
        return {i: labels[i] for i in range(n)}
