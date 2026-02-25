"""
Natural-language verbalization for ALGEBRAID tasks.

Provides template-based prompt generation with multiple surface forms,
entity-name randomization, and context frames to ensure that models are
tested on reasoning rather than pattern matching.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from ..primitives.base import AlgebraicStructure
from ..skins import SemanticSkin
from ..composers.function_composition import AlgebraicOperation, ComposedFunction

# ── Entity-name pools ───────────────────────────────────────────────────────

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

# ── Template pools ──────────────────────────────────────────────────────────

INTRA_TEMPLATES: List[Dict[str, str]] = [
    {
        "header": "Consider the algebraic structure {name} {short_desc}.",
        "start": "Starting with the element x = {x}, perform the following operations in order:",
        "step": "Step {i}: {op_desc}.",
        "footer": "What is the final result? Give only the answer.",
    },
    {
        "header": "You are working in {name} {short_desc}.",
        "start": "Begin with x = {x}. Execute these operations sequentially:",
        "step": "{i}. {op_desc}",
        "footer": "Report the final value. Answer with just the number.",
    },
    {
        "header": "In the algebraic structure {name} {short_desc}, a computation unfolds as follows:",
        "start": "The initial value is {x}. The following transformations are applied one after another:",
        "step": "Transformation {i}: {op_desc}.",
        "footer": "What value remains at the end? State only the answer.",
    },
    {
        "header": "Let G = {name} {short_desc}.",
        "start": "Given x_0 = {x}, compute x_{depth} by applying these steps in sequence:",
        "step": "Step {i}: {op_desc}. Call the result x_{i}.",
        "footer": "What is the value of x_{depth}? Provide the answer only.",
    },
    {
        "header": "In the algebraic system {name} {short_desc}, what is the final result of the following sequence?",
        "start": "Start with {x} and apply these operations one by one, in order:",
        "step": "{i}) {op_desc}",
        "footer": "Final answer:",
    },
    {
        "header": "I need help with a calculation in {name} {short_desc}.",
        "start": "If I start with {x} and do the following:",
        "step": "- {op_desc}",
        "footer": "What do I end up with?",
    },
]

INTER_TEMPLATES: List[Dict[str, str]] = [
    {
        "header": "Consider the direct product group {name}. {desc}",
        "task": "Calculate: {op_desc}",
        "footer": "What is the resulting tuple?",
    },
    {
        "header": "Working in the product group {name}. {desc}",
        "task": "{op_desc}",
        "footer": "Express the answer as a tuple.",
    },
    {
        "header": "In {name}, solve the following. {desc}",
        "task": "{op_desc}",
        "footer": "Provide the tuple answer only.",
    },
    {
        "header": "Problem: {name}. {desc}",
        "task": "Compute {op_desc}",
        "footer": "Answer (tuple):",
    },
    {
        "header": "Let G = {name}. {desc}",
        "task": "Evaluate: {op_desc}",
        "footer": "Give the result as a tuple.",
    },
]

FIELD_TEMPLATES: List[Dict[str, str]] = [
    {
        "header": "Consider the finite field {name} {short_desc}.",
        "task": "Compute the following expression: {expr}",
        "footer": "What is the result? Give only the numerical answer.",
    },
    {
        "header": "In {name} {short_desc}, evaluate:",
        "task": "{expr}",
        "footer": "Answer (single number):",
    },
    {
        "header": "Working in {name} {short_desc}, what is {expr}?",
        "task": "",
        "footer": "Give only the numerical result.",
    },
    {
        "header": "Calculate the following in the finite field {name} {short_desc}.",
        "task": "Expression: {expr}",
        "footer": "Final numerical answer:",
    },
    {
        "header": "Problem: Evaluate an arithmetic expression in {name} {short_desc}.",
        "task": "{expr}",
        "footer": "What does this equal? State only the number.",
    },
]

RULE_TEMPLATES: List[Dict[str, str]] = [
    {
        "header": "Consider the algebraic structure {name} {short_desc}.",
        "intro": "An unknown function f maps elements of this structure to other elements. Here are some examples of f:",
        "example": "  f({x}) = {y}",
        "question": "What is f({test})? Give only the answer.",
    },
    {
        "header": "In {name} {short_desc}, a mystery transformation T is defined by the following input-output pairs:",
        "intro": "",
        "example": "  T({x}) = {y}",
        "question": "Predict T({test}). Answer with just the value.",
    },
    {
        "header": "A function f operates on elements of {name} {short_desc}.",
        "intro": "You are given these observations:",
        "example": "  Input: {x}  ->  Output: {y}",
        "question": "What output does input {test} produce? Give only the answer.",
    },
    {
        "header": "Pattern recognition in {name} {short_desc}.",
        "intro": "The following mappings have been observed:",
        "example": "  {x} -> {y}",
        "question": "Following the same pattern, what does {test} map to?",
    },
    {
        "header": "Decode the rule. Working in {name} {short_desc}.",
        "intro": "Clues:",
        "example": "  f({x}) = {y}",
        "question": "Apply the rule to {test}. State only the answer.",
    },
]


class Verbalizer:
    """
    Generates diverse natural language prompts for ALGEBRAID tasks.

    Usage:
        verb = Verbalizer(seed=42)
        prompt = verb.verbalize_intra(structure, composed_func, x, skin=skin)
        prompt = verb.verbalize_inter(composed, a, b, op_type="op")
        prompt = verb.verbalize_field(field_struct, expr)
        prompt = verb.verbalize_rule(structure, examples, test_input)
    """

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
            frame: Optional[Dict[str, str]] = CONTEXT_FRAMES.get(self.context_frame)
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

    def _clean(self, text: str) -> str:
        """Normalize whitespace and fix stray double-periods."""
        text = re.sub(r'(?<!,\s)\.{2,}(?!,)', '.', text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text

    def verbalize_intra(
        self,
        structure: AlgebraicStructure,
        composed_func: ComposedFunction,
        x: Any,
        skin: Optional[SemanticSkin] = None,
    ) -> str:
        """Generate a diverse prompt for an intra-structure task."""
        tmpl: dict = self._pick_template(INTRA_TEMPLATES)
        depth = len(composed_func.operations)

        if skin:
            structure_name = skin.structure_name(structure)
            x_str = skin.element_name(x, structure)
            op_descs = [
                skin.op_description(op.name, op.fixed_args, structure)
                for op in composed_func.operations
            ]
        else:
            structure_name = structure.name
            x_str = structure.element_to_str(x)
            op_descs = [op.description for op in composed_func.operations]

        lines: List[str] = [tmpl["header"].format(
            name=structure_name,
            short_desc=structure.short_description,
        )]
        lines.append("")
        lines.append(tmpl["start"].format(
            x=x_str,
            depth=depth,
        ))
        lines.append("")

        for i, op_desc in enumerate(op_descs, 1):
            step_str = tmpl["step"].format(
                i=i,
                i_prev=i - 1,
                op_desc=op_desc,
                depth=depth,
            )
            lines.append(step_str)

        lines.append("")
        lines.append(tmpl["footer"].format(depth=depth))

        order = getattr(structure, 'n', getattr(structure, 'p', None))
        lines = self._maybe_add_context(lines, order)

        return self._clean("\n".join(lines))

    def verbalize_inter(
        self,
        composed: AlgebraicStructure,
        a: Any,
        b: Any = None,
        op_type: str = "op",
        skin: Optional[SemanticSkin] = None,
    ) -> str:
        """Generate a diverse prompt for an inter-structure task."""
        tmpl: dict = self._pick_template(INTER_TEMPLATES)

        if skin:
            a_str = skin.element_name(a, composed)
            b_str = skin.element_name(b, composed) if b is not None else None
            composed_name = skin.structure_name(composed)
        else:
            a_str = composed.element_to_str(a)
            b_str = composed.element_to_str(b) if b is not None else None
            composed_name = composed.name

        if op_type == "inverse":
            op_desc = f"Compute the inverse of {a_str} in {composed_name}."
        elif op_type == "op_then_inverse":
            b_str_val = composed.element_to_str(b) if skin is None else skin.element_name(b, composed)
            op_desc = f"Compute the inverse of ({a_str} * {b_str_val}) in {composed_name}."
        else:
            b_str_val = composed.element_to_str(b) if skin is None else skin.element_name(b, composed)
            op_desc = f"{a_str} {composed.operation_symbol()} {b_str_val}"

        lines: List[str] = [tmpl["header"].format(
            name=composed_name,
            desc=composed.description,
        )]
        lines.append("")
        lines.append(tmpl["task"].format(op_desc=op_desc))
        lines.append("")
        lines.append(tmpl["footer"])

        return self._clean("\n".join(lines))

    def verbalize_field(
        self,
        field_struct: AlgebraicStructure,
        expr: str,
        skin: Optional[SemanticSkin] = None,
    ) -> str:
        """Generate a diverse prompt for a field arithmetic task."""
        tmpl: dict = self._pick_template(FIELD_TEMPLATES)

        structure_name = skin.structure_name(field_struct) if skin else field_struct.name

        lines: List[str] = [tmpl["header"].format(
            name=structure_name,
            short_desc=field_struct.short_description,
        )]
        if tmpl["task"]:  # Some templates have empty task strings
            lines.append("")
            lines.append(tmpl["task"].format(
                name=structure_name,
                expr=expr,
            ))
        lines.append("")
        lines.append(tmpl["footer"])

        return self._clean("\n".join(lines))

    def verbalize_rule(
        self,
        structure: AlgebraicStructure,
        train_examples: List[Tuple[Any, Any]],
        test_input: Any,
        skin: Optional[SemanticSkin] = None,
    ) -> str:
        """Generate a diverse prompt for a rule induction task."""
        tmpl: dict = self._pick_template(RULE_TEMPLATES)

        structure_name = skin.structure_name(structure) if skin else structure.name

        lines: List[str] = [tmpl["header"].format(
            name=structure_name,
            short_desc=structure.short_description,
        )]
        lines.append("")

        if tmpl["intro"]:
            lines.append(tmpl["intro"])

        for x_val, y_val in train_examples:
            x_str = skin.element_name(x_val, structure) if skin else structure.element_to_str(x_val)
            y_str = skin.element_name(y_val, structure) if skin else structure.element_to_str(y_val)
            lines.append(tmpl["example"].format(x=x_str, y=y_str))

        lines.append("")
        test_str = skin.element_name(test_input, structure) if skin else structure.element_to_str(test_input)
        lines.append(tmpl["question"].format(test=test_str))

        return self._clean("\n".join(lines))

    def relabel_elements(
        self,
        num_elements: int,
        label_pool: str = "greek",
    ) -> Dict[int, str]:
        """
        Generate a random relabeling map for elements.

        Args:
            num_elements: Number of elements to relabel.
            label_pool: Name of the label pool to use from ELEMENT_LABEL_POOLS.

        Returns:
            Mapping from integer elements to string labels.
        """
        pool_list = ELEMENT_LABEL_POOLS.get(label_pool, ELEMENT_LABEL_POOLS["greek"])
        if len(pool_list) < num_elements:
            return {i: str(i) for i in range(num_elements)}
        pool = self._rng.sample(pool_list, num_elements)
        return {i: str(pool[i]) for i in range(num_elements)}
