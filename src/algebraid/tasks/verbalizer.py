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

# -- Entity-name pools -------------------------------------------------------

STRUCTURE_ALIASES: List[str] = [
    "system", "structure", "set with operation", "algebraic system",
    "mathematical group", "operation table", "number system",
    "closed system", "abstract algebra", "computational domain",
]

ELEMENT_LABEL_POOLS: Dict[str, List[str]] = {
    "greek": list("alphaβγδεζηθικλμνξοπρστυφχψω"),
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

# -- Template pools ----------------------------------------------------------

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

# -- Conceptual query templates -----------------------------------------------
# Keyed by query_subtype; each has 3+ surface-form variants.

CONCEPTUAL_TEMPLATES: Dict[str, List[str]] = {
    "identity": [
        "In {name} {short_desc}, what is the identity element under {sym}? Give only the answer.",
        "What element e in {name} {short_desc} satisfies e {sym} x = x {sym} e = x for every element x? State only the answer.",
        "Name the identity element of {name} {short_desc}. Give only the answer.",
    ],
    "element_order": [
        "In {name} {short_desc}, what is the order of the element {x}?\n(The order is the smallest positive integer k such that applying the operation k times to {x} returns the identity.)\nGive only the integer answer.",
        "How many times must you apply {sym} to {x} with itself (i.e. compute {x}, {x}{sym}{x}, {x}{sym}{x}{sym}{x}, ...) before you first reach the identity in {name} {short_desc}? Give only the count.",
        "Find ord({x}) in {name} {short_desc}. Give only the integer answer.",
    ],
    "commutativity_check": [
        "In {name} {short_desc}, does {a} {sym} {b} equal {b} {sym} {a}? Answer Yes or No.",
        "Working in {name} {short_desc}: is the operation {sym} commutative for the specific pair ({a}, {b})? Answer Yes or No.",
        "Does the order of applying {sym} matter for {a} and {b} in {name} {short_desc}? That is, is {a} {sym} {b} = {b} {sym} {a}? Answer Yes or No.",
    ],
    "structure_order": [
        "How many elements does {name} {short_desc} contain? Give only the integer answer.",
        "What is |{name}|, the order of the group {name} {short_desc}? Give only the integer.",
        "State the order (number of elements) of the group {name} {short_desc}. Give only the integer.",
    ],
    "is_abelian": [
        "Is {name} {short_desc} an abelian (commutative) group? Answer Yes or No.",
        "Does every pair of elements in {name} {short_desc} satisfy a {sym} b = b {sym} a? Answer Yes or No.",
        "Is the group {name} {short_desc} commutative? Answer Yes or No.",
    ],
    "inverse_of": [
        "In {name} {short_desc}, what is the inverse of {x} under {sym}? Give only the answer.",
        "Find {x}^(-1) in {name} {short_desc}. Give only the answer.",
        "Which element y in {name} {short_desc} satisfies {x} {sym} y = e, where e is the identity? Give only the answer.",
    ],
    "is_generator": [
        "Is {x} a generator of {name} {short_desc}? (A generator is an element whose repeated application reaches every element.) Answer Yes or No.",
        "Can every element of {name} {short_desc} be expressed as a multiple (repeated application) of {x}? Answer Yes or No.",
        "Does {x} generate the entire group {name} {short_desc}? Answer Yes or No.",
    ],
}

# -- Intermediate-state templates ---------------------------------------------

INTERMEDIATE_TEMPLATES: List[str] = [
    (
        "In {name} {short_desc}, starting with x = {x}, apply the following {total} operations in order:\n"
        "{steps}\n"
        "What is the value immediately after step {k}? (Do not continue past step {k}.)\n"
        "Give only the answer."
    ),
    (
        "Working in {name} {short_desc} with initial value {x}:\n"
        "{steps}\n"
        "Report the intermediate result right after step {k} has been applied. Give only the answer."
    ),
    (
        "Consider this computation in {name} {short_desc}. Start: {x}.\n"
        "{steps}\n"
        "What value does the element hold after exactly {k} of these {total} operations? Give only the answer."
    ),
]

# -- Verify-format templates ---------------------------------------------------

VERIFY_TEMPLATES: List[str] = [
    "In {name} {short_desc}: is it true that {claim}? Answer Yes or No.",
    "Working in {name} {short_desc}  - does the following hold? {claim}\nAnswer Yes or No.",
    "True or False (in {name} {short_desc}): {claim}\nAnswer Yes or No.",
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

    def _maybe_add_context(
        self, lines: List[str], n: Any = None, skin: Optional[SemanticSkin] = None
    ) -> List[str]:
        """Optionally wrap the prompt in a context frame.

        Suppressed when a semantic skin is active — skins provide their own
        coherent narrative and mixing a context frame on top would create
        incoherent prompts (e.g. "You are designing a cipher" followed by
        "the possible seating arrangements of Alice, Bob, Carol, Dave").
        """
        if skin is not None:
            return lines  # skin owns the narrative; no overlay allowed

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
            elif "{n}" in intro:
                # Frame requires {n} but structure has no such attribute - skip frame
                return lines
            return [intro, ""] + lines
        return lines

    def _clean(self, text: str) -> str:
        """Normalize whitespace and collapse accidental multi-period sequences."""
        text = re.sub(r'\.{2,}', '.', text)
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text

    def _format_hint_chain(self, depth: int, x_str: str) -> str:
        """Step-by-step output template for chain tasks (intra / adversarial)."""
        lines = ["Show your work step by step:"]
        lines.append(f"Start: {x_str}")
        for i in range(1, depth + 1):
            lines.append(f"Step {i}: ___")
        lines.append("Final Answer: \\boxed{}")
        return "\n".join(lines)

    def _format_hint_intermediate(self, total: int, k: int, x_str: str) -> str:
        """Step-by-step output template for intermediate-state tasks (stop at step k)."""
        lines = [f"Show your work step by step (stop after step {k}):"]
        lines.append(f"Start: {x_str}")
        for i in range(1, k + 1):
            lines.append(f"Step {i}: ___")
        if k < total:
            lines.append(f"(Do not continue past step {k}.)")
        lines.append("Final Answer: \\boxed{}")
        return "\n".join(lines)

    def _format_hint_simple(self, answer_type: str) -> str:
        """Single-line answer format hint for non-chain tasks."""
        placeholders = {
            "integer": "\\boxed{integer}",
            "element": "\\boxed{element}",
            "yes_no": "Yes or No",          # binary: box notation is unnatural
            "tuple": "\\boxed{(a, b)}",
        }
        ph = placeholders.get(answer_type, "\\boxed{answer}")
        return f"Final Answer: {ph}"

    def verbalize_intra(
        self,
        structure: AlgebraicStructure,
        composed_func: ComposedFunction,
        x: Any,
        skin: Optional[SemanticSkin] = None,
        note: Optional[str] = None,
    ) -> str:
        """Generate a diverse prompt for an intra-structure task.

        Args:
            note: Optional one-line adversarial instruction inserted before the
                  format hint (e.g. "Apply all operations in order.").
        """
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
        if note:
            lines.append(note)
            lines.append("")
        lines.append(self._format_hint_chain(depth, x_str))

        order = getattr(structure, 'n', getattr(structure, 'p', None))
        lines = self._maybe_add_context(lines, order, skin=skin)

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

        sym = composed.operation_symbol()
        if op_type == "inverse":
            op_desc = f"Compute the inverse of {a_str} in {composed_name}."
        elif op_type == "op_then_inverse":
            b_str_val = composed.element_to_str(b) if skin is None else skin.element_name(b, composed)
            op_desc = f"Compute the inverse of ({a_str} {sym} {b_str_val}) in {composed_name}."
        else:
            b_str_val = composed.element_to_str(b) if skin is None else skin.element_name(b, composed)
            op_desc = f"{a_str} {sym} {b_str_val}"

        lines: List[str] = [tmpl["header"].format(
            name=composed_name,
            desc=composed.description,
        )]
        lines.append("")
        lines.append(tmpl["task"].format(op_desc=op_desc))
        lines.append("")
        lines.append(self._format_hint_simple("tuple"))

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
            expr=expr,
        )]
        if tmpl["task"]:  # Some templates have empty task strings
            lines.append("")
            lines.append(tmpl["task"].format(
                name=structure_name,
                expr=expr,
            ))
        lines.append("")
        lines.append(self._format_hint_simple("integer"))

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
        lines.append("")
        lines.append(self._format_hint_simple("element"))

        return self._clean("\n".join(lines))

    def verbalize_conceptual(
        self,
        structure: AlgebraicStructure,
        query_subtype: str,
        x: Any = None,
        a: Any = None,
        b: Any = None,
        skin: Optional[SemanticSkin] = None,
    ) -> str:
        """Generate a diverse prompt for a conceptual query task."""
        templates = CONCEPTUAL_TEMPLATES.get(query_subtype, CONCEPTUAL_TEMPLATES["identity"])
        tmpl = self._rng.choice(templates)

        structure_name = skin.structure_name(structure) if skin else structure.name
        sym = structure.operation_symbol()

        x_str = structure.element_to_str(x) if x is not None else "x"
        a_str = structure.element_to_str(a) if a is not None else "a"
        b_str = structure.element_to_str(b) if b is not None else "b"

        prompt = tmpl.format(
            name=structure_name,
            short_desc=structure.short_description,
            sym=sym,
            x=x_str,
            a=a_str,
            b=b_str,
        )

        # When * is used as the operation symbol in an unskinned prompt, explicitly
        # disambiguate it from ordinary arithmetic multiplication.
        if sym == "*" and skin is None:
            prompt = "Let * denote the group operation.\n" + prompt

        _YES_NO_SUBTYPES = {"commutativity_check", "is_abelian", "is_generator"}
        _INTEGER_SUBTYPES = {"element_order", "structure_order"}
        if query_subtype in _YES_NO_SUBTYPES:
            hint = self._format_hint_simple("yes_no")
        elif query_subtype in _INTEGER_SUBTYPES:
            hint = self._format_hint_simple("integer")
        else:  # identity, inverse_of
            hint = self._format_hint_simple("element")
        return self._clean(prompt + "\n" + hint)

    def verbalize_intermediate_state(
        self,
        structure: AlgebraicStructure,
        composed_func: Any,
        x: Any,
        query_step: int,
        skin: Optional[SemanticSkin] = None,
    ) -> str:
        """Generate a prompt asking for the value at a specific step in a chain."""
        tmpl = self._rng.choice(INTERMEDIATE_TEMPLATES)

        structure_name = skin.structure_name(structure) if skin else structure.name
        x_str = skin.element_name(x, structure) if skin else structure.element_to_str(x)
        total = len(composed_func.operations)

        if skin:
            op_descs = [
                skin.op_description(op.name, op.fixed_args, structure)
                for op in composed_func.operations
            ]
        else:
            op_descs = [op.description for op in composed_func.operations]

        steps = "\n".join(f"Step {i + 1}: {desc}." for i, desc in enumerate(op_descs))

        prompt = tmpl.format(
            name=structure_name,
            short_desc=structure.short_description,
            x=x_str,
            total=total,
            steps=steps,
            k=query_step,
        )
        hint = self._format_hint_intermediate(total, query_step, x_str)
        return self._clean(prompt + "\n\n" + hint)

    def verbalize_verify(
        self,
        structure: AlgebraicStructure,
        claim_str: str,
        skin: Optional[SemanticSkin] = None,
    ) -> str:
        """Generate a Yes/No verification prompt around a given claim string."""
        tmpl = self._rng.choice(VERIFY_TEMPLATES)
        structure_name = skin.structure_name(structure) if skin else structure.name

        prompt = tmpl.format(
            name=structure_name,
            short_desc=structure.short_description,
            claim=claim_str,
        )
        return self._clean(prompt)

    def verbalize_multiple_choice(
        self,
        base_prompt: str,
        choices: Dict[str, str],
    ) -> str:
        """Append A/B/C/D choices to an existing prompt and return the modified prompt."""
        choice_lines = "\n".join(
            f"({letter.upper()}) {value}"
            for letter, value in sorted(choices.items())
        )
        return self._clean(base_prompt.rstrip() + "\n\n" + choice_lines + "\n\nSelect one letter.")

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
