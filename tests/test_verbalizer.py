"""
Tests for the Verbalizer: all verbalize_* methods, template selection,
and the new conceptual/intermediate/verify/multiple-choice methods.
"""

import pytest

from algebraid.tasks.verbalizer import (
    Verbalizer, CONCEPTUAL_TEMPLATES, INTERMEDIATE_TEMPLATES,
    VERIFY_TEMPLATES, INTRA_TEMPLATES, RULE_TEMPLATES,
)
from algebraid.primitives import CyclicGroup, SymmetricGroup, QuaternionGroup
from algebraid.composers import make_standard_operations, ComposedFunction, DirectProduct


# ── Template pool integrity ───────────────────────────────────────────────────

class TestTemplatePools:
    def test_intra_templates_nonempty(self):
        assert len(INTRA_TEMPLATES) >= 5

    def test_rule_templates_nonempty(self):
        assert len(RULE_TEMPLATES) >= 5

    def test_conceptual_templates_all_subtypes(self):
        subtypes = [
            "identity", "element_order", "commutativity_check",
            "structure_order", "is_abelian", "inverse_of", "is_generator",
        ]
        for st in subtypes:
            assert st in CONCEPTUAL_TEMPLATES, f"Missing template for subtype: {st}"
            assert len(CONCEPTUAL_TEMPLATES[st]) >= 3

    def test_intermediate_templates_count(self):
        assert len(INTERMEDIATE_TEMPLATES) >= 3

    def test_verify_templates_count(self):
        assert len(VERIFY_TEMPLATES) >= 3

    def test_all_conceptual_templates_have_name_placeholder(self):
        for subtype, templates in CONCEPTUAL_TEMPLATES.items():
            for t in templates:
                assert "{name}" in t, f"Missing {{name}} in {subtype} template: {t}"

    def test_intermediate_templates_have_required_placeholders(self):
        for t in INTERMEDIATE_TEMPLATES:
            assert "{steps}" in t
            assert "{k}" in t
            assert "{x}" in t


# ── verbalize_intra ───────────────────────────────────────────────────────────

class TestVerbalizeIntra:
    def test_returns_nonempty_string(self, verbalizer, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0], ops[1]], z7)
        x = z7.random_element(rng)
        prompt = verbalizer.verbalize_intra(z7, chain, x)
        assert isinstance(prompt, str) and len(prompt) > 20

    def test_contains_structure_name(self, verbalizer, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0]], z7)
        x = 3
        prompt = verbalizer.verbalize_intra(z7, chain, x)
        assert "Z_7" in prompt or "Z 7" in prompt or "mod 7" in prompt

    def test_no_unresolved_placeholders(self, verbalizer, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0], ops[1]], z7)
        x = z7.random_element(rng)
        prompt = verbalizer.verbalize_intra(z7, chain, x)
        import re
        assert not re.search(r'\{[a-z_]+\}', prompt), "Unresolved template placeholder found"

    def test_with_skin(self, verbalizer, z7, rng):
        from algebraid.skins import SKIN_REGISTRY
        skin = SKIN_REGISTRY["CyclicGroup"][0]
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0]], z7)
        x = z7.random_element(rng)
        prompt = verbalizer.verbalize_intra(z7, chain, x, skin=skin)
        assert isinstance(prompt, str) and len(prompt) > 20

    def test_skin_suppresses_context_frame(self, z7, rng):
        """A skinned prompt must never contain a context-frame overlay.

        Context frames like "You are designing a cipher" are incoherent when
        the skin already provides its own narrative (e.g. seating arrangements).
        """
        from algebraid.skins import SKIN_REGISTRY
        from algebraid.tasks.verbalizer import CONTEXT_FRAMES
        context_intros = {v["intro"].split(".")[0].lower() for v in CONTEXT_FRAMES.values()}
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0], ops[1]], z7)
        x = z7.random_element(rng)
        for skin in SKIN_REGISTRY["CyclicGroup"]:
            # Force a context frame via context_frame= to maximise test pressure
            for frame_key in list(CONTEXT_FRAMES.keys()):
                v = Verbalizer(seed=0, context_frame=frame_key)
                prompt = v.verbalize_intra(z7, chain, x, skin=skin)
                intro_fragment = CONTEXT_FRAMES[frame_key]["intro"].split(".")[0].lower()
                # The context frame intro must NOT appear in the skinned prompt
                assert intro_fragment not in prompt.lower(), (
                    f"Context frame '{frame_key}' bled into skinned prompt for "
                    f"skin '{skin.name}': found '{intro_fragment}' in prompt"
                )

    def test_no_skin_allows_context_frame(self, z7, rng):
        """Without a skin, the context frame MUST be applied (when forced)."""
        from algebraid.tasks.verbalizer import CONTEXT_FRAMES
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0]], z7)
        x = z7.random_element(rng)
        # Use a frame that doesn't require {n} substitution
        v = Verbalizer(seed=0, context_frame="pure_math")
        prompt = v.verbalize_intra(z7, chain, x, skin=None)
        assert "Consider the following mathematical problem" in prompt

    def test_q8_verbalization(self, verbalizer, q8, rng):
        ops = make_standard_operations(q8, rng)
        chain = ComposedFunction([ops[0]], q8)
        x = q8.random_element(rng)
        prompt = verbalizer.verbalize_intra(q8, chain, x)
        assert isinstance(prompt, str) and len(prompt) > 10


# ── verbalize_conceptual ──────────────────────────────────────────────────────

class TestVerbalizeConceptual:
    def test_identity_prompt(self, verbalizer, z7):
        prompt = verbalizer.verbalize_conceptual(z7, "identity")
        assert isinstance(prompt, str) and len(prompt) > 10
        assert "Z_7" in prompt or "mod 7" in prompt

    def test_element_order_includes_x(self, verbalizer, z7):
        prompt = verbalizer.verbalize_conceptual(z7, "element_order", x=3)
        assert "3" in prompt

    def test_commutativity_check_includes_a_and_b(self, verbalizer, z7):
        prompt = verbalizer.verbalize_conceptual(z7, "commutativity_check", a=2, b=5)
        assert "2" in prompt and "5" in prompt

    def test_commutativity_asks_yes_or_no(self, verbalizer, z7):
        prompt = verbalizer.verbalize_conceptual(z7, "commutativity_check", a=2, b=5)
        assert "Yes or No" in prompt or "yes or no" in prompt.lower()

    def test_is_abelian_asks_yes_or_no(self, verbalizer, z7):
        prompt = verbalizer.verbalize_conceptual(z7, "is_abelian")
        assert "Yes or No" in prompt or "yes or no" in prompt.lower()

    def test_structure_order_prompt(self, verbalizer, q8):
        prompt = verbalizer.verbalize_conceptual(q8, "structure_order")
        assert "Q_8" in prompt or "quaternion" in prompt.lower()

    def test_no_unresolved_placeholders(self, verbalizer, z7):
        import re
        for subtype in ["identity", "inverse_of", "is_abelian", "structure_order"]:
            prompt = verbalizer.verbalize_conceptual(z7, subtype, x=3, a=1, b=2)
            assert not re.search(r'\{[a-zA-Z_]+\}', prompt), (
                f"Unresolved placeholder in {subtype} prompt: {prompt}"
            )

    def test_unknown_subtype_falls_back(self, verbalizer, z7):
        # Unknown subtype should use identity templates as fallback
        prompt = verbalizer.verbalize_conceptual(z7, "nonexistent_subtype")
        assert isinstance(prompt, str) and len(prompt) > 10


# ── verbalize_intermediate_state ─────────────────────────────────────────────

class TestVerbalizeIntermediateState:
    def test_returns_string(self, verbalizer, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0], ops[1], ops[2] if len(ops) > 2 else ops[0]], z7)
        x = z7.random_element(rng)
        prompt = verbalizer.verbalize_intermediate_state(z7, chain, x, query_step=1)
        assert isinstance(prompt, str) and len(prompt) > 20

    def test_includes_query_step_number(self, verbalizer, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0], ops[1]], z7)
        prompt = verbalizer.verbalize_intermediate_state(z7, chain, 3, query_step=1)
        assert "1" in prompt

    def test_includes_total_steps(self, verbalizer, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0], ops[1], ops[0]], z7)
        prompt = verbalizer.verbalize_intermediate_state(z7, chain, 3, query_step=2)
        assert "3" in prompt   # total steps = 3

    def test_does_not_ask_for_final_result(self, verbalizer, z7, rng):
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0], ops[1]], z7)
        prompt = verbalizer.verbalize_intermediate_state(z7, chain, 3, query_step=1)
        lower = prompt.lower()
        assert "after step 1" in lower or "step 1" in lower


# ── verbalize_verify ──────────────────────────────────────────────────────────

class TestVerbalizeVerify:
    def test_returns_string(self, verbalizer, z7):
        prompt = verbalizer.verbalize_verify(z7, "3 + 5 = 1")
        assert isinstance(prompt, str) and len(prompt) > 10

    def test_includes_claim(self, verbalizer, z7):
        prompt = verbalizer.verbalize_verify(z7, "i * j = k")
        assert "i * j = k" in prompt

    def test_asks_yes_or_no(self, verbalizer, z7):
        prompt = verbalizer.verbalize_verify(z7, "3 + 4 = 0")
        assert "Yes or No" in prompt or "yes or no" in prompt.lower()


# ── verbalize_multiple_choice ─────────────────────────────────────────────────

class TestVerbalizeMultipleChoice:
    def test_returns_string(self, verbalizer):
        choices = {"a": "1", "b": "2", "c": "3", "d": "4"}
        prompt = verbalizer.verbalize_multiple_choice("What is 1+1?", choices)
        assert isinstance(prompt, str)

    def test_contains_all_options(self, verbalizer):
        choices = {"a": "red", "b": "blue", "c": "green", "d": "yellow"}
        prompt = verbalizer.verbalize_multiple_choice("Choose a color:", choices)
        for label in ["(A)", "(B)", "(C)", "(D)"]:
            assert label in prompt

    def test_contains_all_values(self, verbalizer):
        choices = {"a": "0", "b": "3", "c": "5", "d": "6"}
        prompt = verbalizer.verbalize_multiple_choice("Choose:", choices)
        for val in choices.values():
            assert val in prompt

    def test_ends_with_select_instruction(self, verbalizer):
        choices = {"a": "1", "b": "2", "c": "3", "d": "4"}
        prompt = verbalizer.verbalize_multiple_choice("Pick one:", choices)
        assert "letter" in prompt.lower() or "select" in prompt.lower()


# ── verbalize_rule ────────────────────────────────────────────────────────────

class TestVerbalizeRule:
    def test_returns_string(self, verbalizer, z7):
        examples = [(0, 2), (1, 3), (2, 4)]
        prompt = verbalizer.verbalize_rule(z7, examples, test_input=5)
        assert isinstance(prompt, str) and len(prompt) > 20

    def test_contains_examples(self, verbalizer, z7):
        examples = [(0, 2), (1, 3)]
        prompt = verbalizer.verbalize_rule(z7, examples, test_input=4)
        assert "0" in prompt and "2" in prompt


# ── verbalize_inter ───────────────────────────────────────────────────────────

class TestVerbalizeInter:
    @pytest.fixture
    def z3xz4(self):
        return DirectProduct(CyclicGroup(3), CyclicGroup(4))

    def test_returns_nonempty_string(self, verbalizer, z3xz4):
        a = (1, 2)
        b = (2, 3)
        prompt = verbalizer.verbalize_inter(z3xz4, a, b, op_type="op")
        assert isinstance(prompt, str) and len(prompt) > 20

    def test_no_unresolved_placeholders(self, verbalizer, z3xz4):
        import re
        for op_type in ["op", "inverse", "op_then_inverse"]:
            prompt = verbalizer.verbalize_inter(z3xz4, (1, 2), (2, 3), op_type=op_type)
            assert not re.search(r'\{[a-z_]+\}', prompt), (
                f"Unresolved placeholder in {op_type} prompt: {prompt}"
            )

    def test_question_symbol_matches_description_symbol(self, verbalizer, z3xz4):
        """The operator used in the question must match the operator defined in the header.

        This is the regression for the DirectProduct symbol mixing bug:
        description defined '+', but the question used '*' or '.'.
        """
        sym = z3xz4.operation_symbol()  # should be '+'
        desc = z3xz4.description       # should also use '+'

        # Verify that the description itself is consistent first
        assert f") {sym} (" in desc, (
            f"description symbol mismatch: op_sym={sym!r}, desc={desc!r}"
        )

        # Generate prompts for all three op_types and verify no foreign symbol appears
        for op_type in ["op", "op_then_inverse"]:
            prompt = verbalizer.verbalize_inter(z3xz4, (1, 2), (2, 3), op_type=op_type)
            # The prompt contains the full description (via {desc}) and the question.
            # Neither '.' nor '*' should appear in the question line when sym is '+'.
            # We look for the question-specific fragment after the empty line.
            lines = prompt.split("\n")
            question_lines = [l for l in lines if any(
                marker in l for marker in ["Compute", "(1, 2)", "(2, 3)"]
            )]
            for qline in question_lines:
                assert "*" not in qline, (
                    f"op_type={op_type}: found '*' in question line when sym='{sym}': {qline!r}"
                )
                assert "· " not in qline, (
                    f"op_type={op_type}: found '·' in question line when sym='{sym}': {qline!r}"
                )

    def test_cyclic_product_uses_plus_not_dot(self, verbalizer, z3xz4):
        """Z_m x Z_n prompts must not use '.' as the outer operator anywhere."""
        for op_type in ["op", "op_then_inverse"]:
            prompt = verbalizer.verbalize_inter(z3xz4, (1, 2), (2, 3), op_type=op_type)
            # '. ' (dot-space) in a math context signals the wrong operator
            # Exclude the '...' ellipsis in natural text — look for paired dot like ') . ('
            assert ") . (" not in prompt, (
                f"op_type={op_type}: found ') . (' in cyclic product prompt:\n{prompt}"
            )


# ── Diversity: different calls produce different prompts ─────────────────────

class TestTemplateDiversity:
    def test_multiple_intra_prompts_differ(self, z7, rng):
        from algebraid.composers import make_standard_operations, ComposedFunction
        ops = make_standard_operations(z7, rng)
        chain = ComposedFunction([ops[0]], z7)
        x = 3
        prompts = set()
        for seed in range(10):
            v = Verbalizer(seed=seed)
            prompts.add(v.verbalize_intra(z7, chain, x))
        # With 6 templates and 8 context frames, expect at least 3 distinct prompts
        assert len(prompts) >= 3

    def test_multiple_conceptual_prompts_differ(self, z7):
        prompts = set()
        for seed in range(10):
            v = Verbalizer(seed=seed)
            prompts.add(v.verbalize_conceptual(z7, "identity"))
        assert len(prompts) >= 2
