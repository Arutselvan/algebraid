"""
Tests for the answer verification pipeline:
  - normalize_answer
  - _extract_binary_answer
  - _extract_multiple_choice
  - extract_answer (all formats)
  - check_answer (all formats)
"""

import pytest

from algebraid.tasks.verifier import (
    normalize_answer,
    extract_answer,
    check_answer,
    _extract_binary_answer,
    _extract_multiple_choice,
)


# ── normalize_answer ──────────────────────────────────────────────────────────

class TestNormalizeAnswer:
    def test_strips_whitespace(self):
        assert normalize_answer("  3  ") == "3"

    def test_lowercases(self):
        assert normalize_answer("YES") == "yes"

    def test_strips_trailing_period(self):
        assert normalize_answer("5.") == "5"

    def test_strips_trailing_comma(self):
        assert normalize_answer("5,") == "5"

    def test_collapses_spaces(self):
        assert normalize_answer("1,  2,  3") == "1, 2, 3"

    def test_normalizes_parens(self):
        assert normalize_answer("( 1, 2, 3 )") == "(1, 2, 3)"

    def test_passthrough_normal(self):
        assert normalize_answer("42") == "42"


# ── _extract_binary_answer ────────────────────────────────────────────────────

class TestExtractBinaryAnswer:
    # Exact matches
    def test_yes(self):
        assert _extract_binary_answer("yes") == "yes"

    def test_no(self):
        assert _extract_binary_answer("no") == "no"

    def test_true_maps_to_yes(self):
        assert _extract_binary_answer("true") == "yes"

    def test_false_maps_to_no(self):
        assert _extract_binary_answer("false") == "no"

    def test_case_insensitive(self):
        assert _extract_binary_answer("YES") == "yes"
        assert _extract_binary_answer("No") == "no"

    # Sentence patterns
    def test_the_answer_is_yes(self):
        assert _extract_binary_answer("The answer is yes.") == "yes"

    def test_the_answer_is_no(self):
        assert _extract_binary_answer("The final answer is no") == "no"

    def test_result_is_true(self):
        assert _extract_binary_answer("The result is true.") == "yes"

    # Start of sentence
    def test_yes_comma_because(self):
        assert _extract_binary_answer("Yes, because the group is abelian.") == "yes"

    def test_no_dot(self):
        assert _extract_binary_answer("No. The group is non-abelian.") == "no"

    # Non-binary returns None
    def test_none_for_number(self):
        assert _extract_binary_answer("42") is None

    def test_none_for_empty(self):
        assert _extract_binary_answer("") is None

    def test_none_for_tuple(self):
        assert _extract_binary_answer("(1, 2, 3)") is None


# ── _extract_multiple_choice ──────────────────────────────────────────────────

class TestExtractMultipleChoice:
    def test_answer_is_b(self):
        assert _extract_multiple_choice("The answer is B") == "b"

    def test_answer_colon_c(self):
        assert _extract_multiple_choice("Answer: C") == "c"

    def test_option_d(self):
        assert _extract_multiple_choice("option D") == "d"

    def test_choice_a(self):
        assert _extract_multiple_choice("choice A") == "a"

    def test_parenthesized_b(self):
        assert _extract_multiple_choice("(B) is the correct answer") == "b"

    def test_bracketed_c(self):
        assert _extract_multiple_choice("[C]. because it equals 5.") == "c"

    def test_standalone_a_at_start(self):
        assert _extract_multiple_choice("A. the identity element") == "a"

    def test_standalone_b_at_end_short(self):
        assert _extract_multiple_choice("The answer is b") == "b"

    def test_none_for_number(self):
        assert _extract_multiple_choice("42") is None

    def test_none_for_yes(self):
        assert _extract_multiple_choice("yes") is None

    def test_case_insensitive(self):
        assert _extract_multiple_choice("ANSWER IS B") == "b"


# ── extract_answer ────────────────────────────────────────────────────────────

class TestExtractAnswer:
    # LaTeX boxed
    def test_boxed(self):
        assert extract_answer("\\boxed{42}") == "42"

    def test_boxed_tuple(self):
        assert extract_answer("\\boxed{(1, 2, 3)}") == "(1, 2, 3)"

    # Binary answers extracted before other patterns
    def test_yes_response(self):
        assert extract_answer("Yes") == "yes"

    def test_no_response(self):
        assert extract_answer("No") == "no"

    def test_binary_in_sentence(self):
        assert extract_answer("The answer is yes.") == "yes"

    # Multiple choice
    def test_mc_b(self):
        assert extract_answer("The answer is B") == "b"

    def test_mc_parenthesized(self):
        assert extract_answer("(C)") == "c"

    # "the answer is X"
    def test_answer_is_pattern(self):
        assert extract_answer("The answer is 5") == "5"

    def test_result_is_pattern(self):
        assert extract_answer("The result is (2, 3)") == "(2, 3)"

    # equals at end
    def test_equals_at_end(self):
        assert extract_answer("3 + 2 = 5") == "5"

    # Last non-empty line
    def test_last_line_fallback(self):
        assert extract_answer("Let me compute...\n\n4") == "4"

    def test_multiline_last_line(self):
        # The last-line fallback returns the full normalized last line.
        result = extract_answer("Step 1: 3\nStep 2: 6\nFinal: 2")
        assert result == "final: 2"


# ── check_answer ──────────────────────────────────────────────────────────────

class TestCheckAnswer:
    # Exact numeric matches
    def test_exact_match(self):
        assert check_answer("4", "4") is True

    def test_exact_match_after_normalize(self):
        assert check_answer("  4  ", "4") is True

    def test_wrong_number(self):
        assert check_answer("5", "4") is False

    # Numeric comparison
    def test_float_match(self):
        assert check_answer("3.0", "3") is True

    # Tuple comparison
    def test_tuple_match(self):
        assert check_answer("(1, 2, 3)", "(1, 2, 3)") is True

    def test_tuple_different_spacing(self):
        assert check_answer("(1,2,3)", "(1, 2, 3)") is True

    def test_wrong_tuple(self):
        assert check_answer("(1, 3, 2)", "(1, 2, 3)") is False

    # Binary answers
    def test_yes_yes(self):
        assert check_answer("Yes", "yes") is True

    def test_no_no(self):
        assert check_answer("No", "no") is True

    def test_true_yes(self):
        assert check_answer("True", "yes") is True

    def test_false_no(self):
        assert check_answer("False", "no") is True

    def test_yes_no_mismatch(self):
        assert check_answer("Yes", "no") is False

    def test_binary_in_sentence(self):
        assert check_answer("The answer is yes, because Z_7 is abelian.", "yes") is True

    # Multiple choice
    def test_mc_correct_letter(self):
        assert check_answer("The answer is B", "b") is True

    def test_mc_wrong_letter(self):
        assert check_answer("The answer is A", "b") is False

    def test_mc_parenthesized(self):
        assert check_answer("(C)", "c") is True

    # Boxed answer
    def test_boxed_correct(self):
        assert check_answer("\\boxed{5}", "5") is True

    # Substring match (non-strict)
    def test_substring_match(self):
        assert check_answer("the value is 5 in the group", "5") is True

    # Strict mode
    def test_strict_no_substring(self):
        assert check_answer("the value is 5 in the group", "5", strict=True) is False

    def test_strict_exact(self):
        assert check_answer("5", "5", strict=True) is True

    # Dihedral/symmetric group elements (string answers)
    def test_permutation_answer(self):
        assert check_answer("(2, 1, 3)", "(2, 1, 3)") is True

    # Quaternion string elements
    def test_quaternion_element_i(self):
        assert check_answer("i", "i") is True

    def test_quaternion_element_neg_k(self):
        assert check_answer("-k", "-k") is True
