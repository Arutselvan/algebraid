"""
Symbolic answer verification.

Handles multiple equivalent representations of algebraic elements
(e.g. ``(1 2 3)`` vs ``(1, 2, 3)``) and extracts answers from common
model output formats including LaTeX ``\\boxed{}``.
"""

import re
from typing import Any, Optional


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    s: str = answer.strip().lower()
    s = s.rstrip(".,")
    s = " ".join(s.split())
    s = re.sub(r'\s*,\s*', ', ', s)
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    return s


def extract_answer(response: str) -> str:
    """Extract the final answer from a model's response."""
    text: str = response.strip()

    # Check for \\boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return normalize_answer(boxed_match.group(1))

    # Check for "the answer is X" or "the result is X"
    answer_match = re.search(
        r'(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is|=)\s*(.+)',
        text, re.IGNORECASE
    )
    if answer_match:
        return normalize_answer(answer_match.group(1).split("\n")[0])

    # Check for "= X" at the end
    equals_match = re.search(r'=\s*(.+?)\.?\s*$', text, re.MULTILINE)
    if equals_match:
        candidate: str = equals_match.group(1).strip()
        if len(candidate) < 50:
            return normalize_answer(candidate)

    # Take the last non-empty line
    lines: list = [s for s in text.split("\n") if s.strip()]
    if lines:
        return normalize_answer(lines[-1])

    return normalize_answer(text)


def parse_and_verify(model_response: str, ground_truth: str) -> bool:
    """Alias for check_answer for backward compatibility."""
    return check_answer(model_response, ground_truth)


def check_answer(
    model_response: str,
    ground_truth: str,
    strict: bool = False,
) -> bool:
    """Check if a model's response matches the ground truth."""
    extracted: str = extract_answer(model_response)
    truth: str = normalize_answer(ground_truth)

    # Exact match after normalization
    if extracted == truth:
        return True

    # Check if truth is contained in extracted (non-strict mode)
    if not strict and truth in extracted:
        return True

    # Numeric comparison
    try:
        if float(extracted.replace(",", "")) == float(truth.replace(",", "")):
            return True
    except (ValueError, TypeError):
        pass

    # Tuple comparison
    try:
        extracted_tuple = _parse_tuple(extracted)
        truth_tuple = _parse_tuple(truth)
        if extracted_tuple is not None and truth_tuple is not None:
            if extracted_tuple == truth_tuple:
                return True
    except Exception:
        pass

    return False


def _parse_tuple(s: str) -> Optional[tuple]:
    """Try to parse a string as a nested tuple of integers."""
    s = s.strip()
    if not s:
        return None
    try:
        result = eval(s, {"__builtins__": {}}, {})
        if isinstance(result, (tuple, int)):
            return result
    except Exception:
        return None
    return None
