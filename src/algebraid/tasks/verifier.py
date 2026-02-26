"""
Symbolic answer verification.

Handles multiple equivalent representations of algebraic elements
(e.g. ``(1 2 3)`` vs ``(1, 2, 3)``) and extracts answers from common
model output formats including LaTeX ``\\boxed{}``, Yes/No binary
answers, and multiple-choice letters (A/B/C/D).
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


def _extract_binary_answer(text: str) -> Optional[str]:
    """Return 'yes' or 'no' if the response unambiguously states one.

    Matches standalone True/False as well, mapping both to yes/no.
    Returns None if no clear binary answer is found.
    """
    t = text.strip().lower()
    # Exact single-token answer
    if t in ("yes", "no", "true", "false"):
        return "yes" if t in ("yes", "true") else "no"
    # "the answer is yes/no/true/false" anywhere in the text
    m = re.search(
        r'(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is\s*)?[:\s]*\b(yes|no|true|false)\b',
        t,
    )
    if m:
        val = m.group(1)
        return "yes" if val in ("yes", "true") else "no"
    # Standalone word at the very start (e.g. "Yes, because...")
    m = re.match(r'^(yes|no|true|false)\b', t)
    if m:
        val = m.group(1)
        return "yes" if val in ("yes", "true") else "no"
    return None


def _extract_multiple_choice(text: str) -> Optional[str]:
    """Return the selected letter (a/b/c/d) if clearly indicated.

    Returns None if no clear single-letter choice is found.
    """
    t = text.strip().lower()
    # "answer is B" / "option B" / "choice B" / "answer: B"
    m = re.search(
        r'\b(?:answer|option|choice)\s*(?:is\s*)?[:\s]*[(\[]?([abcd])[)\]]?',
        t,
    )
    if m:
        return m.group(1)
    # "(B)" or "[B]" or "B." as a standalone token
    m = re.search(r'[(\[]([abcd])[)\]][.:\s]', t)
    if m:
        return m.group(1)
    # Single letter at start of response
    m = re.match(r'^[(\[]?([abcd])[)\]]?[.:\s]', t)
    if m:
        return m.group(1)
    # Single letter at end of response (final answer format)
    m = re.search(r'[(\[]?([abcd])[)\]]?\.?\s*$', t)
    if m and len(t.split()) <= 5:  # only for short responses to avoid false positives
        return m.group(1)
    return None


def extract_answer(response: str) -> str:
    """Extract the final answer from a model's response."""
    text: str = response.strip()

    # Check for \\boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return normalize_answer(boxed_match.group(1))

    # Check for "Final Answer: X" (explicit format hint from prompt)
    final_matches = re.findall(r'final\s+answer\s*:\s*(.+)', text, re.IGNORECASE)
    if final_matches:
        candidate = final_matches[-1].strip()
        if candidate.lower() in ("yes", "no", "true", "false"):
            return "yes" if candidate.lower() in ("yes", "true") else "no"
        mc = _extract_multiple_choice(candidate.lower())
        if mc:
            return mc
        return normalize_answer(candidate)

    # Check for binary Yes/No answers before other patterns
    binary = _extract_binary_answer(text.lower())
    if binary is not None:
        return binary

    # Check for multiple-choice letter answers
    mc = _extract_multiple_choice(text.lower())
    if mc is not None:
        return mc

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
        if isinstance(result, tuple):
            return result
    except Exception:
        return None
    return None
