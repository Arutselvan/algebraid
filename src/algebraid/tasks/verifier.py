"""
Symbolic answer verification.

Handles multiple equivalent representations of algebraic elements
(e.g. ``(1 2 3)`` vs ``(1, 2, 3)``) and extracts answers from common
model output formats including LaTeX ``\\boxed{}``, Yes/No binary
answers, and multiple-choice letters (A/B/C/D).

Reasoning-model robustness notes
---------------------------------
* ``<think>...</think>`` blocks are stripped before any extraction so that
  chain-of-thought text does not pollute the last-line fallback.
* ``<answer>...</answer>`` tags are checked first (higher priority than boxed).
* ``\\boxed{}`` takes the **last** match — a reasoning model may write wrong
  intermediate boxed values before correcting itself.
* The "answer/option/choice is X" patterns also take the **last** match for
  the same reason.
* The ``truth in extracted`` substring check requires ``len(truth) >= 3`` and
  word-boundary isolation to prevent single-digit false positives (e.g. truth
  ``"3"`` must not match inside extracted ``"13"``).
* ``_extract_think_conclusion`` extracts the final answer stated at the **tail
  of the think block** (e.g. "Thus answer: X").  ``check_answer`` tests this
  as a second candidate alongside the post-think extraction, so a concise
  scratchpad conclusion can score correct even when the formatted response is
  verbose or wraps the value in extra prose.
"""

import ast
import math
import re
from typing import Optional


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    s: str = answer.strip().lower()
    # Normalize Unicode minus sign (U+2212) to ASCII hyphen-minus
    s = s.replace('\u2212', '-')
    s = s.rstrip(".,")
    # Strip markdown bold/italic markers: **answer** -> answer, *answer* -> answer
    s = re.sub(r'\*+', '', s)
    # Strip LaTeX \text{...} wrappers: \text{Card 2} -> Card 2
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    # Normalize LaTeX escaped spaces: card\ 2 -> card 2
    s = s.replace('\\ ', ' ')
    s = " ".join(s.split())
    s = re.sub(r'\s*,\s*', ', ', s)
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    return s


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning scratchpad blocks.

    DeepSeek-R1, QwQ, and similar models wrap chain-of-thought in these
    tags; the final answer appears after the closing tag.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()


def _extract_think_conclusion(text: str) -> Optional[str]:
    """Extract a clean answer stated at the tail of the last <think> block.

    Reasoning models often conclude their scratchpad with a terse statement
    like ``"Thus final answer: 2."`` or ``"So answer is flipped."`` before
    writing a verbose formatted response.  This gives a cleaner candidate than
    the post-think output for answers that get wrapped in extra prose or LaTeX.

    Returns None if no think block or no conclusive statement is found.
    """
    blocks = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    # Use the last think block's tail (last 6 non-empty lines)
    lines = [l.strip() for l in blocks[-1].split('\n') if l.strip()]
    tail = '\n'.join(lines[-6:])
    if not tail:
        return None

    # "Final answer: X" / "Thus answer: X" / "So answer is X" — last match
    final_matches = re.findall(r'final\s+answer\s*[:\s]\s*(.+)', tail, re.IGNORECASE)
    if final_matches:
        return normalize_answer(final_matches[-1].split('\n')[0])

    answer_matches = re.findall(
        r'(?:thus|so|therefore)[\s,]*(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is|=|:)\s*(.+)',
        tail, re.IGNORECASE,
    )
    if answer_matches:
        return normalize_answer(answer_matches[-1].split('\n')[0])

    return None


def _extract_binary_answer(text: str) -> Optional[str]:
    """Return 'yes' or 'no' if the response unambiguously states one.

    Matches standalone True/False as well, mapping both to yes/no.
    Returns None if no clear binary answer is found.

    Takes the **last** occurrence of "answer/result is yes/no" so that a
    model that reconsiders mid-reasoning is scored on its final conclusion.
    """
    t = text.strip().lower()
    # Strip markdown bold/italic markers so "**Answer:** Yes" -> "answer: yes"
    t = re.sub(r'\*+', '', t)
    # Exact single-token answer
    if t in ("yes", "no", "true", "false"):
        return "yes" if t in ("yes", "true") else "no"
    # "the answer is yes/no/true/false" — take LAST match
    matches = re.findall(
        r'(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is\s*)?[:\s]*\b(yes|no|true|false)\b',
        t,
    )
    if matches:
        val = matches[-1]
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

    Takes the **last** "answer/option/choice is X" occurrence so that a
    model that eliminates options before selecting one is scored correctly.
    """
    t = text.strip().lower()
    # "answer is B" / "option B" / "choice B" / "answer: B" — take LAST match
    matches = re.findall(
        r'\b(?:answer|option|choice)\s*(?:is\s*)?[:\s]*[(\[]?([abcd])[)\]]?',
        t,
    )
    if matches:
        return matches[-1]
    # "(B)" or "[B]" or "B." as a standalone token
    m = re.search(r'[(\[]([abcd])[)\]][.:\s]', t)
    if m:
        return m.group(1)
    # Single letter at start of response
    m = re.match(r'^[(\[]?([abcd])[)\]]?[.:\s]', t)
    if m:
        return m.group(1)
    # Single letter at end of response (final answer format).
    # The (?<!\w) lookbehind prevents matching a letter inside a word
    # (e.g. "bob]" must not extract "b").
    m = re.search(r'(?<!\w)[(\[]?([abcd])[)\]]?\.?\s*$', t)
    if m and len(t.split()) <= 5:  # only for short responses to avoid false positives
        return m.group(1)
    return None


def extract_answer(response: str) -> str:
    """Extract the final answer from a model's response."""
    # Strip reasoning scratchpad blocks before any extraction
    text: str = _strip_think_blocks(response.strip())

    # Check for <answer>...</answer> tags (some instruction-tuned models)
    answer_tags = re.findall(r'<answer>\s*(.+?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    if answer_tags:
        return normalize_answer(answer_tags[-1])

    # Check for \\boxed{...} — take LAST occurrence (reasoning models may
    # write wrong intermediate values before correcting themselves).
    # Pattern allows one level of nested braces to handle \text{...} inside lists.
    boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text)
    if boxed_matches:
        return normalize_answer(boxed_matches[-1])

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

    # Check for "the answer is X" or "the result is X" — take LAST match
    answer_matches = re.findall(
        r'(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is|=)\s*(.+)',
        text, re.IGNORECASE,
    )
    if answer_matches:
        return normalize_answer(answer_matches[-1].split("\n")[0])

    # Check for "= X" at the end
    equals_match = re.search(r'=\s*(.+?)\.?\s*$', text, re.MULTILINE)
    if equals_match:
        eq_candidate: str = equals_match.group(1).strip()
        if len(eq_candidate) < 50:
            return normalize_answer(eq_candidate)

    # Take the last non-empty line
    lines: list = [s for s in text.split("\n") if s.strip()]
    if lines:
        return normalize_answer(lines[-1])

    return normalize_answer(text)


def parse_and_verify(model_response: str, ground_truth: str) -> bool:
    """Alias for check_answer for backward compatibility."""
    return check_answer(model_response, ground_truth)


def _candidate_matches(extracted: str, truth: str, strict: bool) -> bool:
    """Return True if a single extracted candidate matches the ground truth."""
    if extracted == truth:
        return True
    if not strict and len(truth) >= 3:
        if re.search(r'(?<!\w)' + re.escape(truth) + r'(?!\w)', extracted):
            return True
    try:
        e_float = float(extracted.replace(",", ""))
        t_float = float(truth.replace(",", ""))
        # Reject inf, nan, and scientific notation — algebra answers are plain integers
        if (math.isfinite(e_float) and math.isfinite(t_float)
                and "e" not in extracted.lower() and "e" not in truth.lower()
                and e_float == t_float):
            return True
    except (ValueError, TypeError):
        pass
    try:
        extracted_tuple = _parse_tuple(extracted)
        truth_tuple = _parse_tuple(truth)
        if extracted_tuple is not None and truth_tuple is not None:
            if extracted_tuple == truth_tuple:
                return True
    except Exception:
        pass
    return False


def check_answer(
    model_response: str,
    ground_truth: str,
    strict: bool = False,
) -> bool:
    """Check if a model's response matches the ground truth.

    Extraction strategy (in order):
    1. Post-think output — primary source; ``extract_answer`` strips
       ``<think>`` blocks and applies all extraction patterns to what remains.
    2. Think-block conclusion — fallback used only when the primary extraction
       does not match.  Reasoning models often state a terse conclusion at the
       tail of their scratchpad (e.g. "Thus answer: 2.") before writing a
       verbose formatted response.  If the post-think text fails to match, this
       cleaner candidate is checked as a second attempt.
    """
    extracted: str = extract_answer(model_response)
    truth: str = normalize_answer(ground_truth)

    if _candidate_matches(extracted, truth, strict):
        return True

    # Fallback: try the terse conclusion from the tail of the think block
    think_conclusion: Optional[str] = _extract_think_conclusion(model_response)
    if think_conclusion is not None and _candidate_matches(think_conclusion, truth, strict):
        return True

    return False


def _parse_tuple(s: str) -> Optional[tuple]:
    """Try to parse a string as a nested tuple of integers.

    Accepts both tuple ``(a, b, c)`` and list ``[a, b, c]`` notation so that
    models which output bracket-delimited permutations are not penalised.
    Rejects tuples containing non-integer elements (floats, strings, etc.).
    """
    s = s.strip()
    if not s:
        return None
    try:
        result = ast.literal_eval(s)
        if isinstance(result, (tuple, list)):
            flat = _flatten_to_list(result)
            # Only accept if all leaf values are ints (not floats or strings)
            if all(isinstance(v, int) and not isinstance(v, bool) for v in flat):
                return tuple(result) if isinstance(result, list) else result
    except Exception:
        return None
    return None


def _flatten_to_list(t) -> list:
    """Recursively flatten nested tuples/lists to a flat list of leaf values."""
    out: list = []
    if isinstance(t, (tuple, list)):
        for item in t:
            out.extend(_flatten_to_list(item))
    else:
        out.append(t)
    return out


_Q8_ELEMENTS = frozenset({"1", "-1", "i", "-i", "j", "-j", "k", "-k"})


def _quaternion_canonical(response: str) -> Optional[str]:
    """Normalize alternative Q_8 notation to canonical element strings.

    Handles common model-output variants:
      "+i" -> "i",  "+1" -> "1"        (strip leading +)
      "e"  -> "1"                       (identity alias)
      "1i" / "1j" / "1k" -> "i" / "j" / "k"   (strip unit coefficient)
      "-1i" / "-1j" / "-1k" -> "-i" / "-j" / "-k"

    Returns the canonical Q_8 element string, or None if not parseable.
    """
    raw = extract_answer(response).strip()
    # Already canonical?
    if raw in _Q8_ELEMENTS:
        return raw
    # Strip leading '+'
    if raw.startswith("+"):
        raw = raw[1:]
    # Identity aliases
    if raw == "e":
        raw = "1"
    elif raw == "-e":
        raw = "-1"
    # "-1i" -> "-i", "-1j" -> "-j", "-1k" -> "-k"
    m = re.fullmatch(r'-1([ijk])', raw)
    if m:
        raw = f"-{m.group(1)}"
    # "1i" -> "i", "1j" -> "j", "1k" -> "k"
    m = re.fullmatch(r'1([ijk])', raw)
    if m:
        raw = m.group(1)
    return raw if raw in _Q8_ELEMENTS else None


def _dihedral_canonical(response: str, n: int) -> Optional[str]:
    """Convert alternative dihedral notation to canonical form for comparison.

    In D_n the relation s*r^k = r^(n-k)*s means ``s r^k`` and ``r^{n-k}s``
    name the same element.  Models trained on group theory often output the
    ``s r^k`` form while the generator uses the canonical ``r^ks`` form.

    Also handles spacing variants (``r^2 s``) and LaTeX composition notation
    (``r^2 \\circ s``).

    Returns the canonical ``r^ks`` / ``s`` / ``e`` string if the response
    parses as a dihedral element in any recognised notation, else None.
    """
    raw = extract_answer(response).strip()

    # Normalise LaTeX: remove \circ, strip surrounding $
    raw = re.sub(r'\\circ', '', raw)
    raw = re.sub(r'^\$+|\$+$', '', raw).strip()
    # Uppercase R_k / R^k -> r^k  (some models write R_4 for r^4)
    raw = re.sub(r'\bR_(\d+)\b', r'r^\1', raw)
    raw = re.sub(r'\bR\^(\d+)\b', r'r^\1', raw)
    # Map 'f' (flip) -> 's' (reflection) -- alternate standard notation for D_n
    raw = re.sub(r'\bf\b', 's', raw)
    # "r^k s" -> "r^ks",  "r s" -> "r^1s"
    raw = re.sub(r'(r\^\d+)\s+s\b', r'\1s', raw)
    raw = re.sub(r'\br\s+s\b', 'r^1s', raw)
    # Remove remaining spaces
    raw = raw.replace(' ', '')

    # Already canonical: e, s, r^k, r^ks -- but reduce exponent mod n
    if re.fullmatch(r'e|s|r\^\d+|r\^\d+s', raw):
        m_canon = re.fullmatch(r'r\^(\d+)(s?)', raw)
        if m_canon:
            k = int(m_canon.group(1)) % n
            has_s = m_canon.group(2)
            if k == 0:
                return "s" if has_s else "e"
            return f"r^{k}{has_s}"
        return raw

    # s r^k (collapsed to sr^k) -> r^{n-k}s
    m = re.fullmatch(r'sr\^(\d+)', raw)
    if m:
        k = int(m.group(1))
        r_val = (n - k) % n
        return "s" if r_val == 0 else f"r^{r_val}s"

    # bare 'rs' -> 'r^1s'
    if raw == 'rs':
        return "r^1s"

    return None
