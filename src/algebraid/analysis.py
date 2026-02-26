"""
Error analysis suite for ALGEBRAID evaluation results.

Addresses paper Blocker 2 (New Insight) by providing analyses beyond
simple accuracy-vs-depth tables:

  fit_scaling_law()      Power-law decay fit:  acc(d) ≈ A · depth^(−α)
  find_phase_transition() Steepest accuracy drop and first sub-50% depth
  error_taxonomy()       Mechanistic failure-mode classification
  hallucination_onset()  Depth at which refusal/nonsense rate rises
  stability_breakdown()  Per-depth curve enriched with taxonomy + fitted values
  run_analysis()         Consolidated report dict (all of the above)
  print_analysis()       Human-readable console output

All functions accept an EvalReport from evaluator.py (or a list of them
for compare_models()).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .evaluator import EvalReport, EvalResult
from .task_model import CompositionDimension


# ── 1. Error Scaling Law ─────────────────────────────────────────────────────

def fit_scaling_law(report: EvalReport) -> Dict[str, Any]:
    """
    Fit a power-law decay to accuracy as a function of composition depth.

        acc(d) ≈ A · d^(−α)

    Fit via ordinary least-squares in log-log space:

        log(acc) = log(A) − α · log(d)

    Returns
    -------
    A            Pre-factor  (model accuracy extrapolated to depth 1)
    alpha        Decay exponent  (α > 0 → accuracy falls with depth)
    r2           Coefficient of determination
    interpretation  Human-readable sentence
    data         Per-depth {depth, accuracy, fitted} list for plotting
    """
    depths = sorted(report.accuracy_by_depth.keys())
    accs = [report.accuracy_by_depth[d]["accuracy"] for d in depths]

    # Need depth > 0 and accuracy > 0 for log transform
    pairs = [(d, a) for d, a in zip(depths, accs) if d > 0 and a > 0]
    if len(pairs) < 2:
        return {
            "A": None, "alpha": None, "r2": None, "data": [],
            "note": "Insufficient data for fit (need >= 2 depths with non-zero accuracy).",
        }

    log_d = [math.log(d) for d, _ in pairs]
    log_a = [math.log(a) for _, a in pairs]
    n = len(pairs)

    mean_ld = sum(log_d) / n
    mean_la = sum(log_a) / n
    ss_xx = sum((x - mean_ld) ** 2 for x in log_d)
    ss_xy = sum((x - mean_ld) * (y - mean_la) for x, y in zip(log_d, log_a))

    if ss_xx == 0:
        return {
            "A": None, "alpha": None, "r2": None, "data": [],
            "note": "Zero variance in depths; cannot fit.",
        }

    alpha = -ss_xy / ss_xx           # slope in log-log space is −α
    log_A = mean_la - (-alpha) * mean_ld
    A = math.exp(log_A)

    ss_res = sum((la - (log_A - alpha * ld)) ** 2 for ld, la in zip(log_d, log_a))
    ss_tot = sum((la - mean_la) ** 2 for la in log_a)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Percentage accuracy drop when depth doubles: 1 − 2^(−α)
    halving_drop = (1 - 2 ** (-alpha)) * 100

    data = [
        {
            "depth": d,
            "accuracy": round(a, 4),
            "fitted": round(min(1.0, A * d ** (-alpha)), 4),
        }
        for d, a in zip(depths, accs)
    ]

    return {
        "A": round(A, 4),
        "alpha": round(alpha, 4),
        "r2": round(r2, 4),
        "interpretation": (
            f"acc(d) ~= {A:.3f} * d^(-{alpha:.3f}).  "
            f"Each doubling of depth reduces accuracy by ~{halving_drop:.1f}%."
        ),
        "data": data,
    }


# ── 2. Phase Transition ──────────────────────────────────────────────────────

def find_phase_transition(report: EvalReport) -> Dict[str, Any]:
    """
    Identify the composition depth at which accuracy collapses most sharply.

    Two complementary signals:
      steepest_drop_at_depth — depth d where |acc[d] − acc[d+1]| is largest
      first_sub50_depth      — first depth where accuracy < 50 %

    critical_depth = minimum of the two (earliest warning).
    """
    depths = sorted(report.accuracy_by_depth.keys())
    accs = {d: report.accuracy_by_depth[d]["accuracy"] for d in depths}

    if len(depths) < 2:
        return {"critical_depth": None, "note": "Need at least 2 depth levels."}

    drops = [
        (depths[i], abs(accs[depths[i]] - accs[depths[i + 1]]))
        for i in range(len(depths) - 1)
    ]
    steepest_depth, steepest_drop = max(drops, key=lambda x: x[1])
    collapse_depth = next((d for d in depths if accs[d] < 0.50), None)

    signals = [steepest_depth]
    if collapse_depth is not None:
        signals.append(collapse_depth)
    critical = min(signals)

    deltas = [
        {
            "from_depth": d,
            "to_depth": depths[i + 1],
            "drop": round(abs(accs[d] - accs[depths[i + 1]]), 4),
        }
        for i, d in enumerate(depths[:-1])
    ]

    return {
        "critical_depth": critical,
        "steepest_drop_at_depth": steepest_depth,
        "steepest_drop_magnitude": round(steepest_drop, 4),
        "first_sub50_depth": collapse_depth,
        "accuracy_at_critical": round(accs.get(critical, float("nan")), 4),
        "drops": deltas,
    }


# ── 3. Mechanistic Error Taxonomy ────────────────────────────────────────────

_HALLUCINATION_RE = re.compile(
    r"cannot|undefined|infinity|idk|unknown|impossible|not defined|n/a|none|sorry|don.t know",
    re.IGNORECASE,
)


def _to_num(s: str) -> Optional[float]:
    """Try to extract a leading number from a string."""
    try:
        token = s.strip().split()[0] if s.strip() else ""
        cleaned = re.sub(r"[^0-9\-\.]", "", token)
        return float(cleaned) if cleaned else None
    except (ValueError, IndexError):
        return None


def _classify_error(result: EvalResult) -> str:
    """
    Classify a single wrong answer into a mechanistic failure mode.

    Categories
    ----------
    off_by_one          Numeric answer differs from ground truth by exactly 1.
    inverse_confusion   Model returns the additive inverse of the correct answer
                        (a + resp ~= 0 mod something).
    commutativity_swap  Error on an adversarial/commutativity-trap task.
    identity_confusion  Model answers with an identity-like token (0, 1, "e").
    hallucination       Response contains refusal or nonsense tokens.
    other               Numeric error >= 2 or structurally unclassifiable.
    """
    resp = result.model_response.strip()

    if _HALLUCINATION_RE.search(resp):
        return "hallucination"

    resp_num = _to_num(resp)
    gt_num = _to_num(result.ground_truth)

    if resp_num is not None and gt_num is not None:
        diff = abs(resp_num - gt_num)
        if diff == 1:
            return "off_by_one"
        if resp_num != 0 and abs(resp_num + gt_num) <= 1:
            return "inverse_confusion"

    if resp.lower() in ("0", "1", "e", "identity", "(0)", "(1)"):
        return "identity_confusion"

    dim_val = result.dimension
    adv_val = CompositionDimension.ADVERSARIAL.value
    if dim_val == adv_val:
        return "commutativity_swap"

    return "other"


def error_taxonomy(report: EvalReport) -> Dict[str, Any]:
    """
    Classify every incorrect prediction into a mechanistic failure mode.

    Returns
    -------
    total_errors   Total number of wrong predictions
    categories     {category: {count, pct}} sorted by frequency
    by_depth       {depth: {category: count}}
    dominant_error The most common failure mode
    """
    wrong = [r for r in report.results if not r.correct]
    if not wrong:
        return {"total_errors": 0, "categories": {}, "by_depth": {}, "dominant_error": None}

    counts: Dict[str, int] = defaultdict(int)
    by_depth: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in wrong:
        cat = _classify_error(r)
        counts[cat] += 1
        by_depth[r.depth][cat] += 1

    total = len(wrong)
    categories = {
        cat: {"count": cnt, "pct": round(cnt / total * 100, 1)}
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1])
    }

    return {
        "total_errors": total,
        "categories": categories,
        "by_depth": {d: dict(cats) for d, cats in sorted(by_depth.items())},
        "dominant_error": max(counts, key=counts.__getitem__) if counts else None,
    }


# ── 4. Hallucination Onset ───────────────────────────────────────────────────

def hallucination_onset(report: EvalReport, threshold: float = 0.15) -> Dict[str, Any]:
    """
    Estimate the depth at which hallucination becomes prevalent.

    Hallucination is defined as a response matching _HALLUCINATION_RE.
    The onset depth is the first depth where the hallucination fraction
    (wrong-and-hallucinated / total-at-depth) exceeds *threshold*.

    Returns onset_depth, per-depth curve, and a plain-language note.
    """
    by_depth: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "hall": 0})

    for r in report.results:
        by_depth[r.depth]["total"] += 1
        if not r.correct and _HALLUCINATION_RE.search(r.model_response):
            by_depth[r.depth]["hall"] += 1

    curve = []
    onset_depth = None
    for depth in sorted(by_depth.keys()):
        d = by_depth[depth]
        rate = d["hall"] / d["total"] if d["total"] > 0 else 0.0
        curve.append({
            "depth": depth,
            "hallucination_rate": round(rate, 4),
            "hallucinations": d["hall"],
            "total": d["total"],
        })
        if onset_depth is None and rate >= threshold:
            onset_depth = depth

    return {
        "onset_depth": onset_depth,
        "threshold": threshold,
        "curve": curve,
        "note": (
            f"Hallucination rate first exceeds {threshold:.0%} at depth {onset_depth}."
            if onset_depth is not None else
            f"Hallucination rate never exceeded {threshold:.0%} across all depths tested."
        ),
    }


# ── 5. Stability Breakdown Curve ─────────────────────────────────────────────

def stability_breakdown(report: EvalReport) -> List[Dict[str, Any]]:
    """
    Return per-depth accuracy data enriched with error taxonomy and fitted values.

    Each entry: depth, accuracy, correct, total, fitted_accuracy,
                errors_by_category.

    Suitable for plotting Figure 1 of the paper.
    """
    law = fit_scaling_law(report)
    fitted_map = {row["depth"]: row["fitted"] for row in law.get("data", [])}
    by_depth_errors = error_taxonomy(report).get("by_depth", {})

    return [
        {
            "depth": depth,
            "accuracy": round(d["accuracy"], 4),
            "correct": d["correct"],
            "total": d["total"],
            "fitted_accuracy": fitted_map.get(depth),
            "errors_by_category": by_depth_errors.get(depth, {}),
        }
        for depth, d in sorted(report.accuracy_by_depth.items())
    ]


# ── 6. Consolidated report ───────────────────────────────────────────────────

def run_analysis(report: EvalReport) -> Dict[str, Any]:
    """Run all five analyses and return a single consolidated dict."""
    return {
        "model": report.model_name,
        "task_set": report.task_set_name,
        "overall_accuracy": round(report.accuracy_overall, 4),
        "scaling_law": fit_scaling_law(report),
        "phase_transition": find_phase_transition(report),
        "error_taxonomy": error_taxonomy(report),
        "hallucination_onset": hallucination_onset(report),
        "stability_curve": stability_breakdown(report),
    }


def print_analysis(analysis: Dict[str, Any]) -> None:
    """Print a human-readable analysis summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ALGEBRAID Analysis Report")
    print(f"  Model    : {analysis['model']}")
    print(f"  Task Set : {analysis['task_set']}")
    print(f"  Overall  : {analysis['overall_accuracy']:.1%}")
    print(sep)

    law = analysis["scaling_law"]
    if law.get("alpha") is not None:
        print(f"\n  Error Scaling Law  (acc ~= A * depth^(-alpha)):")
        print(f"    A     = {law['A']}  (pre-factor)")
        print(f"    alpha = {law['alpha']}  (decay exponent)")
        print(f"    R^2   = {law['r2']}")
        print(f"    {law['interpretation']}")
    else:
        print(f"\n  Error Scaling Law: {law.get('note', 'unavailable')}")

    pt = analysis["phase_transition"]
    if pt.get("critical_depth") is not None:
        print(f"\n  Phase Transition:")
        print(f"    Critical depth       : {pt['critical_depth']}")
        print(f"    Steepest drop        : {pt['steepest_drop_magnitude']:.1%}"
              f"  (at depth {pt['steepest_drop_at_depth']}->{pt['steepest_drop_at_depth']+1})")
        print(f"    First sub-50% depth  : {pt['first_sub50_depth']}")

    tax = analysis["error_taxonomy"]
    if tax["total_errors"] > 0:
        print(f"\n  Error Taxonomy ({tax['total_errors']} wrong predictions):")
        for cat, info in tax["categories"].items():
            bar = "#" * int(info["pct"] / 5)
            print(f"    {cat:<25} {info['count']:>4}  ({info['pct']:>5.1f}%)  {bar}")
        print(f"    Dominant: {tax['dominant_error']}")
    else:
        print(f"\n  Error Taxonomy: no errors (perfect accuracy).")

    ho = analysis["hallucination_onset"]
    print(f"\n  Hallucination Onset: {ho['note']}")

    print(f"\n  Stability Breakdown Curve:")
    print(f"  {'Depth':>5}  {'Acc':>6}  {'Fitted':>6}  {'N':>5}  Top error")
    print(f"  {'-'*50}")
    for row in analysis["stability_curve"]:
        fitted = f"{row['fitted_accuracy']:.4f}" if row['fitted_accuracy'] is not None else "  -   "
        top_err = max(row["errors_by_category"], key=row["errors_by_category"].get) \
                  if row["errors_by_category"] else "-"
        print(f"  {row['depth']:>5}  {row['accuracy']:>6.4f}  {fitted:>6}  "
              f"{row['total']:>5}  {top_err}")

    print(f"\n{sep}\n")
