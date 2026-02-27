"""
Error analysis suite for ALGEBRAID evaluation results.

Six structured analyses are available via run_analysis():

  accuracy_by_depth       Per-depth accuracy curve (chain families) + per-family breakdown
  accuracy_by_family      Flat per-family accuracy across all task families
  accuracy_by_dimension   Flat per-dimension accuracy across all compositional dimensions
  complexity_analysis     Algebraic complexity metrics by depth + per-task vs accuracy
  error_taxonomy          Mechanistic failure-mode classification (five specific categories,
                          no catch-all "other")
  hallucination_onset     Depth at which hallucination/refusal rate first exceeds threshold

Standalone advanced functions (not in run_analysis output):

  fit_scaling_law()           Pooled power-law fit across chain families
  fit_scaling_law_by_family() Independent power-law fit per chain family
  find_phase_transition()     Steepest accuracy drop and first sub-50% depth

Depth-based analyses operate only on chain families (intra-structure, inter-structure,
field arithmetic) — adversarial and intermediate dimensions are excluded because their
depths do not reflect monotonic difficulty scaling.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .evaluator import EvalReport, EvalResult, CHAIN_FAMILIES, CHAIN_EXCLUDED_DIMENSIONS
from .task_model import CompositionDimension


# Module-level aliases for readability in this file.
_CHAIN_FAMILIES = CHAIN_FAMILIES
_CHAIN_EXCLUDED_DIMENSIONS = CHAIN_EXCLUDED_DIMENSIONS


def _chain_results(report: EvalReport) -> List[EvalResult]:
    """Return only the results from genuine chain families.

    Filters by family AND excludes adversarial/intermediate dimensions,
    since both share the 'intra-structure composition' family label but
    have depths that do not reflect monotonic difficulty scaling.
    """
    return [
        r for r in report.results
        if r.family in _CHAIN_FAMILIES
        and r.dimension not in _CHAIN_EXCLUDED_DIMENSIONS
    ]


def _depth_stats_from_results(results: List[EvalResult]) -> Dict[int, Dict[str, Any]]:
    """Compute per-depth accuracy stats from a list of EvalResult objects."""
    stats: Dict[int, Dict[str, Any]] = {}
    for r in results:
        if r.depth not in stats:
            stats[r.depth] = {"correct": 0, "total": 0}
        stats[r.depth]["total"] += 1
        if r.correct:
            stats[r.depth]["correct"] += 1
    for v in stats.values():
        v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0
    return stats


# -- Standalone Advanced: Error Scaling Law ------------------------------------

def _fit_power_law(depth_stats: Dict[int, Dict]) -> Dict[str, Any]:
    """
    Core OLS power-law fit on a pre-computed depth_stats dict.

    Fits  acc(d) ~= A * d^(-alpha)  via OLS in log-log space.

    Returns a dict with keys: A, alpha, alpha_se, r2, interpretation, data,
    and optionally a 'note' key for data-quality warnings.  On failure to fit
    (too few points, zero variance) A and alpha are None.
    """
    depths = sorted(depth_stats.keys())
    accs = [depth_stats[d]["accuracy"] for d in depths]

    # Need depth > 0 and accuracy > 0 for log transform
    pairs = [(d, a) for d, a in zip(depths, accs) if d > 0 and a > 0]
    if len(pairs) < 2:
        return {
            "A": None, "alpha": None, "alpha_se": None, "r2": None, "data": [],
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
            "A": None, "alpha": None, "alpha_se": None, "r2": None, "data": [],
            "note": "Zero variance in depths; cannot fit.",
        }

    alpha = -ss_xy / ss_xx           # slope in log-log space is -alpha
    log_A = mean_la - (-alpha) * mean_ld
    A = math.exp(log_A)

    ss_res = sum((la - (log_A - alpha * ld)) ** 2 for ld, la in zip(log_d, log_a))
    ss_tot = sum((la - mean_la) ** 2 for la in log_a)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard error of alpha (from OLS theory): se = sqrt(MSE / ss_xx)
    alpha_se: Optional[float] = (
        math.sqrt(ss_res / (n - 2) / ss_xx) if n > 2 else None
    )

    # Build interpretation based on alpha sign and R² quality.
    if alpha <= 0:
        interpretation = (
            f"Accuracy does not decrease with depth "
            f"(alpha={alpha:.3f}, R\u00b2={r2:.3f})."
        )
    else:
        halving_drop = (1 - 2 ** (-alpha)) * 100
        if r2 >= 0.95:
            fit_quality = f"Strong fit (R\u00b2={r2:.3f})."
        elif r2 >= 0.80:
            fit_quality = f"Moderate fit (R\u00b2={r2:.3f}); interpret with caution."
        else:
            fit_quality = (
                f"Weak fit (R\u00b2={r2:.3f}) \u2014 the power-law form may not "
                "describe this data well."
            )
        interpretation = (
            f"acc(d) \u2248 {A:.3f} \u00d7 d^(-{alpha:.3f}).  "
            f"Each doubling of depth reduces accuracy by ~{halving_drop:.1f}%.  "
            f"{fit_quality}"
        )

    data = [
        {
            "depth": d,
            "accuracy": round(a, 4),
            "fitted": round(min(1.0, A * d ** (-alpha)), 4) if alpha > 0 else None,
        }
        for d, a in zip(depths, accs)
    ]

    result: Dict[str, Any] = {
        "A": round(A, 4),
        "alpha": round(alpha, 4),
        "alpha_se": round(alpha_se, 4) if alpha_se is not None else None,
        "r2": round(r2, 4),
        "interpretation": interpretation,
        "data": data,
    }
    if n < 8:
        result["note"] = (
            f"Fit based on only {n} depth levels. For a reliable power-law "
            "estimate consider using generate_productivity_suite (depths up to 20)."
        )
    return result


def fit_scaling_law(report: EvalReport) -> Dict[str, Any]:
    """
    Fit a pooled power-law decay to accuracy across all chain families.

        acc(d) ~= A * d^(-alpha)

    Pools intra-structure, inter-structure, and field arithmetic results.
    Note: depth semantics differ across these families (sequential chain
    length vs. product dimensionality vs. expression-tree height).  Use
    fit_scaling_law_by_family() for family-stratified fits.

    Returns
    -------
    A              Pre-factor
    alpha          Decay exponent (alpha > 0 -> accuracy falls with depth)
    alpha_se       Standard error of alpha (None when < 3 data points)
    r2             Coefficient of determination
    interpretation Human-readable sentence with fit quality grade
    data           Per-depth {depth, accuracy, fitted} list
    families       The families pooled in this fit
    note           Data-quality warning if < 8 depth levels
    """
    chain = _chain_results(report)
    if not chain:
        return {
            "A": None, "alpha": None, "alpha_se": None, "r2": None, "data": [],
            "families": sorted(_CHAIN_FAMILIES),
            "note": "No chain-family results found.",
        }

    depth_stats = _depth_stats_from_results(chain)
    result = _fit_power_law(depth_stats)
    result["families"] = sorted(_CHAIN_FAMILIES)
    return result


def fit_scaling_law_by_family(report: EvalReport) -> Dict[str, Any]:
    """
    Fit an independent power-law for each chain family.

    Returns {family_name: fit_dict} for intra-structure, inter-structure,
    and field arithmetic separately.  Each fit_dict has the same keys as
    fit_scaling_law() (minus 'families').

    This is the preferred analysis when comparing depth-difficulty scaling
    across families because each family has a distinct notion of 'depth'.
    """
    results: Dict[str, Any] = {}
    for family in sorted(_CHAIN_FAMILIES):
        family_results = [
            r for r in report.results
            if r.family == family
            and r.dimension not in _CHAIN_EXCLUDED_DIMENSIONS
        ]
        if not family_results:
            results[family] = {
                "A": None, "alpha": None, "alpha_se": None, "r2": None,
                "data": [], "note": "No results for this family.",
            }
            continue
        depth_stats = _depth_stats_from_results(family_results)
        results[family] = _fit_power_law(depth_stats)
    return results


# -- Standalone Advanced: Phase Transition -------------------------------------

def find_phase_transition(report: EvalReport) -> Dict[str, Any]:
    """
    Identify the composition depth at which accuracy collapses most sharply.

    Two complementary signals:
      steepest_drop_at_depth - depth d where |acc[d] - acc[d+1]| is largest
      first_sub50_depth      - first depth where accuracy < 50 %

    critical_depth = minimum of the two (earliest warning).

    Operates on chain families only (same rationale as fit_scaling_law).
    """
    chain = _chain_results(report)
    depth_stats = _depth_stats_from_results(chain)
    depths = sorted(depth_stats.keys())
    accs = {d: depth_stats[d]["accuracy"] for d in depths}

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


# -- 1. Error Taxonomy ---------------------------------------------------------

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
    adversarial_trap    Error on any adversarial-dimension task (double_inverse,
                        self_cancelling, identity_bait, commutativity_trap).
    hallucination       Response contains refusal or nonsense tokens.
    off_by_one          Numeric answer differs from ground truth by exactly 1.
    inverse_confusion   Model returns the additive inverse of the correct answer.
    identity_confusion  Model answers with an identity-like token (0, 1, "e").
    wrong_value         Any other structured but incorrect response.
    """
    resp = result.model_response.strip()

    if _HALLUCINATION_RE.search(resp):
        return "hallucination"

    # Adversarial-dimension errors are always labelled adversarial_trap regardless
    # of the numeric diff, since the whole point is to track trap-specific failures.
    if result.dimension == CompositionDimension.ADVERSARIAL.value:
        return "adversarial_trap"

    # Skip numeric comparison for tuple answers (permutation/dihedral elements).
    # _to_num would parse "(2, 1, 3)" as 2 and "(1, 2, 3)" as 1, giving a
    # spurious diff=1 "off_by_one" classification for any permutation error.
    is_tuple_answer = result.ground_truth.strip().startswith("(")

    resp_num = None if is_tuple_answer else _to_num(resp)
    gt_num = None if is_tuple_answer else _to_num(result.ground_truth)

    if resp_num is not None and gt_num is not None:
        diff = abs(resp_num - gt_num)
        if diff == 1:
            return "off_by_one"
        if resp_num != 0 and abs(resp_num + gt_num) <= 1:
            return "inverse_confusion"

    if resp.lower() in ("0", "1", "e", "identity", "(0)", "(1)"):
        return "identity_confusion"

    return "wrong_value"


def error_taxonomy(report: EvalReport) -> Dict[str, Any]:
    """
    Classify every incorrect prediction into a mechanistic failure mode.

    Returns
    -------
    total_errors   Total number of wrong predictions
    categories     {category: {count, pct}} sorted by frequency
    by_depth       {depth: {category: count}}
    by_family      {family: {category: count}}
    dominant_error The most common failure mode
    """
    wrong = [r for r in report.results if not r.correct]
    if not wrong:
        return {
            "total_errors": 0, "categories": {}, "by_depth": {},
            "by_family": {}, "dominant_error": None,
        }

    counts: Dict[str, int] = defaultdict(int)
    by_depth: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_family: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in wrong:
        cat = _classify_error(r)
        counts[cat] += 1
        by_depth[r.depth][cat] += 1
        by_family[r.family][cat] += 1

    total = len(wrong)
    categories = {
        cat: {"count": cnt, "pct": round(cnt / total * 100, 1)}
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1])
    }

    return {
        "total_errors": total,
        "categories": categories,
        "by_depth": {d: dict(cats) for d, cats in sorted(by_depth.items())},
        "by_family": {f: dict(cats) for f, cats in sorted(by_family.items())},
        "dominant_error": max(counts, key=counts.__getitem__) if counts else None,
    }


# -- 2. Hallucination Onset ----------------------------------------------------

def hallucination_onset(report: EvalReport, threshold: float = 0.15) -> Dict[str, Any]:
    """
    Estimate the depth at which hallucination becomes prevalent.

    Hallucination is defined as a response matching _HALLUCINATION_RE.
    The onset depth is the first depth where the hallucination fraction
    (wrong-and-hallucinated / total-at-depth) exceeds *threshold*.

    Operates on chain families only, since conceptual and rule tasks at
    depth=1 have a different hallucination pattern than chain tasks.

    Returns onset_depth, per-depth curve, and a plain-language note.
    """
    by_depth: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "hall": 0})

    for r in _chain_results(report):
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


# -- 3. Accuracy by Depth ------------------------------------------------------

def accuracy_by_depth(report: EvalReport) -> Dict[str, Any]:
    """Per-depth accuracy for chain families, plus per-family per-depth breakdown.

    Returns a dict with two keys:

    curve
        List of {depth, accuracy, correct, total, errors_by_category} for each
        depth level in chain families (intra, inter, field), excluding adversarial
        and intermediate dimensions.

    by_family
        {family: [{depth, accuracy, correct, total}, ...]} — the same chain-family
        population broken out per family, useful for grouped bar charts.
    """
    chain = _chain_results(report)
    chain_stats = _depth_stats_from_results(chain)
    by_depth_errors = error_taxonomy(report).get("by_depth", {})

    curve = [
        {
            "depth": depth,
            "accuracy": round(d["accuracy"], 4),
            "correct": d["correct"],
            "total": d["total"],
            "errors_by_category": by_depth_errors.get(depth, {}),
        }
        for depth, d in sorted(chain_stats.items())
    ]

    by_family: Dict[str, List[Dict[str, Any]]] = {}
    for family in sorted(_CHAIN_FAMILIES):
        family_results = [
            r for r in report.results
            if r.family == family
            and r.dimension not in _CHAIN_EXCLUDED_DIMENSIONS
        ]
        if not family_results:
            continue
        depth_stats = _depth_stats_from_results(family_results)
        by_family[family] = [
            {
                "depth":    d,
                "accuracy": round(v["accuracy"], 4),
                "correct":  v["correct"],
                "total":    v["total"],
            }
            for d, v in sorted(depth_stats.items())
        ]

    return {"curve": curve, "by_family": by_family}


# -- 4. Accuracy by Family -----------------------------------------------------

def accuracy_by_family(report: EvalReport) -> Dict[str, Dict[str, Any]]:
    """Flat per-family accuracy across all task families.

    Unlike accuracy_by_depth (which covers chain families only), this function
    includes every family in the report: intra, inter, field, rule, conceptual,
    plus adversarial and intermediate tasks (which share the intra family label).

    Returns {family_label: {total, correct, accuracy}}.
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in report.results:
        stats[r.family]["total"] += 1
        if r.correct:
            stats[r.family]["correct"] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for fam, d in sorted(stats.items()):
        result[fam] = {
            "total":    d["total"],
            "correct":  d["correct"],
            "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
        }
    return result


# -- 5. Accuracy by Dimension --------------------------------------------------

def accuracy_by_dimension(report: EvalReport) -> Dict[str, Dict[str, Any]]:
    """Flat per-dimension accuracy across all compositional dimensions.

    Covers all seven dimensions: general, systematicity, substitutivity,
    productivity, overgeneralization, adversarial, intermediate_state.

    Returns {dimension_value: {total, correct, accuracy}}.
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in report.results:
        stats[r.dimension]["total"] += 1
        if r.correct:
            stats[r.dimension]["correct"] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for dim, d in sorted(stats.items()):
        result[dim] = {
            "total":    d["total"],
            "correct":  d["correct"],
            "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
        }
    return result


# -- 6. Complexity Analysis ----------------------------------------------------

def complexity_by_depth(report: EvalReport) -> List[Dict[str, Any]]:
    """Per-depth average of each algebraic complexity metric.

    Operates on chain-family results only (same population as fit_scaling_law).
    Results with no complexity data (e.g. loaded from JSON) are silently skipped.

    Each entry: depth, n, avg_algebraic_entropy, avg_commutativity_distance,
                avg_orbit_complexity, avg_structural_interference.
    """
    depth_totals: Dict[int, Dict] = defaultdict(lambda: {
        "n": 0, "h_alg": 0.0, "d_comm": 0.0, "o_c": 0.0, "i_s": 0.0,
    })

    for r in _chain_results(report):
        if r.complexity is None:
            continue
        d = depth_totals[r.depth]
        d["n"] += 1
        d["h_alg"]  += r.complexity.algebraic_entropy
        d["d_comm"] += r.complexity.commutativity_distance
        d["o_c"]    += r.complexity.orbit_complexity
        d["i_s"]    += r.complexity.structural_interference

    result: List[Dict[str, Any]] = []
    for depth in sorted(depth_totals.keys()):
        d = depth_totals[depth]
        n = d["n"]
        if n == 0:
            continue
        result.append({
            "depth": depth,
            "n": n,
            "avg_algebraic_entropy":       round(d["h_alg"]  / n, 4),
            "avg_commutativity_distance":  round(d["d_comm"] / n, 4),
            "avg_orbit_complexity":        round(d["o_c"]    / n, 4),
            "avg_structural_interference": round(d["i_s"]    / n, 4),
        })
    return result


def complexity_vs_accuracy(report: EvalReport) -> List[Dict[str, Any]]:
    """Return per-task complexity and outcome category for analysis plots.

    Each entry contains H_alg, D_comm, and O_c (the three primary metrics),
    plus the outcome category: "correct" or the mechanistic error label.

    I_s (Structural Interference) is omitted — it is zero for all non-inter-
    structure tasks and therefore uninformative for most datasets.

    Only includes tasks for which complexity data is available.
    """
    result: List[Dict[str, Any]] = []
    for r in report.results:
        if r.complexity is None:
            continue
        category = "correct" if r.correct else _classify_error(r)
        result.append({
            "task_id":  r.task_id,
            "family":   r.family,
            "depth":    r.depth,
            "H_alg":    r.complexity.algebraic_entropy,
            "O_c":      r.complexity.orbit_complexity,
            "D_comm":   r.complexity.commutativity_distance,
            "category": category,
            "correct":  r.correct,
        })
    return result


def complexity_analysis(report: EvalReport) -> Dict[str, Any]:
    """Combined complexity analysis: metrics by depth + per-task vs accuracy.

    Returns a dict with two keys:

    by_depth
        Per-depth averages of H_alg, D_comm, O_c (chain families only).

    vs_accuracy
        Per-task list of {task_id, family, depth, H_alg, O_c, D_comm,
        category, correct} — includes all families with complexity data.
    """
    return {
        "by_depth":    complexity_by_depth(report),
        "vs_accuracy": complexity_vs_accuracy(report),
    }


# -- Consolidated report -------------------------------------------------------

def run_analysis(report: EvalReport) -> Dict[str, Any]:
    """Run all analyses and return a single consolidated dict.

    Contains five structured analyses:

      accuracy_by_depth     Per-depth curve + per-family breakdown (chain families)
      accuracy_by_family    Flat per-family accuracy (all families)
      accuracy_by_dimension Flat per-dimension accuracy (all dimensions)
      complexity_analysis   Complexity metrics by depth + per-task vs accuracy
      hallucination_onset   Depth-resolved hallucination rate curve

    Scaling law, phase transition are available as standalone advanced functions
    (fit_scaling_law, fit_scaling_law_by_family, find_phase_transition) — they
    require many depth levels to be reliable.
    """
    return {
        "model":                 report.model_name,
        "task_set":              report.task_set_name,
        "overall_accuracy":      round(report.accuracy_overall, 4),
        "total_tasks":           report.total_tasks,
        "total_correct":         report.total_correct,
        "missing_predictions":   report.missing_predictions,
        "errored_predictions":   report.errored_predictions,
        # Five structured analyses
        "accuracy_by_depth":     accuracy_by_depth(report),
        "accuracy_by_family":    accuracy_by_family(report),
        "accuracy_by_dimension": accuracy_by_dimension(report),
        "complexity_analysis":   complexity_analysis(report),
        "hallucination_onset":   hallucination_onset(report),
    }


# -- Backward compatibility aliases -------------------------------------------

def stability_breakdown(report: EvalReport) -> List[Dict[str, Any]]:
    """Backward-compatible alias for accuracy_by_depth(report)['curve']."""
    return accuracy_by_depth(report)["curve"]


def accuracy_by_family_depth(report: EvalReport) -> Dict[str, List[Dict[str, Any]]]:
    """Backward-compatible alias for accuracy_by_depth(report)['by_family']."""
    return accuracy_by_depth(report)["by_family"]


# -- Console output ------------------------------------------------------------

def print_analysis(analysis: Dict[str, Any]) -> None:
    """Print a human-readable analysis summary to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ALGEBRAID Analysis Report")
    print(f"  Model    : {analysis['model']}")
    print(f"  Task Set : {analysis['task_set']}")
    print(f"  Overall  : {analysis['overall_accuracy']:.1%}")
    print(sep)

    # Error taxonomy
    tax = analysis.get("error_taxonomy", {})
    if tax.get("total_errors", 0) > 0:
        print(f"\n  Error Taxonomy ({tax['total_errors']} wrong predictions):")
        for cat, info in tax["categories"].items():
            bar = "#" * int(info["pct"] / 5)
            print(f"    {cat:<25} {info['count']:>4}  ({info['pct']:>5.1f}%)  {bar}")
        print(f"    Dominant: {tax['dominant_error']}")
    else:
        print(f"\n  Error Taxonomy: no errors (perfect accuracy).")

    # Accuracy by family
    by_fam = analysis.get("accuracy_by_family", {})
    if by_fam:
        print(f"\n  Accuracy by Family:")
        print(f"  {'Family':<32}  {'Acc':>6}  {'Correct':>7}  {'Total':>5}")
        print(f"  {'-'*55}")
        for fam, d in sorted(by_fam.items(), key=lambda kv: -kv[1]["accuracy"]):
            from .plots import _short_family
            print(f"  {_short_family(fam):<32}  {d['accuracy']:>6.1%}  "
                  f"{d['correct']:>7}  {d['total']:>5}")

    # Accuracy by dimension
    by_dim = analysis.get("accuracy_by_dimension", {})
    if by_dim:
        print(f"\n  Accuracy by Dimension:")
        print(f"  {'Dimension':<25}  {'Acc':>6}  {'Correct':>7}  {'Total':>5}")
        print(f"  {'-'*48}")
        for dim, d in sorted(by_dim.items(), key=lambda kv: -kv[1]["accuracy"]):
            print(f"  {dim:<25}  {d['accuracy']:>6.1%}  "
                  f"{d['correct']:>7}  {d['total']:>5}")

    # Depth curve
    depth_data = analysis.get("accuracy_by_depth", {})
    curve = depth_data.get("curve", [])
    if curve:
        print(f"\n  Accuracy by Depth (chain families):")
        print(f"  {'Depth':>5}  {'Acc':>6}  {'N':>5}  Top error")
        print(f"  {'-'*45}")
        for row in curve:
            top_err = (
                max(row["errors_by_category"], key=row["errors_by_category"].get)
                if row.get("errors_by_category") else "-"
            )
            print(f"  {row['depth']:>5}  {row['accuracy']:>6.1%}  "
                  f"{row['total']:>5}  {top_err}")

    # Hallucination onset
    hall = analysis.get("hallucination_onset", {})
    if hall:
        print(f"\n  Hallucination Onset: {hall.get('note', 'N/A')}")

    print(f"\n{sep}\n")
