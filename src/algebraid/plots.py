"""
Figure generation and PDF reporting for ALGEBRAID analysis results.

PNG figures (via matplotlib)
-----------------------------
Four core plots are saved to a ``figures/`` sub-directory:

    accuracy_vs_depth.png       Grouped bar chart: accuracy at each depth, one bar
                                per chain family.
    accuracy_by_family.png      Horizontal bar chart: overall accuracy per family.
    accuracy_by_dimension.png   Horizontal bar chart: accuracy per compositional
                                dimension.
    complexity_vs_accuracy.png  Accuracy vs H_alg, D_comm, O_c (3-panel).

Additional figure functions (stability_curve, complexity_profile,
hallucination_onset) are available for standalone use.

PDF report (via matplotlib PdfPages)
-------------------------------------
``generate_report_pdf(analysis, out_dir)`` produces a vector PDF with a
metrics summary page followed by all core figures.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; set before any pyplot import


# ── Colour palette ────────────────────────────────────────────────────────────

_ERROR_PALETTE: Dict[str, str] = {
    "adversarial_trap":   "#DC2626",
    "off_by_one":         "#F59E0B",
    "inverse_confusion":  "#9333EA",
    "identity_confusion": "#EA580C",
    "hallucination":      "#6B7280",
    "wrong_value":        "#2563EB",
}

_FAMILY_COLORS = ["#2563EB", "#16A34A", "#DC2626", "#9333EA", "#EA580C"]

_DIMENSION_COLORS: Dict[str, str] = {
    "general":            "#2563EB",
    "systematicity":      "#16A34A",
    "substitutivity":     "#9333EA",
    "productivity":       "#0891B2",
    "overgeneralization": "#D97706",
    "adversarial":        "#DC2626",
    "intermediate_state": "#EA580C",
}


def _short_family(name: str) -> str:
    """Shorten verbose family labels for legend / table readability."""
    return (
        name.replace("intra-structure composition", "intra")
            .replace("inter-structure composition", "inter")
            .replace("field arithmetic", "field")
    )


def _get_depth_by_family(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Read per-family per-depth data from new or legacy analysis dict."""
    # New structure: analysis["accuracy_by_depth"]["by_family"]
    depth_data = analysis.get("accuracy_by_depth")
    if isinstance(depth_data, dict) and "by_family" in depth_data:
        return depth_data["by_family"]
    # Legacy fallback
    return analysis.get("accuracy_by_family_depth", {})


def _get_depth_curve(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read per-depth stability curve from new or legacy analysis dict."""
    depth_data = analysis.get("accuracy_by_depth")
    if isinstance(depth_data, dict) and "curve" in depth_data:
        return depth_data["curve"]
    # Legacy fallback
    return analysis.get("stability_curve", [])


def _get_complexity_by_depth(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read complexity by depth from new or legacy analysis dict."""
    cx = analysis.get("complexity_analysis")
    if isinstance(cx, dict) and "by_depth" in cx:
        return cx["by_depth"]
    return analysis.get("complexity_by_depth", [])


def _get_complexity_vs_accuracy(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read complexity vs accuracy data from new or legacy analysis dict."""
    cx = analysis.get("complexity_analysis")
    if isinstance(cx, dict) and "vs_accuracy" in cx:
        return cx["vs_accuracy"]
    return analysis.get("complexity_vs_accuracy", [])


def _get_family_accuracy(analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Read flat per-family accuracy from new or legacy analysis dict."""
    # New structure: analysis["accuracy_by_family"] = {family: {total, correct, accuracy}}
    by_fam = analysis.get("accuracy_by_family")
    if isinstance(by_fam, dict) and by_fam:
        return by_fam
    # Legacy: aggregate from accuracy_by_family_depth
    by_fam_depth = analysis.get("accuracy_by_family_depth", {})
    if not by_fam_depth:
        return {}
    result = {}
    for fam, rows in by_fam_depth.items():
        tot = sum(r["total"] for r in rows)
        cor = sum(r["correct"] for r in rows)
        result[fam] = {
            "total": tot, "correct": cor,
            "accuracy": cor / tot if tot > 0 else 0.0,
        }
    return result


# ── Figure 1: accuracy vs depth (grouped by family) ──────────────────────────

def _accuracy_vs_depth(analysis: Dict[str, Any]) -> Optional[Any]:
    """Grouped bar chart: accuracy at each depth, one bar per chain family."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    by_fam_depth = _get_depth_by_family(analysis)
    if not by_fam_depth:
        curve = _get_depth_curve(analysis)
        if not curve:
            return None
        depths = [row["depth"] for row in curve]
        accs   = [row["accuracy"] for row in curve]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(depths, accs, "o-", color="#2563EB", linewidth=2, markersize=6)
        ax.axhline(0.5, color="#9CA3AF", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel("Composition depth", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(
            f"Accuracy vs. Composition Depth — {analysis.get('model', '')}",
            fontsize=12,
        )
        ax.set_ylim(-0.05, 1.10)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xticks(depths)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # Grouped bar chart — one bar per family per depth
    all_depths = sorted({row["depth"] for rows in by_fam_depth.values() for row in rows})
    families   = list(by_fam_depth.keys())
    n_fam      = len(families)
    bar_w      = 0.7 / max(n_fam, 1)
    x          = np.arange(len(all_depths))

    fig, ax = plt.subplots(figsize=(max(7, len(all_depths) * 1.2), 4.5))

    for i, (fam, color) in enumerate(zip(families, _FAMILY_COLORS)):
        depth_acc = {row["depth"]: row["accuracy"] for row in by_fam_depth[fam]}
        heights   = [depth_acc.get(d, 0.0) for d in all_depths]
        offset    = (i - n_fam / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, heights, width=bar_w * 0.9, color=color,
                      alpha=0.80, label=_short_family(fam))
        for bar, h in zip(bars, heights):
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7,
                        color="#374151")

    ax.axhline(0.5, color="#9CA3AF", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Composition depth", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        f"Accuracy by Depth — {analysis.get('model', '')}  ·  "
        f"{analysis.get('overall_accuracy', 0):.1%} overall",
        fontsize=12,
    )
    ax.set_ylim(-0.05, 1.20)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(all_depths)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Figure 2: accuracy by family ─────────────────────────────────────────────

def _accuracy_by_family(analysis: Dict[str, Any]) -> Optional[Any]:
    """Horizontal bar chart: overall accuracy per task family (all families)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    by_fam = _get_family_accuracy(analysis)
    if not by_fam:
        return None

    # Sort by accuracy descending for a clean ranked view
    families = sorted(by_fam, key=lambda f: by_fam[f]["accuracy"], reverse=True)
    accs     = [by_fam[f]["accuracy"] for f in families]
    ns       = [by_fam[f]["total"]    for f in families]
    labels   = [_short_family(f)      for f in families]
    colors   = [_FAMILY_COLORS[i % len(_FAMILY_COLORS)] for i in range(len(families))]

    fig, ax = plt.subplots(figsize=(7, max(3.0, len(families) * 0.75 + 1.0)))
    bars = ax.barh(labels[::-1], accs[::-1], color=colors[::-1],
                   edgecolor="white", height=0.6)

    for bar, acc, n in zip(bars, accs[::-1], ns[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}  (n={n})", va="center", ha="left", fontsize=9)

    ax.axvline(0.5, color="#9CA3AF", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy by Task Family", fontsize=12)
    ax.set_xlim(0, 1.38)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Figure 3: accuracy by dimension ──────────────────────────────────────────

def _accuracy_by_dimension(analysis: Dict[str, Any]) -> Optional[Any]:
    """Horizontal bar chart: accuracy per compositional dimension."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    by_dim = analysis.get("accuracy_by_dimension", {})
    if not by_dim:
        return None

    # Sort by accuracy descending
    dims   = sorted(by_dim, key=lambda d: by_dim[d]["accuracy"], reverse=True)
    accs   = [by_dim[d]["accuracy"] for d in dims]
    ns     = [by_dim[d]["total"]    for d in dims]
    colors = [_DIMENSION_COLORS.get(d, "#2563EB") for d in dims]

    fig, ax = plt.subplots(figsize=(7, max(3.0, len(dims) * 0.75 + 1.0)))
    bars = ax.barh(dims[::-1], accs[::-1], color=colors[::-1],
                   edgecolor="white", height=0.6)

    for bar, acc, n in zip(bars, accs[::-1], ns[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1%}  (n={n})", va="center", ha="left", fontsize=9)

    ax.axvline(0.5, color="#9CA3AF", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy by Compositional Dimension", fontsize=12)
    ax.set_xlim(0, 1.38)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Figure 4: two-panel stability curve ──────────────────────────────────────

def _stability_curve(analysis: Dict[str, Any]) -> Optional[Any]:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    curve = _get_depth_curve(analysis)
    if not curve:
        return None

    depths = [row["depth"]    for row in curve]
    accs   = [row["accuracy"] for row in curve]
    all_cats = sorted({
        cat for row in curve for cat in row.get("errors_by_category", {})
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot(depths, accs, "o-", color="#2563EB", linewidth=2,
             markersize=6, label="Accuracy")
    ax1.axhline(0.5, color="#9CA3AF", linewidth=1, linestyle="--", alpha=0.6)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.set_ylim(-0.05, 1.10)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Stability Breakdown  —  {analysis.get('model', '')}", fontsize=12)

    has_errors = any(row.get("errors_by_category") for row in curve)
    if all_cats and has_errors:
        bottoms = [0.0] * len(depths)
        for cat in all_cats:
            heights = [row.get("errors_by_category", {}).get(cat, 0) for row in curve]
            ax2.bar(depths, heights, bottom=bottoms,
                    color=_ERROR_PALETTE.get(cat, "#2563EB"),
                    label=cat, edgecolor="white", width=0.6)
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        ax2.set_ylabel("Error count", fontsize=10)
        ax2.legend(fontsize=7, loc="upper left")
        ax2.grid(True, axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No per-depth error data available",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=9, color="#6B7280")

    ax2.set_xlabel("Composition depth", fontsize=10)
    ax2.set_xticks(depths)
    fig.tight_layout()
    return fig


# ── Figure 6: complexity metrics by depth ────────────────────────────────────

_METRIC_LABELS: List[tuple] = [
    ("avg_algebraic_entropy",      "H_alg  (Algebraic Entropy)",       "#2563EB"),
    ("avg_commutativity_distance", "D_comm (Commutativity Distance)",  "#16A34A"),
    ("avg_orbit_complexity",       "O_c    (Orbit Complexity)",        "#DC2626"),
]
# I_s (Structural Interference) is intentionally omitted: it is zero for all
# non-inter-structure tasks and therefore uninformative for most datasets.


def _complexity_profile(analysis: Dict[str, Any]) -> Optional[Any]:
    import matplotlib.pyplot as plt

    cx_data = _get_complexity_by_depth(analysis)
    if not cx_data:
        return None

    depths = [row["depth"] for row in cx_data]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True)

    for ax, (key, label, color) in zip(axes, _METRIC_LABELS):
        values = [row.get(key, 0.0) for row in cx_data]
        ax.bar(depths, values, color=color, alpha=0.70, edgecolor="white", width=0.6)
        ax.plot(depths, values, "o-", color=color, linewidth=1.5, markersize=5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Avg value", fontsize=9)
        ax.set_xlabel("Composition depth", fontsize=9)
        ax.set_xticks(depths)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Algebraic Complexity Metrics by Depth  —  {analysis.get('model', '')}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ── Figure 7: complexity vs accuracy (stacked error breakdown) ────────────────

_N_BINS = 6

# Ordered category palette: "correct" first, then error types
_CAT_ORDER = [
    "correct",
    "adversarial_trap",
    "off_by_one",
    "inverse_confusion",
    "identity_confusion",
    "hallucination",
    "wrong_value",
]
_CAT_COLORS_STACKED = {
    "correct":            "#16A34A",
    "adversarial_trap":   "#DC2626",
    "off_by_one":         "#F59E0B",
    "inverse_confusion":  "#9333EA",
    "identity_confusion": "#EA580C",
    "hallucination":      "#6B7280",
    "wrong_value":        "#2563EB",
}
_CAT_LABELS = {
    "correct":            "Correct",
    "adversarial_trap":   "Adversarial trap",
    "off_by_one":         "Off-by-one",
    "inverse_confusion":  "Inverse confusion",
    "identity_confusion": "Identity confusion",
    "hallucination":      "Hallucination",
    "wrong_value":        "Wrong value",
}


def _ols_trend(xs, ys):
    """Return (slope, intercept) for a simple OLS fit, or (None, None) if degenerate."""
    n = len(xs)
    if n < 2:
        return None, None
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    if ss_xx == 0:
        return None, None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ss_xx
    return slope, my - slope * mx


def _cx_accuracy_panel(ax, data, key, title):
    """Draw one complexity metric panel: single accuracy line across complexity bins."""
    import matplotlib.ticker as mticker

    pairs = [(d[key], d["correct"]) for d in data if key in d]
    if not pairs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#6B7280")
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    values = [p[0] for p in pairs]
    val_min, val_max = min(values), max(values)

    bw = (val_max - val_min) / _N_BINS if val_min != val_max else 1.0
    bins: List[List[bool]] = [[] for _ in range(_N_BINS)]
    for v, correct in pairs:
        idx = min(int((v - val_min) / bw), _N_BINS - 1) if val_min != val_max else 0
        bins[idx].append(correct)

    bin_mids = [val_min + (i + 0.5) * bw for i in range(_N_BINS)]
    non_empty = [(m, b) for m, b in zip(bin_mids, bins) if b]
    xs = [m for m, _ in non_empty]
    bns = [b for _, b in non_empty]
    ns  = [len(b) for b in bns]
    ys  = [sum(b) / len(b) for b in bns]

    ax.plot(xs, ys, "o-", color="#16A34A", linewidth=2, markersize=6)

    for x, n, y in zip(xs, ns, ys):
        ax.text(x, y + 0.04, f"n={n}", ha="center", va="bottom",
                fontsize=7, color="#374151")

    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Complexity value", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _complexity_vs_accuracy(analysis: Dict[str, Any]) -> Optional[Any]:
    """3-panel accuracy line chart: H_alg, D_comm, O_c vs accuracy."""
    import matplotlib.pyplot as plt

    data = _get_complexity_vs_accuracy(analysis)
    if not data:
        return None

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    _cx_accuracy_panel(ax1, data, "H_alg",  r"Accuracy vs. $H_\mathrm{alg}$ (Algebraic Entropy)")
    _cx_accuracy_panel(ax2, data, "D_comm", r"Accuracy vs. $D_\mathrm{comm}$ (Commutativity Distance)")
    _cx_accuracy_panel(ax3, data, "O_c",    r"Accuracy vs. $O_c$ (Orbit Complexity)")

    fig.suptitle(
        f"Complexity vs. Accuracy  —  {analysis.get('model', '')}",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ── Figure 8: hallucination onset ────────────────────────────────────────────

def _hallucination_onset_chart(analysis: Dict[str, Any]) -> Optional[Any]:
    """Line chart: hallucination rate by depth with threshold and onset marked."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    onset_data = analysis.get("hallucination_onset", {})
    curve = onset_data.get("curve", [])
    if not curve:
        return None

    depths    = [r["depth"]             for r in curve]
    rates     = [r["hallucination_rate"] for r in curve]
    threshold = onset_data.get("threshold", 0.15)
    onset_depth = onset_data.get("onset_depth")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(depths, rates, "o-", color="#6B7280", linewidth=2,
            markersize=7, label="Hallucination rate", zorder=3)

    # Threshold line
    ax.axhline(threshold, color="#DC2626", linewidth=1.5, linestyle="--",
               alpha=0.8, label=f"Threshold ({threshold:.0%})")

    # Onset marker
    if onset_depth is not None:
        ax.axvline(onset_depth, color="#DC2626", linewidth=1.5, linestyle=":",
                   alpha=0.6)
        onset_rate = next(
            (r["hallucination_rate"] for r in curve if r["depth"] == onset_depth), None
        )
        if onset_rate is not None:
            ax.annotate(
                f"Onset depth={onset_depth}\n({onset_rate:.1%})",
                xy=(onset_depth, onset_rate),
                xytext=(onset_depth + 0.3, onset_rate + 0.05),
                fontsize=8, color="#DC2626",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.0),
            )

    # Annotate each point
    for depth, rate in zip(depths, rates):
        ax.text(depth, rate + 0.012, f"{rate:.1%}", ha="center",
                va="bottom", fontsize=7.5, color="#374151")

    ax.set_xlabel("Composition depth", fontsize=11)
    ax.set_ylabel("Hallucination rate", fontsize=11)
    ax.set_title(
        f"Hallucination Onset by Depth  —  {analysis.get('model', '')}",
        fontsize=12,
    )
    ax.set_ylim(-0.02, min(1.0, max(rates + [threshold]) * 1.4 + 0.05))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xticks(depths)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Figure registry ──────────────────────────────────────────────────────────

_FIGURE_REGISTRY = [
    ("accuracy_vs_depth.png",      _accuracy_vs_depth),
    ("accuracy_by_family.png",     _accuracy_by_family),
    ("accuracy_by_dimension.png",  _accuracy_by_dimension),
    ("complexity_vs_accuracy.png", _complexity_vs_accuracy),
]


# ── Metrics summary page ────────────────────────────────────────────────────

def _metrics_summary(analysis: Dict[str, Any]) -> Optional[Any]:
    """Single-page text figure with model metadata and accuracy tables."""
    import matplotlib.pyplot as plt

    model   = analysis.get("model", "unknown")
    task_set = analysis.get("task_set", "unknown")
    overall = analysis.get("overall_accuracy", 0.0)
    total   = analysis.get("total_tasks", 0)
    correct = analysis.get("total_correct", 0)

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    y = 0.94
    ax.text(0.5, y, "ALGEBRAID: Model Evaluation Report",
            ha="center", fontsize=18, fontweight="bold", color="#1E3A5F")
    y -= 0.04
    ax.text(0.5, y, "Compositional Algebraic Reasoning Benchmark",
            ha="center", fontsize=10, fontstyle="italic", color="#6E6E6E")

    y -= 0.06
    meta_lines = [
        f"Model:     {model}",
        f"Task set:  {task_set}",
        f"Evaluated: {correct}/{total}  ({overall:.1%} accuracy)",
    ]
    for line in meta_lines:
        ax.text(0.08, y, line, fontsize=11, fontfamily="monospace")
        y -= 0.03

    # Per-family table
    by_fam = _get_family_accuracy(analysis)
    if by_fam:
        y -= 0.03
        ax.text(0.08, y, "Accuracy by Task Family", fontsize=13,
                fontweight="bold")
        y -= 0.01
        ax.axhline(y=y, xmin=0.06, xmax=0.94, color="#333", linewidth=0.5)
        y -= 0.025
        ax.text(0.08, y, f"{'Family':<30s} {'n':>6s} {'Correct':>8s} {'Acc':>8s}",
                fontsize=10, fontfamily="monospace", fontweight="bold")
        y -= 0.025
        for fam in sorted(by_fam, key=lambda f: by_fam[f]["accuracy"],
                          reverse=True):
            d = by_fam[fam]
            label = _short_family(fam)
            ax.text(0.08, y,
                    f"{label:<30s} {d['total']:>6d} {d['correct']:>8d} "
                    f"{d['accuracy']:>7.1%}",
                    fontsize=10, fontfamily="monospace")
            y -= 0.025

    # Per-dimension table
    by_dim = analysis.get("accuracy_by_dimension", {})
    if by_dim:
        y -= 0.03
        ax.text(0.08, y, "Accuracy by Compositional Dimension", fontsize=13,
                fontweight="bold")
        y -= 0.01
        ax.axhline(y=y, xmin=0.06, xmax=0.94, color="#333", linewidth=0.5)
        y -= 0.025
        ax.text(0.08, y, f"{'Dimension':<30s} {'n':>6s} {'Correct':>8s} {'Acc':>8s}",
                fontsize=10, fontfamily="monospace", fontweight="bold")
        y -= 0.025
        for dim in sorted(by_dim, key=lambda d: by_dim[d]["accuracy"],
                          reverse=True):
            d = by_dim[dim]
            ax.text(0.08, y,
                    f"{dim:<30s} {d['total']:>6d} {d['correct']:>8d} "
                    f"{d['accuracy']:>7.1%}",
                    fontsize=10, fontfamily="monospace")
            y -= 0.025

    fig.tight_layout()
    return fig


# ── PNG figure generation entry point ────────────────────────────────────────

def generate_figures(analysis: Dict[str, Any], out_dir: str) -> List[str]:
    """Generate all analysis figures and save as PNGs.

    Returns list of saved PNG paths.
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    for filename, func in _FIGURE_REGISTRY:
        try:
            fig = func(analysis)
            if fig is not None:
                path = os.path.join(out_dir, filename)
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
        except Exception as exc:
            name = func.__name__.lstrip("_")
            print(f"  WARNING: figure '{name}' could not be generated: {exc}")

    return saved


# ── Vector PDF report (matplotlib PdfPages) ──────────────────────────────────

def generate_report_pdf(
    analysis: Dict[str, Any],
    out_dir: str,
) -> Optional[str]:
    """Generate a vector PDF report with metrics summary + all figures.

    Parameters
    ----------
    analysis:
        Consolidated analysis dict from ``run_analysis()``.
    out_dir:
        Directory where ``report.pdf`` will be saved.

    Returns
    -------
    str or None
        Path to saved PDF, or None on failure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.pdf")

    try:
        with PdfPages(out_path) as pdf:
            # Page 1: metrics summary
            summary_fig = _metrics_summary(analysis)
            if summary_fig is not None:
                pdf.savefig(summary_fig, bbox_inches="tight")
                plt.close(summary_fig)

            # Remaining pages: one per figure
            for _filename, func in _FIGURE_REGISTRY:
                try:
                    fig = func(analysis)
                    if fig is not None:
                        pdf.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                except Exception:
                    pass
    except Exception as exc:
        print(f"  WARNING: PDF generation failed: {exc}")
        return None

    return out_path
