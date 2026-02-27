"""
Figure generation and PDF reporting for ALGEBRAID analysis results.

PNG figures (via matplotlib)
-----------------------------
Seven plots are saved to a ``figures/`` sub-directory inside the run folder:

    accuracy_vs_depth.png       Grouped bar chart: accuracy at each depth, one bar
                                per chain family.
    accuracy_by_family.png      Horizontal bar chart: overall accuracy per family
                                (all task families).
    accuracy_by_dimension.png   Horizontal bar chart: accuracy per compositional
                                dimension.
    stability_curve.png         Two-panel: accuracy + stacked error counts per depth.
    complexity_profile.png      1×3 grid: H_alg · D_comm · O_c by depth.
    complexity_vs_accuracy.png  Stacked outcome bars (correct + error types) vs
                                H_alg and O_c, with accuracy trend line.
    hallucination_onset.png     Line chart: hallucination rate by depth with onset
                                threshold marked.

PDF report (via fpdf2)
-----------------------
``generate_report_pdf(analysis, out_dir, png_paths)`` generates a clean
academic-style PDF using *fpdf2* (pure Python, no external binaries):

    Cover   Model metadata and overall accuracy summary
    § 1     Accuracy by composition depth — table + figure
    § 2     Accuracy by task family — table + figure
    § 3     Accuracy by compositional dimension — table + figure
    § 4     Algebraic complexity vs. accuracy — metric table + figure

Matplotlib is required for figures. *fpdf2* must be installed
(``pip install fpdf2``) for PDF generation.
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

def _accuracy_vs_depth(analysis: Dict[str, Any], out_dir: str) -> Optional[str]:
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
        path = os.path.join(out_dir, "accuracy_vs_depth.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

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

    path = os.path.join(out_dir, "accuracy_vs_depth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 2: accuracy by family ─────────────────────────────────────────────

def _accuracy_by_family(analysis: Dict[str, Any], out_dir: str) -> Optional[str]:
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

    path = os.path.join(out_dir, "accuracy_by_family.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 3: accuracy by dimension ──────────────────────────────────────────

def _accuracy_by_dimension(analysis: Dict[str, Any], out_dir: str) -> Optional[str]:
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

    path = os.path.join(out_dir, "accuracy_by_dimension.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 4: two-panel stability curve ──────────────────────────────────────

def _stability_curve(analysis: Dict[str, Any], out_dir: str) -> Optional[str]:
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

    path = os.path.join(out_dir, "stability_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 6: complexity metrics by depth ────────────────────────────────────

_METRIC_LABELS: List[tuple] = [
    ("avg_algebraic_entropy",      "H_alg  (Algebraic Entropy)",       "#2563EB"),
    ("avg_commutativity_distance", "D_comm (Commutativity Distance)",  "#16A34A"),
    ("avg_orbit_complexity",       "O_c    (Orbit Complexity)",        "#DC2626"),
]
# I_s (Structural Interference) is intentionally omitted: it is zero for all
# non-inter-structure tasks and therefore uninformative for most datasets.


def _complexity_profile(analysis: Dict[str, Any], out_dir: str) -> Optional[str]:
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

    path = os.path.join(out_dir, "complexity_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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


def _complexity_vs_accuracy(analysis: Dict[str, Any], out_dir: str) -> Optional[str]:
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

    path = os.path.join(out_dir, "complexity_vs_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Figure 8: hallucination onset ────────────────────────────────────────────

def _hallucination_onset_chart(analysis: Dict[str, Any], out_dir: str) -> Optional[str]:
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

    path = os.path.join(out_dir, "hallucination_onset.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── PNG figure generation entry point ────────────────────────────────────────

def generate_figures(analysis: Dict[str, Any], out_dir: str) -> List[str]:
    """Generate all analysis figures and save as PNGs.

    Parameters
    ----------
    analysis:
        Consolidated analysis dict returned by ``run_analysis()``.
    out_dir:
        Directory where PNGs will be saved (created if absent).

    Returns
    -------
    List[str]
        Paths of successfully saved PNG files (empty list if no data).
    """
    os.makedirs(out_dir, exist_ok=True)

    generators = [
        _accuracy_vs_depth,
        _accuracy_by_family,
        _accuracy_by_dimension,
        _complexity_vs_accuracy,
    ]

    saved: List[str] = []
    for gen in generators:
        try:
            path = gen(analysis, out_dir)
            if path:
                saved.append(path)
        except Exception as exc:
            name = gen.__name__.lstrip("_")
            print(f"  WARNING: figure '{name}' could not be generated: {exc}")

    return saved


# ── fpdf2 PDF report ──────────────────────────────────────────────────────────

_C_NAVY = (30, 58, 95)    # title accent
_C_DARK = (20, 20, 20)    # body text
_C_GRAY = (110, 110, 110) # captions / secondary text


def _fmt_pct(v: float) -> str:
    """Format *v* (0.0–1.0) as a percentage string, e.g. ``'73.2%'``."""
    return f"{v * 100:.1f}%"


def _build_report_pdf(analysis: Dict[str, Any], png_paths: List[str]) -> bytes:
    """Build the evaluation report and return raw PDF bytes."""
    from fpdf import FPDF, FontFace, XPos, YPos
    from fpdf.enums import TableCellFillMode, TableBordersLayout
    import datetime

    pngs = {os.path.basename(p): p for p in png_paths if os.path.exists(p)}

    model         = analysis.get("model",    "unknown")
    task_set      = analysis.get("task_set", "unknown")
    overall       = analysis.get("overall_accuracy", 0.0)
    curve         = _get_depth_curve(analysis)
    by_fam        = _get_family_accuracy(analysis)
    by_dim        = analysis.get("accuracy_by_dimension", {})

    total_tasks   = analysis.get(
        "total_tasks", sum(r["total"]   for r in curve) if curve else 0
    )
    total_correct = analysis.get(
        "total_correct", sum(r["correct"] for r in curve) if curve else 0
    )
    errored_preds = analysis.get("errored_predictions", 0) or 0
    total_bench   = total_tasks + errored_preds

    if curve:
        depths      = [r["depth"] for r in curve]
        depth_range = (f"{min(depths)}-{max(depths)}"
                       if len(depths) > 1 else str(depths[0]))
        n_depths    = len(depths)
    else:
        depth_range = "--"
        n_depths    = 0

    date_str   = datetime.date.today().isoformat()
    n_families = len(by_fam)

    # ── PDF subclass with footer ───────────────────────────────────────────────
    class _PDF(FPDF):
        def footer(self) -> None:
            self.set_y(-12)
            self.set_font("Times", "", 8)
            self.set_text_color(*_C_GRAY)
            self.cell(0, 6, f"ALGEBRAID  |  {model}", align="L",
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.cell(0, 6, str(self.page_no()), align="R",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(*_C_DARK)

    pdf = _PDF(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=20, top=20, right=20)
    lw   = pdf.w - 40    # usable text width (~175.9 mm)
    fig_w = lw * 0.88    # figures slightly narrower than full text width

    # ── Helpers ────────────────────────────────────────────────────────────────

    def rule(thick: bool = False) -> None:
        pdf.set_draw_color(*_C_DARK)
        pdf.set_line_width(0.5 if thick else 0.2)
        y = pdf.get_y()
        pdf.line(20, y, 20 + lw, y)
        pdf.set_line_width(0.2)

    def sec_title(n: int, title: str) -> None:
        pdf.ln(5)
        pdf.set_font("Times", "B", 11)
        pdf.set_text_color(*_C_DARK)
        pdf.cell(0, 6, f"{n}.  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        rule(thick=False)
        pdf.ln(3)

    def body_text(text: str) -> None:
        pdf.set_font("Times", "", 9)
        pdf.set_text_color(*_C_DARK)
        pdf.multi_cell(0, 5, text, align="J")
        pdf.ln(3)

    def academic_table(
        col_labels: List[str],
        rows: List[List[str]],
        col_widths: List[float],
        right_from: int = 1,
    ) -> None:
        """Booktabs-style table: toprule/midrule/bottomrule, no fill."""
        LINE_H = 5.0
        PAD    = 1.0
        HDR_H  = LINE_H + 2 * PAD   # approx header row height
        tbl_w  = sum(col_widths)
        x0     = 20

        headings_style = FontFace(emphasis="BOLD")
        pdf.set_font("Times", "", 9)
        pdf.set_text_color(*_C_DARK)

        y_top = pdf.get_y()

        with pdf.table(
            col_widths=tuple(col_widths),
            headings_style=headings_style,
            cell_fill_mode=TableCellFillMode.NONE,
            borders_layout=TableBordersLayout.NONE,
            line_height=LINE_H,
            padding=PAD,
            align="LEFT",
            gutter_height=0,
        ) as table:
            hdr = table.row()
            for lbl in col_labels:
                hdr.cell(lbl)
            for data_row in rows:
                row = table.row()
                for i, cell_val in enumerate(data_row):
                    align = "R" if i >= right_from else "L"
                    row.cell(str(cell_val), align=align)

        y_bot = pdf.get_y()

        # Overlay booktabs rules (drawn after table so they sit on top)
        pdf.set_draw_color(*_C_DARK)
        pdf.set_line_width(0.5)
        pdf.line(x0, y_top, x0 + tbl_w, y_top)          # toprule (thick)
        pdf.set_line_width(0.2)
        pdf.line(x0, y_top + HDR_H, x0 + tbl_w, y_top + HDR_H)  # midrule
        pdf.set_line_width(0.5)
        pdf.line(x0, y_bot, x0 + tbl_w, y_bot)           # bottomrule (thick)
        pdf.set_line_width(0.2)
        pdf.ln(4)

    def embed_figure(name: str, caption: str) -> None:
        path = pngs.get(name)
        if not path:
            return
        pdf.image(path, w=fig_w)
        pdf.set_font("Times", "I", 8)
        pdf.set_text_color(*_C_GRAY)
        pdf.multi_cell(0, 4.5, caption)
        pdf.set_text_color(*_C_DARK)
        pdf.ln(2)

    # ── Title header (page 1) ──────────────────────────────────────────────────
    pdf.add_page()

    # Brand + report title
    pdf.set_font("Times", "B", 20)
    pdf.set_text_color(*_C_NAVY)
    pdf.cell(0, 11, "ALGEBRAID: Model Evaluation Report",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Times", "I", 10)
    pdf.set_text_color(*_C_GRAY)
    pdf.cell(0, 5, "Compositional Algebraic Reasoning Benchmark",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    rule(thick=True)
    pdf.ln(3)

    # Metadata: two columns
    hw = lw / 2
    for k1, v1, k2, v2 in [
        ("Model:",       model,       "Task Set:",    task_set),
        ("Date:",        date_str,    "Depth Range:", depth_range),
        ("Evaluated:",   str(total_tasks),  "Correct:",     str(total_correct)),
        ("Accuracy:",    _fmt_pct(overall), "Errors:",      str(errored_preds)),
    ]:
        pdf.set_font("Times", "B", 9)
        pdf.set_text_color(*_C_DARK)
        pdf.cell(hw * 0.38, 5.5, k1, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Times", "", 9)
        pdf.cell(hw * 0.62, 5.5, str(v1), new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Times", "B", 9)
        pdf.cell(hw * 0.38, 5.5, k2, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Times", "", 9)
        pdf.cell(hw * 0.62, 5.5, str(v2), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    rule(thick=True)
    pdf.ln(3)

    # Abstract
    pdf.set_font("Times", "B", 9)
    pdf.set_text_color(*_C_DARK)
    pdf.cell(0, 5, "Abstract", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    abstract = (
        f"  This report evaluates {model} on ALGEBRAID -- a procedurally generated "
        f"suite of compositional algebraic reasoning tasks. The model answered "
        f"{total_correct} of {total_tasks} evaluated tasks correctly "
        f"({_fmt_pct(overall)} accuracy)"
    )
    if n_depths > 0:
        abstract += f", spanning {n_depths} composition depth{'s' if n_depths != 1 else ''}"
    if n_families > 0:
        abstract += f" across {n_families} task {'families' if n_families != 1 else 'family'}"
    abstract += "."
    if errored_preds > 0:
        abstract += (
            f" {errored_preds} response{'s' if errored_preds != 1 else ''} "
            "excluded due to parsing errors."
        )
    pdf.set_font("Times", "", 9)
    pdf.multi_cell(0, 5, abstract, align="J")

    # ── Section 1: Accuracy by Composition Depth ──────────────────────────────
    sec_title(1, "Accuracy by Composition Depth")
    body_text(
        "Intra-structure, inter-structure, and field arithmetic tasks only. "
        "Adversarial and intermediate-state tasks are excluded -- their depth "
        "parameter does not index monotone difficulty."
    )
    if curve:
        depth_rows = [
            [str(r["depth"]), str(r["correct"]), str(r["total"]),
             _fmt_pct(r["accuracy"])]
            for r in curve
        ]
        academic_table(
            ["Depth", "Correct", "Total", "Accuracy"],
            depth_rows, [28, 32, 32, 32], right_from=1,
        )
    else:
        body_text("No chain-family depth data available.")
    embed_figure(
        "accuracy_vs_depth.png",
        "Figure 1. Accuracy by composition depth, grouped by task family.",
    )

    # ── Section 2: Accuracy by Task Family ────────────────────────────────────
    sec_title(2, "Accuracy by Task Family")
    body_text(
        "All seven generator families. Adversarial and intermediate-state tasks "
        "are attributed to the intra-structure family for this breakdown."
    )
    _FAM_DESC: Dict[str, str] = {
        "intra-structure composition":
            "Operations chained within one algebraic structure.",
        "inter-structure composition":
            "Component-wise operations across a direct product G x H.",
        "field arithmetic":
            "Expression evaluation in a finite field GF(p).",
        "rule induction":
            "Infer the pattern from examples; give the next element.",
        "conceptual query":
            "Structural property query (identity, order, commutativity, etc.).",
    }
    if by_fam:
        fam_rows = [
            [
                _short_family(f),
                _FAM_DESC.get(f, ""),
                str(d["total"]),
                str(d["correct"]),
                _fmt_pct(d["accuracy"]),
            ]
            for f, d in sorted(
                by_fam.items(), key=lambda kv: kv[1]["accuracy"], reverse=True
            )
        ]
        academic_table(
            ["Family", "Description", "n", "Correct", "Acc."],
            fam_rows, [16, 90, 14, 22, 24], right_from=2,
        )
    else:
        body_text("No per-family data available.")
    embed_figure(
        "accuracy_by_family.png",
        "Figure 2. Overall accuracy per task family, ranked descending.",
    )

    # ── Section 3: Accuracy by Compositional Dimension ────────────────────────
    sec_title(3, "Accuracy by Compositional Dimension")
    body_text(
        "The first four dimensions follow Hupkes et al. (2020); the remaining "
        "three are ALGEBRAID-specific extensions."
    )
    _DIM_DESC: Dict[str, str] = {
        "general":
            "Standard task; no specific compositional stress.",
        "systematicity":
            "Unseen combination of known operations and structures.",
        "substitutivity":
            "Synonym-substituted prompt; answer must be invariant.",
        "productivity":
            "Chain length exceeds the training distribution.",
        "overgeneralization":
            "Rule applied to a context where it should not hold.",
        "adversarial":
            "Chain designed to trigger a reasoning shortcut.",
        "intermediate_state":
            "Value at step k of an N-step chain (k < N).",
    }
    if by_dim:
        dim_rows = [
            [
                dim,
                _DIM_DESC.get(dim, ""),
                str(d["total"]),
                str(d["correct"]),
                _fmt_pct(d["accuracy"]),
            ]
            for dim, d in sorted(
                by_dim.items(), key=lambda kv: kv[1]["accuracy"], reverse=True
            )
        ]
        academic_table(
            ["Dimension", "Description", "n", "Correct", "Acc."],
            dim_rows, [36, 70, 14, 22, 24], right_from=2,
        )
    else:
        body_text("No dimension data available.")
    embed_figure(
        "accuracy_by_dimension.png",
        "Figure 3. Accuracy per compositional dimension, ranked descending.",
    )

    # ── Section 4: Algebraic Complexity vs. Accuracy ──────────────────────────
    sec_title(4, "Algebraic Complexity vs. Accuracy")
    body_text(
        "Three task-intrinsic complexity metrics -- H_alg, D_comm, O_c -- are "
        "plotted against accuracy. Each panel bins tasks by metric value and "
        "shows accuracy per bin."
    )
    cx_rows = [
        ["H_alg",  "Algebraic Entropy",
         "log2(|G|) x depth -- group size and chain length combined."],
        ["D_comm", "Commutativity Distance",
         "Fraction of op-pairs where operand order changes the result."],
        ["O_c",    "Orbit Complexity",
         "Distinct intermediate values / |G| -- element traversal breadth."],
    ]
    academic_table(
        ["Symbol", "Metric", "Definition"],
        cx_rows, [18, 46, 102], right_from=3,
    )
    embed_figure(
        "complexity_vs_accuracy.png",
        "Figure 4. Accuracy vs. H_alg (left), D_comm (centre), O_c (right).",
    )

    return bytes(pdf.output())


def generate_report_pdf(
    analysis: Dict[str, Any],
    out_dir: str,
    png_paths: List[str],
) -> Optional[str]:
    """Generate a PDF report from analysis data and PNG figures using fpdf2.

    Parameters
    ----------
    analysis:
        Consolidated analysis dict returned by ``run_analysis()``.
    out_dir:
        Directory where ``report.pdf`` will be saved (created if absent).
    png_paths:
        Ordered list of PNG file paths from ``generate_figures()``.

    Returns
    -------
    str or None
        Absolute path to the saved PDF, or None on failure.
    """
    try:
        from fpdf import FPDF  # noqa: F401
    except ImportError:
        print(
            "  WARNING: fpdf2 is not installed. "
            "Run:  pip install fpdf2"
        )
        return None

    os.makedirs(out_dir, exist_ok=True)

    try:
        pdf_bytes = _build_report_pdf(analysis, png_paths)
    except Exception as exc:
        print(f"  WARNING: PDF generation failed: {exc}")
        return None

    out_path = os.path.join(out_dir, "report.pdf")
    try:
        with open(out_path, "wb") as fh:
            fh.write(pdf_bytes)
    except Exception as exc:
        print(f"  WARNING: Could not write report.pdf: {exc}")
        return None

    return out_path
