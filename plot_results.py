"""Generate charts from the COMPEVAL evaluation results."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

with open("results/eval_report.json") as f:
    report = json.load(f)

MODEL = report["model_name"]
OVERALL_ACC = report["accuracy_overall"]
TOTAL = report["total_tasks"]
CORRECT = report["total_correct"]
CEILING_50 = report["compositional_ceiling_50"]
CEILING_25 = report["compositional_ceiling_25"]
COMPLEXITY = report["algebraic_complexity"]

# ── Colour palette ──────────────────────────────────────────────────────────
BLUE   = "#2563EB"
ORANGE = "#F59E0B"
GREEN  = "#10B981"
RED    = "#EF4444"
PURPLE = "#8B5CF6"
GREY   = "#6B7280"
BG     = "#F8FAFC"

fig = plt.figure(figsize=(16, 12), facecolor=BG)
fig.suptitle(
    f"COMPEVAL Evaluation — {MODEL}\n"
    f"Overall Accuracy: {OVERALL_ACC:.1%}  ({CORRECT}/{TOTAL} tasks)",
    fontsize=16, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 1. Depth curve ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
depth_data = report["accuracy_by_depth"]
depths = sorted(int(k) for k in depth_data.keys())
accs   = [depth_data[str(d)]["accuracy"] for d in depths]
totals = [depth_data[str(d)]["total"]    for d in depths]

bars = ax1.bar(depths, [a * 100 for a in accs], color=BLUE, alpha=0.85, zorder=3)
ax1.axhline(50, color=ORANGE, linestyle="--", linewidth=1.5, label="50% threshold")
ax1.axhline(25, color=RED,    linestyle=":",  linewidth=1.5, label="25% threshold")
for bar, acc, tot in zip(bars, accs, totals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{acc:.0%}\n(n={tot})", ha="center", va="bottom", fontsize=8)
ax1.set_xlabel("Composition Depth")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy by Depth\n(Depth-Degradation Curve)")
ax1.set_ylim(0, 110)
ax1.set_xticks(depths)
ax1.legend(fontsize=8)
ax1.set_facecolor(BG)
ax1.grid(axis="y", alpha=0.3, zorder=0)

# ── 2. Family accuracy ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
fam_data = report["accuracy_by_family"]
fam_labels = [k.replace(" ", "\n") for k in fam_data.keys()]
fam_accs   = [v["accuracy"] * 100 for v in fam_data.values()]
fam_colors = [BLUE, GREEN, ORANGE, PURPLE]
bars2 = ax2.bar(range(len(fam_labels)), fam_accs, color=fam_colors[:len(fam_labels)], alpha=0.85, zorder=3)
for bar, acc, (k, v) in zip(bars2, fam_accs, fam_data.items()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{acc:.0f}%\n({v['correct']}/{v['total']})", ha="center", va="bottom", fontsize=8)
ax2.set_xticks(range(len(fam_labels)))
ax2.set_xticklabels(fam_labels, fontsize=8)
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy by Task Family")
ax2.set_ylim(0, 110)
ax2.set_facecolor(BG)
ax2.grid(axis="y", alpha=0.3, zorder=0)

# ── 3. Dimension accuracy ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
dim_data = {k: v for k, v in report["accuracy_by_dimension"].items() if v["total"] > 0}
dim_labels = list(dim_data.keys())
dim_accs   = [v["accuracy"] * 100 for v in dim_data.values()]
dim_colors = [BLUE, GREEN, ORANGE, PURPLE, RED]
bars3 = ax3.barh(range(len(dim_labels)), dim_accs, color=dim_colors[:len(dim_labels)], alpha=0.85, zorder=3)
for bar, acc, (k, v) in zip(bars3, dim_accs, dim_data.items()):
    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{acc:.0f}% ({v['correct']}/{v['total']})", va="center", fontsize=8)
ax3.set_yticks(range(len(dim_labels)))
ax3.set_yticklabels([d.capitalize() for d in dim_labels], fontsize=9)
ax3.set_xlabel("Accuracy (%)")
ax3.set_title("Accuracy by Hupkes Dimension")
ax3.set_xlim(0, 115)
ax3.set_facecolor(BG)
ax3.grid(axis="x", alpha=0.3, zorder=0)

# ── 4. Algebraic Complexity Metrics ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
metric_names = ["H_alg\n(Entropy)", "D_comm\n(Commutativity)", "O_c\n(Orbit)", "I_s\n(Interference)"]
metric_vals  = [
    COMPLEXITY["avg_algebraic_entropy"],
    COMPLEXITY["avg_commutativity_distance"],
    COMPLEXITY["avg_orbit_complexity"],
    COMPLEXITY["avg_structural_interference"],
]
metric_colors = [BLUE, ORANGE, GREEN, PURPLE]
bars4 = ax4.bar(range(4), metric_vals, color=metric_colors, alpha=0.85, zorder=3)
for bar, val in zip(bars4, metric_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.4f}", ha="center", va="bottom", fontsize=8)
ax4.set_xticks(range(4))
ax4.set_xticklabels(metric_names, fontsize=8)
ax4.set_ylabel("Average Value")
ax4.set_title("Algebraic Complexity Metrics\n(Averages across all tasks)")
ax4.set_facecolor(BG)
ax4.grid(axis="y", alpha=0.3, zorder=0)

# ── 5. Depth × Family heatmap ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
families_list = ["intra-structure composition", "inter-structure composition",
                 "field arithmetic", "rule induction"]
fam_short = ["Intra", "Inter", "Field", "Rule"]

# Build matrix
matrix = np.zeros((len(families_list), len(depths)))
for di, d in enumerate(depths):
    for fi, fam in enumerate(families_list):
        # Find tasks matching this depth and family
        matching = [r for r in report.get("results", [])
                    if False]  # results not in JSON, use depth+family data
        pass

# Use depth data + family data to approximate
# Build from raw depth_data and family_data
depth_family_acc = {}
for d in depths:
    for fam in families_list:
        depth_family_acc[(d, fam)] = None

# Since we don't have cross-tabulated data in the JSON, show what we have
# as a simple summary table instead
ax5.axis("off")
table_data = []
for fam, fam_s in zip(families_list, fam_short):
    row = [fam_s]
    v = report["accuracy_by_family"].get(fam, {})
    row.append(f"{v.get('accuracy', 0):.0%}")
    row.append(f"{v.get('correct', 0)}/{v.get('total', 0)}")
    table_data.append(row)

table = ax5.table(
    cellText=table_data,
    colLabels=["Family", "Accuracy", "Correct/Total"],
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#E8F4FD")
    cell.set_edgecolor("#D1D5DB")
ax5.set_title("Family Summary Table")

# ── 6. Key metrics summary ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
summary_text = (
    f"Key Metrics Summary\n"
    f"{'─'*32}\n"
    f"Model:           {MODEL}\n"
    f"Total Tasks:     {TOTAL}\n"
    f"Correct:         {CORRECT}\n"
    f"Overall Acc:     {OVERALL_ACC:.1%}\n"
    f"\n"
    f"Compositional Ceiling\n"
    f"  @ 50%:  depth {CEILING_50}\n"
    f"  @ 25%:  depth {CEILING_25}\n"
    f"\n"
    f"Algebraic Complexity\n"
    f"  H_alg:   {COMPLEXITY['avg_algebraic_entropy']:.4f}\n"
    f"  D_comm:  {COMPLEXITY['avg_commutativity_distance']:.4f}\n"
    f"  O_c:     {COMPLEXITY['avg_orbit_complexity']:.4f}\n"
    f"  I_s:     {COMPLEXITY['avg_structural_interference']:.4f}\n"
)
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=BLUE, alpha=0.9))
ax6.set_title("Summary")

plt.savefig("results/compeval_results.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
print("Chart saved to results/compeval_results.png")
