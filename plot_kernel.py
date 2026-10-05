import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BUILD = Path(__file__).parent / "cmake-build-debug"
km = pd.read_csv(BUILD / "kernel_log.csv")

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi":     150,
})

fig = plt.figure(figsize=(12, 4.4))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.7, 1.3], wspace=0.45)

# ── left panel: bar chart ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
colors = ["#1f77b4", "#d62728", "#2ca02c"]
bars = ax.bar(km["model"], km["test_acc"], color=colors, width=0.5, zorder=3)
for bar, val in zip(bars, km["test_acc"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
            f"{val:.0%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
ax.set_ylabel("Test Accuracy")
ax.set_title("Model Comparison: QNN vs MLP vs QK-SVM")
ax.set_ylim(0, 1.10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(True, axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# ── right panel: table ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_title("Details", pad=10)

cell_text = [
    ["2",        "21",       "61 (SV)"],
    ["0.62",     "0.80",     "0.92"],
    ["1.36 s",   "0.03 s",   "0.09 s"],
    ["param-shift", "backprop", "dual grad."],
]
row_labels = ["Parameters", "Test acc.", "Time", "Gradient"]
col_labels  = ["QNN", "MLP", "QK-SVM"]

tbl = ax2.table(cellText=cell_text, rowLabels=row_labels,
                colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 2.05)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if row == 0 or col == -1:
        cell.set_facecolor("#d0dff0")
        cell.set_text_props(fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f4f4f4")
    else:
        cell.set_facecolor("white")

out = Path(__file__).parent / "results_kernel.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
