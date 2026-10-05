import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from pathlib import Path

BUILD = Path(__file__).parent / "cmake-build-debug"
bp = pd.read_csv(BUILD / "bp_log.csv")

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

# ── left panel: variance vs qubits (log scale) ────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.semilogy(bp["n_qubits"], bp["var_grad"],
            color="#1f77b4", lw=1.8, marker="o", ms=6, label="Gradient variance")
ax.fill_between(bp["n_qubits"],
                (bp["var_grad"] - bp["std_grad"]).clip(lower=1e-4),
                bp["var_grad"] + bp["std_grad"],
                alpha=0.15, color="#1f77b4", label="±1 std dev")
ax.set_xlabel("Number of Qubits")
ax.set_ylabel("Gradient Variance (log scale)")
ax.set_title("Barren Plateaus: Gradient Variance vs Circuit Width")
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, which="both")
ax.spines[["top", "right"]].set_visible(False)

# ── right panel: ratio table ──────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_title("Var[n+1] / Var[n] Ratio", pad=10)

ratios = [bp["var_grad"].iloc[i] / bp["var_grad"].iloc[i-1]
          for i in range(1, len(bp))]
cell_text = [[f"{bp['n_qubits'].iloc[i-1]}→{bp['n_qubits'].iloc[i]}",
              f"{r:.3f}"] for i, r in enumerate(ratios, 1)]
col_labels  = ["Qubits", "Ratio"]

tbl = ax2.table(cellText=cell_text, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 2.1)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if row == 0 or col == -1:
        cell.set_facecolor("#d0dff0")
        cell.set_text_props(fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f4f4f4")
    else:
        cell.set_facecolor("white")

out = Path(__file__).parent / "results_barren.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
