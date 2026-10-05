import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

BUILD = Path(__file__).parent / "cmake-build-debug"
vqe = pd.read_csv(BUILD / "vqe_log.csv")
exact = -1.851199

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

# ── left panel: convergence curve ────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
ax.plot(vqe["step"], vqe["energy"], color="#1f77b4", lw=1.8, label="VQE energy")
ax.axhline(exact, color="#d62728", lw=1.5, ls="--", label=f"Exact: {exact} Ha")
ax.set_xlabel("Optimization Step")
ax.set_ylabel("Energy (Ha)")
ax.set_title(r"VQE Convergence: $H_2$ Ground State (STO-3G)")
ax.set_xlim(1, len(vqe))
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# ── right panel: table ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_title("VQE Summary", pad=10)

final_e = vqe["energy"].iloc[-1]
error   = abs(final_e - exact)
cell_text = [
    ["H₂ (STO-3G)"],
    ["HEA, 2 qubits"],
    ["4"],
    ["param-shift GD"],
    ["200"],
    [f"{final_e:.6f} Ha"],
    [f"{error:.6f} Ha"],
    ["YES" if error < 0.01 else "NO"],
]
row_labels = ["Molecule", "Ansatz", "Parameters", "Optimizer",
              "Steps", "VQE energy", "Error", "Chem. accuracy"]

tbl = ax2.table(cellText=cell_text, rowLabels=row_labels,
                colLabels=["Value"], loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.6)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if row == 0 or col == -1:
        cell.set_facecolor("#d0dff0")
        cell.set_text_props(fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f4f4f4")
    else:
        cell.set_facecolor("white")

out = Path(__file__).parent / "results_vqe.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
