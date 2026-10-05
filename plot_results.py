import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── dane ─────────────────────────────────────────────────────────────────────

BUILD = Path(__file__).parent / "cmake-build-debug"
qnn = pd.read_csv(BUILD / "qnn_log.csv")
mlp = pd.read_csv(BUILD / "mlp_log.csv")

QNN_TIME   = "1.36 s"
MLP_TIME   = "0.03 s"
QNN_PARAMS = 4
MLP_PARAMS = 21

# ── styl ─────────────────────────────────────────────────────────────────────

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

# ── lewy panel: krzywe uczenia ────────────────────────────────────────────────

ax = fig.add_subplot(gs[0])

ax.plot(qnn["epoch"], qnn["train_acc"], color="#1f77b4", lw=1.8, label="QNN – train")
ax.plot(qnn["epoch"], qnn["test_acc"],  color="#1f77b4", lw=1.8, ls="--", label="QNN – test")
ax.plot(mlp["epoch"], mlp["train_acc"], color="#d62728", lw=1.8, label="MLP – train")
ax.plot(mlp["epoch"], mlp["test_acc"],  color="#d62728", lw=1.8, ls="--", label="MLP – test")

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curves – QNN vs MLP")
ax.set_xlim(1, max(qnn["epoch"].max(), mlp["epoch"].max()))
ax.set_ylim(0.45, 1.00)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# ── prawy panel: tabela ───────────────────────────────────────────────────────

ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_title("Model Comparison", pad=10)

cell_text = [
    ["2 qubits, L=2",                        "2 → 4 → 1"                     ],
    [str(QNN_PARAMS),                        str(MLP_PARAMS)                  ],
    [f"{qnn['test_acc'].iloc[-1]:.1%}",      f"{mlp['test_acc'].iloc[-1]:.1%}"],
    [QNN_TIME,                               MLP_TIME                         ],
    ["parameter shift",                      "backpropagation"                ],
]
row_labels = ["Architecture", "Parameters", "Test acc.", "Training time", "Gradient"]
col_labels = ["QNN", "MLP"]

tbl = ax2.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 2.05)

# styl nagłówka kolumn i etykiet wierszy
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#bbbbbb")
    if row == 0 or col == -1:           # nagłówki
        cell.set_facecolor("#d0dff0")
        cell.set_text_props(fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f4f4f4")
    else:
        cell.set_facecolor("white")

# ── zapis ─────────────────────────────────────────────────────────────────────

out = Path(__file__).parent / "results.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
