"""Draw the PCA-95% AUC summary figure from a results CSV.

    python scripts/make_results_figure.py reference_results/results_pca95.csv assets/auc_summary.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "reference_results/results_pca95.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "assets/auc_summary.png"
df = pd.read_csv(src)
models = ["Random Forest", "Gradient Boosting", "XGBoost", "MLP"]
sets = [("Radiomics", "#0ea5e9"), ("Clinical", "#8b5cf6"), ("Combined", "#14b8a6")]

fig, ax = plt.subplots(figsize=(10.5, 4.9))
x = np.arange(len(models)); w = 0.26
for k, (fs, colr) in enumerate(sets):
    v = [df[(df.feature_set == fs) & (df.model == m)].oof_auc.iloc[0] for m in models]
    bars = ax.bar(x + (k - 1) * w, v, w, color=colr, label=fs, zorder=3)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + .008, f"{val:.3f}", ha="center",
                va="bottom", fontsize=8.5, color="#334155")
ax.axhline(.5, color="#94a3b8", ls="--", lw=1, zorder=2)
ax.text(-0.49, .503, "chance", fontsize=8.5, color="#64748b", ha="left", va="bottom")
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(.45, .92); ax.set_ylabel("Pooled out-of-fold AUC", fontsize=11)
ax.grid(axis="y", color="#e2e8f0", zorder=0)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=10.5)
ax.set_title("PCA-based models: discrimination of hemorrhage expansion (5-fold stratified CV, n = 86)",
             loc="left", fontsize=12.5, color="#0f172a", pad=14)
fig.tight_layout(); fig.savefig(dst, dpi=170, bbox_inches="tight", facecolor="white")
print("saved", dst)
