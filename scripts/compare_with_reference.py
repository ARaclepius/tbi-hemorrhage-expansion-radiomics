"""Compare a fresh results_pca95.csv with the reference results.

    python scripts/compare_with_reference.py results/results_pca95.csv [--tol 0.005]

Exit code 0 if every metric of every model is within the tolerance.
"""
import argparse
import sys
import pandas as pd

METRICS = ["oof_auc", "oof_acc", "oof_pr_auc", "oof_f1", "sensitivity", "specificity",
           "ppv", "npv", "balanced_accuracy", "mean_pca_components"]

ap = argparse.ArgumentParser()
ap.add_argument("new")
ap.add_argument("--reference", default="reference_results/results_pca95.csv")
ap.add_argument("--tol", type=float, default=0.005, help="max absolute difference per metric")
a = ap.parse_args()

new, ref = pd.read_csv(a.new), pd.read_csv(a.reference)
m = new.merge(ref, on=["feature_set", "model"], suffixes=("_new", "_ref"))
if len(m) != len(ref):
    sys.exit(f"Only {len(m)} of {len(ref)} reference models found in {a.new}")
m["max_abs_diff"] = [max(abs(r[c + "_new"] - r[c + "_ref"]) *
                         (0.1 if c == "mean_pca_components" else 1) for c in METRICS)
                     for _, r in m.iterrows()]
m["ok"] = m.max_abs_diff <= a.tol
print(m[["feature_set", "model", "oof_auc_new", "oof_auc_ref", "max_abs_diff", "ok"]]
      .round(4).to_string(index=False))
n_bad = int((~m.ok).sum())
print(f"\n{len(m) - n_bad}/{len(m)} models match the reference (tol = {a.tol}).")
sys.exit(1 if n_bad else 0)
