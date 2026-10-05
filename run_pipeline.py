#!/usr/bin/env python
"""Run 3 feature sets x 4 classifiers, all with PCA (95% variance) = 12 models.

Usage:
    python run_pipeline.py --data data/TBI_final.csv --output results/
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn, imblearn, xgboost

from tbi_pipeline import __version__, config as C
from tbi_pipeline.data import prepare_dataset
from tbi_pipeline.evaluation import cross_validate_oof, make_folds
from tbi_pipeline.features import build_feature_sets
from tbi_pipeline.pipeline import get_models
from tbi_pipeline.plots import plot_confusion_matrix, plot_rocs

PCT = ["sensitivity", "specificity", "ppv", "npv", "oof_acc", "oof_auc", "oof_f1"]


def slug(s):
    return s.lower().replace(" ", "_").replace("%", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/TBI_final.csv")
    ap.add_argument("--output", default="results")
    ap.add_argument("--feature-sets", nargs="+", default=C.FEATURE_SET_NAMES,
                    choices=C.FEATURE_SET_NAMES)
    args = ap.parse_args()

    out = Path(args.output)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data ...")
    X, y, info = prepare_dataset(args.data)
    print("Feature sets:")
    feature_sets = build_feature_sets(X)
    folds = make_folds(X, y)          # identical folds for every model

    rows = []
    for fs in args.feature_sets:
        Xm, rocs = feature_sets[fs], {}
        for name, clf in get_models().items():
            print(f"\n[{fs} | PCA 95% | {name}]")
            m, roc, pcs, _ = cross_validate_oof(Xm, y, clf, folds)
            rocs[name] = {**roc, "auc": m["oof_auc"]}
            rows.append({
                "feature_set": fs, "pca": "PCA 95%", "model": name,
                "n_features": Xm.shape[1],
                **{k: v for k, v in m.items() if k != "cm"},
                "tn": int(m["cm"][0, 0]), "fp": int(m["cm"][0, 1]),
                "fn": int(m["cm"][1, 0]), "tp": int(m["cm"][1, 1]),
                "mean_pca_components": float(np.mean(pcs)),
                "pca_components_per_fold": pcs,
            })
            plot_confusion_matrix(
                m["cm"], f"{fs} - {name}\nPCA 95% - pooled OOF confusion matrix",
                out / "figures" / f"CM_{slug(fs)}_{slug(name)}.png")
            print(f"    AUC={m['oof_auc']:.4f} Acc={m['oof_acc']:.4f} "
                  f"Sens={m['sensitivity']:.4f} Spec={m['specificity']:.4f}")
        plot_rocs(fs, rocs, out / "figures" / f"ROC_{slug(fs)}.png")

    res = pd.DataFrame(rows)
    res.to_csv(out / "results_pca95.csv", index=False)

    for fs in args.feature_sets:  # manuscript-style tables (percent; AUC as %)
        t = res.loc[res.feature_set == fs, ["model"] + PCT].copy()
        t[PCT] = (t[PCT] * 100).round(2)
        t.columns = ["Model", "Sensitivity (%)", "Specificity (%)", "PPV (%)",
                     "NPV (%)", "Accuracy (%)", "AUC (%)", "F1 (%)"]
        t.to_csv(out / f"table_{slug(fs)}.csv", index=False)
        print(f"\n{fs}\n{t.to_string(index=False)}")

    meta = {
        "pipeline_version": __version__, "random_state": C.RANDOM_STATE,
        "n_splits": C.N_SPLITS, "pca_variance": C.PCA_VARIANCE,
        "n_patients": info["n_patients"], "class_counts": info["class_counts"],
        "dropped_columns": info["dropped_columns"],
        "feature_set_sizes": {k: v.shape[1] for k, v in feature_sets.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "imbalanced-learn": imblearn.__version__,
                     "xgboost": xgboost.__version__},
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. {len(res)} models written to {out}/")


if __name__ == "__main__":
    sys.exit(main())
