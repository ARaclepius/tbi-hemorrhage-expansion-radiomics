# ============================================================================
# TBI HEMORRHAGE-EXPANSION RADIOMICS: ONE-CELL GOOGLE COLAB PIPELINE
# Based on ARaclepius/tbi-hemorrhage-expansion-radiomics (PCA 95% pipeline)
# Adds: stratified bootstrap 95% CIs on pooled OOF predictions, back-mapped
# feature importance averaged across folds, and fold-wise MLP SHAP + an
# explicitly labelled PCA-loading-based approximation in the original feature
# space. The patient-level dataset is not included in the GitHub repository.
# ============================================================================

# ------------------------------ DEFAULT SETTINGS ----------------------------
# Values can be overridden with command-line options in main().
CSV_PATH = "data/TBI_final_corrected.csv"
OUTPUT_DIR = "results"
RANDOM_STATE = 42
N_SPLITS = 5
PCA_VARIANCE = 0.95
N_BOOTSTRAP = 2000
SHAP_NSAMPLES = 100
SHAP_BACKGROUND_SIZE = 10
TOP_N_FEATURES = 20
RUN_SHAP = True
# -----------------------------------------------------------------------------

import argparse
import os, sys, re, json, time, platform, warnings
from pathlib import Path

# Install the pinned/minimum dependencies from requirements.txt before running.
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn, imblearn, xgboost
try:
    import shap
except ImportError:  # permits --skip-shap environments; full runs require SHAP
    shap = None
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# Match the exclusions and label rules in tbi_pipeline/config.py and data.py.
EXACT_EXCLUSIONS = {
    "surgery", "outcome",
    "lenght of hospitalization(days) in icu",
    "length of hospitalization(days) in icu",
    "icu length of stay", "rebleeding on control ct scans", "timestamp",
    "patient", "patient id", "patient_id", "patient_name",
    "__patient_original__", "type of hemorrhage", "rebleeding",
}
REGEX_EXCLUSIONS = [
    r"^patient$", r"patient[_\s-]*name", r"__patient_original__",
    r"patient[_\s-]*id", r"timestamp", r"record[_\s-]*(id|timestamp|time)",
    r"^surgery$", r"^outcome$", r"mortality", r"death", r"icu",
    r"length[_\s-]*of[_\s-]*stay", r"hospitalization", r"follow[_\s-]*up",
    r"control[_\s-]*ct", r"rebleeding[_\s-]*timing",
    r"rebleeding[_\s-]*on[_\s-]*control", r"^rebleeding$",
    r"^unnamed[:\s_]*0$", r"^index$",
]
RADIOMICS_CLASSES = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
NEGATIVE_LABELS = {"0", "no", "n", "negative", "no rebleeding", "none", "absent", "no expansion", "false"}
POSITIVE_LABELS = {"1", "yes", "y", "positive", "rebleeding", "expansion", "hemorrhage expansion", "present", "true"}
FEATURE_SET_NAMES = ["Radiomics", "Clinical", "Combined"]
MODEL_NAMES_ORDER = ["Random Forest", "Gradient Boosting", "XGBoost", "MLP"]
METRIC_COLUMNS = [
    "oof_auc", "oof_pr_auc", "oof_acc", "sensitivity", "specificity",
    "ppv", "npv", "oof_f1", "balanced_accuracy",
]
METRIC_LABELS = {
    "oof_auc": "AUC", "oof_pr_auc": "PR-AUC / Average precision",
    "oof_acc": "Accuracy", "sensitivity": "Sensitivity",
    "specificity": "Specificity", "ppv": "PPV", "npv": "NPV",
    "oof_f1": "F1", "balanced_accuracy": "Balanced accuracy",
}

# Output directories and input validation are handled by _run() so this module
# can be imported by tests without touching files or prompting for an upload.

# --------------------------------- HELPERS -----------------------------------
def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _normalize_target(value):
    """Same target normalisation as the repository: non-empty unknown labels are positive."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 0 if float(value) == 0 else 1
    label = re.sub(r"\s+", " ", str(value).strip().lower())
    if label in NEGATIVE_LABELS:
        return 0
    if label in POSITIVE_LABELS:
        return 1
    if label.startswith(("until", "between", "after")):
        return 1
    return 1 if label != "" else np.nan


def _should_drop_column(column):
    name = str(column).strip()
    if name.lower() in EXACT_EXCLUSIONS:
        return True
    return any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in REGEX_EXCLUSIONS)


def _clean_numeric_like_columns(X, threshold=0.80):
    X = X.copy()
    report = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        raw = X[col]
        cleaned = raw.astype("string").str.strip().str.replace(",", "", regex=False)
        converted = pd.to_numeric(cleaned, errors="coerce")
        n_nonmissing = int(raw.notna().sum())
        if n_nonmissing == 0:
            continue
        rate = float(converted.notna().sum() / n_nonmissing)
        if rate >= threshold:
            invalid = int((raw.notna() & converted.isna()).sum())
            X[col] = converted.astype(float)
            report.append({"column": str(col), "type": "numeric", "conversion_rate": rate, "invalid_to_nan": invalid})
        else:
            report.append({"column": str(col), "type": "categorical", "conversion_rate": rate, "invalid_to_nan": 0})
    return X, report


def _prepare_dataset(csv_path):
    raw_df = pd.read_csv(csv_path, encoding="utf-8-sig")
    raw_df = raw_df.drop(columns=[c for c in raw_df.columns if str(c).strip().lower().startswith("unnamed:")], errors="ignore")
    target_candidates = [c for c in raw_df.columns if str(c).strip().lower() == "rebleeding"]
    if not target_candidates:
        raise ValueError("Target column not found. Expected a column named 'rebleeding' (case-insensitive).")
    target_col = target_candidates[0]
    y = raw_df[target_col].map(_normalize_target)
    keep = y.notna()
    n_missing_target = int((~keep).sum())
    df = raw_df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True).astype(int)
    X = df.drop(columns=[target_col]).copy()

    dropped = [c for c in X.columns if _should_drop_column(c)]
    X = X.drop(columns=dropped, errors="ignore")
    all_nan = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=all_nan, errors="ignore")
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant, errors="ignore")
    dropped = list(dict.fromkeys([str(c) for c in dropped + all_nan + constant]))
    X, conversion_report = _clean_numeric_like_columns(X)

    if len(X) != len(y):
        raise RuntimeError("Internal error: predictors and target have mismatched row counts.")
    counts = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
    if len(counts) != 2 or min(counts.values()) < N_SPLITS:
        raise ValueError(f"Need both target classes with at least {N_SPLITS} cases each; observed class counts: {counts}")
    info = {
        "n_patients": int(len(X)), "class_counts": counts,
        "dropped_columns": dropped, "conversion_report": conversion_report,
        "n_rows_removed_missing_target": n_missing_target,
        "target_column_used": str(target_col),
    }
    return X, y, info


def _is_radiomics_feature(column):
    name = str(column).lower()
    if any(tag in name for tag in ("original__", "wavelet__", "original_", "wavelet_")):
        return True
    return any(f"__{feature_class}__" in name or f"_{feature_class}_" in name for feature_class in RADIOMICS_CLASSES)


def _build_feature_sets(X):
    radiomics_cols = [c for c in X.columns if _is_radiomics_feature(c)]
    clinical_cols = [c for c in X.columns if c not in radiomics_cols]
    feature_sets = {
        "Radiomics": X[radiomics_cols].copy(),
        "Clinical": X[clinical_cols].copy(),
        "Combined": X.copy(),
    }
    for name, frame in feature_sets.items():
        if frame.shape[1] == 0:
            raise ValueError(
                f"Feature set '{name}' has zero columns. Check column names and the repository's "
                "radiomics name-detection rule (_is_radiomics_feature)."
            )
    print("Feature sets:")
    for name, frame in feature_sets.items():
        print(f"  {name:10s}: {frame.shape[0]} patients x {frame.shape[1]} features")
    if len(radiomics_cols) != 107:
        print(f"NOTE: detected {len(radiomics_cols)} radiomics columns; repository sanity-check expects 107 original features.")
    return feature_sets


def _get_models(seed=RANDOM_STATE):
    # Settings match tbi_pipeline/pipeline.py and README.md.
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
            eval_metric="logloss", random_state=seed, n_jobs=-1,
        ),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=seed),
    }


def _onehot_kwargs():
    try:
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return {"handle_unknown": "ignore", "sparse_output": False}
    except TypeError:
        return {"handle_unknown": "ignore", "sparse": False}


def _build_pipeline(X_train, classifier, seed=RANDOM_STATE):
    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    transformers = []
    if numeric_cols:
        numeric_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**_onehot_kwargs())),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    # Deliberately retains the repository's order: preprocessing -> ROS -> PCA -> model.
    return Pipeline([
        ("preprocess", preprocessor),
        ("sampler", RandomOverSampler(random_state=seed)),
        ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=seed)),
        ("model", classifier),
    ])


def _calculate_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    safe = lambda numerator, denominator: (numerator / denominator) if denominator else 0.0
    return {
        "oof_auc": float(roc_auc_score(y_true, y_prob)),
        "oof_pr_auc": float(average_precision_score(y_true, y_prob)),
        "oof_acc": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(safe(tp, tp + fn)),
        "specificity": float(safe(tn, tn + fp)),
        "ppv": float(safe(tp, tp + fp)),
        "npv": float(safe(tn, tn + fn)),
        "oof_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cm": cm,
    }


def _stratified_bootstrap_ci(y, y_prob, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    """Percentile CI from stratified resampling of pooled OOF (y, probability) pairs."""
    y = np.asarray(y, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)  # repository threshold is fixed at 0.5
    class0 = np.flatnonzero(y == 0)
    class1 = np.flatnonzero(y == 1)
    if not len(class0) or not len(class1):
        raise ValueError("Stratified bootstrap requires both target classes.")
    rng = np.random.default_rng(seed)
    observed = _calculate_metrics(y, y_pred, y_prob)
    samples = {metric: [] for metric in METRIC_COLUMNS}
    for _ in range(int(n_bootstrap)):
        # Sample within each class at its original sample size, then pool.
        idx = np.concatenate([
            rng.choice(class0, size=len(class0), replace=True),
            rng.choice(class1, size=len(class1), replace=True),
        ])
        boot = _calculate_metrics(y[idx], y_pred[idx], y_prob[idx])
        for metric in METRIC_COLUMNS:
            samples[metric].append(boot[metric])
    results = {}
    for metric in METRIC_COLUMNS:
        lo, hi = np.percentile(np.asarray(samples[metric]), [2.5, 97.5])
        results[f"{metric}_ci_lower"] = float(lo)
        results[f"{metric}_ci_upper"] = float(hi)
    return observed, results



# -------------------------- PAIRED DELONG AUC TEST ---------------------------
def _compute_midrank(values):
    """Compute 1-based midranks, averaging the ranks of tied values."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    sorted_midranks = np.empty(n, dtype=float)
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        # Positions start..end-1 are zero-based, so add 1 to the average rank.
        sorted_midranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    midranks = np.empty(n, dtype=float)
    midranks[order] = sorted_midranks
    return midranks


def _delong_auc_test(y_true, scores_a, scores_b):
    """Paired DeLong test comparing two correlated ROC AUCs on the same cases.

    Returns AUC A, AUC B, AUC(A)-AUC(B), z statistic, and two-sided p value.
    This standard asymptotic test is applied to the pooled OOF probabilities;
    inference is approximate because CV-fold training sets overlap.
    """
    from scipy.stats import norm
    y_true = np.asarray(y_true, dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("DeLong test inputs must have equal lengths.")
    if not np.isfinite(scores_a).all() or not np.isfinite(scores_b).all():
        raise ValueError("DeLong test received non-finite predicted probabilities.")
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))
    if positive_count < 2 or negative_count < 2:
        raise ValueError("DeLong test requires at least two cases from each target class.")

    # Sort cases so positives precede negatives, as required by fast DeLong.
    order = np.argsort(-y_true)
    predictions = np.vstack([scores_a, scores_b])[:, order]
    m, n = positive_count, negative_count
    k = predictions.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(predictions[r, :m])
        ty[r] = _compute_midrank(predictions[r, m:])
        tz[r] = _compute_midrank(predictions[r, :])

    aucs_rank = (tz[:, :m].sum(axis=1) / m - (m + 1.0) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01, ddof=1) / m + np.cov(v10, ddof=1) / n
    auc_a = float(roc_auc_score(y_true, scores_a))
    auc_b = float(roc_auc_score(y_true, scores_b))
    # Guard against implementation errors in rank-based AUC calculation.
    if not np.allclose(aucs_rank, [auc_a, auc_b], atol=1e-8, rtol=1e-8):
        raise RuntimeError("Internal DeLong check failed: rank-based AUC disagrees with sklearn.")

    difference = auc_a - auc_b
    variance_difference = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    variance_difference = max(0.0, variance_difference)  # numerical round-off guard
    if variance_difference <= 1e-15:
        if abs(difference) <= 1e-12:
            z_statistic, p_value = 0.0, 1.0
        else:
            z_statistic, p_value = float(np.sign(difference) * np.inf), 0.0
    else:
        z_statistic = float(difference / np.sqrt(variance_difference))
        p_value = float(2.0 * norm.sf(abs(z_statistic)))
    return auc_a, auc_b, float(difference), float(z_statistic), float(p_value)


def _holm_adjust(p_values):
    """Holm step-down FWER adjustment; returns values in their original order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p_values))
    if len(valid) == 0:
        return adjusted
    p = np.clip(p_values[valid], 0.0, 1.0)
    order = np.argsort(p)
    sorted_p = p[order]
    m = len(sorted_p)
    sorted_adjusted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)
    valid_ordered = valid[order]
    adjusted[valid_ordered] = sorted_adjusted
    return adjusted


def _plot_delong_auc_difference_heatmap(config_labels, difference_matrix, path):
    """Plot pairwise AUC differences; each cell is row AUC minus column AUC."""
    n = len(config_labels)
    fig_w = max(10.0, 0.72 * n + 4.0)
    fig_h = max(8.0, 0.62 * n + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = max(float(np.nanmax(np.abs(difference_matrix))), 0.01)
    im = ax.imshow(difference_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(n), labels=config_labels, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n), labels=config_labels, fontsize=8)
    ax.set_title("Pairwise pooled OOF AUC differences (row minus column)\nPaired DeLong tests reported in CSV tables")
    for i in range(n):
        for j in range(n):
            value = difference_matrix[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(value) > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, label="AUC difference")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = ["No expansion", "Expansion"]
    ax.set_xticks([0, 1], labels=labels, rotation=15, ha="right")
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    midpoint = cm.max() / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > midpoint else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_rocs(feature_set, roc_by_model, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for model_name, info in roc_by_model.items():
        ax.plot(info["fpr"], info["tpr"], lw=2, label=f"{model_name} (AUC={info['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
           title=f"Pooled out-of-fold ROC — {feature_set} (PCA 95%)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_auc_ci(results, path):
    plot_df = results.sort_values("oof_auc", ascending=True).reset_index(drop=True)
    labels = [f"{r.feature_set} | {r.model}" for r in plot_df.itertuples()]
    x = plot_df["oof_auc"].to_numpy()
    low = plot_df["oof_auc_ci_lower"].to_numpy()
    high = plot_df["oof_auc_ci_upper"].to_numpy()
    err = np.vstack([np.maximum(0, x - low), np.maximum(0, high - x)])
    fig_h = max(6.5, 0.42 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    yloc = np.arange(len(plot_df))
    ax.errorbar(x, yloc, xerr=err, fmt="o", capsize=3, elinewidth=1.5)
    ax.axvline(0.5, linestyle="--", linewidth=1, color="grey", label="Chance AUC")
    ax.set_yticks(yloc, labels=labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pooled OOF AUC (95% stratified-bootstrap CI)")
    ax.set_title("AUC with stratified-bootstrap 95% confidence intervals")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ci_text(row, metric, percent=True):
    point = float(row[metric])
    lo = float(row[f"{metric}_ci_lower"])
    hi = float(row[f"{metric}_ci_upper"])
    if percent:
        return f"{100*point:.1f} ({100*lo:.1f}–{100*hi:.1f})"
    return f"{point:.3f} ({lo:.3f}–{hi:.3f})"

# ----------------------- POST-HOC INTERPRETABILITY ---------------------------
def _encoded_to_source_feature_map(pipe, X_train):
    """Return original column name for every post-imputation/one-hot column."""
    pre = pipe.named_steps["preprocess"]
    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]
    source_names = list(numeric_cols)
    if categorical_cols:
        cat_pipe = pre.named_transformers_["cat"]
        onehot = cat_pipe.named_steps["onehot"]
        for col, categories in zip(categorical_cols, onehot.categories_):
            source_names.extend([col] * len(categories))
    n_preprocessed = len(pre.get_feature_names_out())
    if len(source_names) != n_preprocessed:
        raise RuntimeError(
            "Could not map one-hot encoded variables back to source columns: "
            f"mapping has {len(source_names)} names but transformer emits {n_preprocessed} features."
        )
    return source_names


def _component_loading_weights(pca):
    """Absolute loading proportions: each PC importance is allocated over input features."""
    weights = np.abs(np.asarray(pca.components_, dtype=float))
    denom = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)
    return weights  # n_components x n_encoded_features; each PC row sums to 1


def _aggregate_encoded_to_sources(encoded_values, source_names, source_columns):
    """Aggregate dummy-variable contributions into their original source variable."""
    encoded_values = np.asarray(encoded_values, dtype=float)
    output = np.zeros((encoded_values.shape[0], len(source_columns)), dtype=float)
    source_to_idx = {name: i for i, name in enumerate(source_columns)}
    for encoded_i, source in enumerate(source_names):
        if source in source_to_idx:
            output[:, source_to_idx[source]] += encoded_values[:, encoded_i]
    return output


def _get_binary_shap_matrix(explainer, X_pc, nsamples=SHAP_NSAMPLES):
    """Compatibility wrapper for SHAP versions with slightly different APIs."""
    try:
        values = explainer.shap_values(X_pc, nsamples=nsamples, l1_reg="num_features(15)", silent=True)
    except TypeError:
        try:
            values = explainer.shap_values(X_pc, nsamples=nsamples, l1_reg="num_features(15)")
        except TypeError:
            values = explainer.shap_values(X_pc, nsamples=nsamples)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        # Depending on SHAP version a class dimension can be appended or inserted.
        if values.shape[1] == X_pc.shape[1]:
            values = values[:, :, -1]
        elif values.shape[2] == X_pc.shape[1]:
            values = values[:, -1, :]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape != X_pc.shape:
        raise ValueError(f"Unexpected SHAP shape {values.shape}; expected {X_pc.shape}.")
    return values


def _plot_importance(top_df, title, path, value_col="mean_normalized_importance"):
    if top_df.empty:
        return
    d = top_df.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.28 * len(d))))
    ax.barh(d["feature"].astype(str), d[value_col].astype(float))
    ax.set_xlabel("Mean normalized importance across folds")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_shap_beeswarm(shap_df, title, path):
    """A simple SHAP-attribution swarm plot, avoiding misleading categorical color encoding."""
    if shap_df.empty:
        return
    pivot = shap_df.pivot(index="sample_index", columns="feature", values="shap_value")
    mean_abs = pivot.abs().mean(axis=0).sort_values(ascending=False)
    features = list(mean_abs.head(TOP_N_FEATURES).index)
    fig, ax = plt.subplots(figsize=(10, max(5.5, 0.36 * len(features))))
    rng = np.random.default_rng(RANDOM_STATE)
    for ypos, feature in enumerate(features):
        vals = pivot[feature].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.22, 0.22, size=len(vals))
        ax.scatter(vals, ypos + jitter, s=17, alpha=0.65, linewidths=0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(features)), labels=features)
    ax.invert_yaxis()
    ax.set_xlabel("Back-mapped SHAP attribution to predicted expansion probability")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _run():
    """Execute CV evaluation, uncertainty quantification, model comparison and interpretation."""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    FIG_DIR = Path(OUTPUT_DIR) / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"Input CSV not found: {CSV_PATH}. Pass --data /path/to/TBI_final_corrected.csv. "
            "The patient-level dataset is intentionally not included in this repository."
        )
    if RUN_SHAP and shap is None:
        raise ImportError("SHAP is required for the full interpretability pass. Install requirements.txt or pass --skip-shap.")
    # ---------------------------------- RUN --------------------------------------
    start_time = time.time()
    print("Loading and cleaning dataset ...")
    X, y, data_info = _prepare_dataset(CSV_PATH)
    print(f"Patients analysed: {len(X)} | class counts: {data_info['class_counts']} | removed for missing target: {data_info['n_rows_removed_missing_target']}")
    feature_sets = _build_feature_sets(X)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))  # same patient folds for all feature sets and models

    # 1) Primary evaluation: one pooled out-of-fold prediction per patient/model.
    result_rows = []
    oof_rows = []
    roc_records = {}
    fold_pca_counts = {}

    for feature_set_name in FEATURE_SET_NAMES:
        X_set = feature_sets[feature_set_name]
        roc_by_model = {}
        for model_name, base_model in _get_models().items():
            print(f"\n[OOF CV] {feature_set_name} | PCA 95% | {model_name}")
            oof_prob = np.full(len(y), np.nan, dtype=float)
            oof_pred = np.full(len(y), -1, dtype=int)
            pca_counts = []
            for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
                pipe = _build_pipeline(X_set.iloc[train_idx], clone(base_model))
                pipe.fit(X_set.iloc[train_idx], y.iloc[train_idx])
                fold_prob = pipe.predict_proba(X_set.iloc[val_idx])[:, 1]
                oof_prob[val_idx] = fold_prob
                oof_pred[val_idx] = (fold_prob >= 0.5).astype(int)
                pca_counts.append(int(pipe.named_steps["pca"].n_components_))
                print(f"  fold {fold_no}/{N_SPLITS}: train={len(train_idx)} validation={len(val_idx)} PCs={pca_counts[-1]}")
            if np.isnan(oof_prob).any() or (oof_pred < 0).any():
                raise RuntimeError("At least one row lacks a pooled out-of-fold prediction.")

            observed, ci = _stratified_bootstrap_ci(y.to_numpy(), oof_prob, n_bootstrap=N_BOOTSTRAP,
                                                    seed=RANDOM_STATE + FEATURE_SET_NAMES.index(feature_set_name) * 100 + MODEL_NAMES_ORDER.index(model_name))
            cm = observed.pop("cm")
            row = {
                "feature_set": feature_set_name, "pca": "PCA 95%", "model": model_name,
                "n_patients": int(len(y)), "n_features": int(X_set.shape[1]),
                **observed, **ci,
                "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
                "mean_pca_components": float(np.mean(pca_counts)),
                "pca_components_per_fold": json.dumps(pca_counts),
                "bootstrap_replicates": int(N_BOOTSTRAP),
                "bootstrap_method": "stratified percentile bootstrap on pooled OOF predictions",
            }
            result_rows.append(row)
            fold_pca_counts[f"{feature_set_name}|{model_name}"] = pca_counts
            fpr, tpr, _ = roc_curve(y, oof_prob)
            roc_by_model[model_name] = {"fpr": fpr, "tpr": tpr, "auc": observed["oof_auc"]}
            _plot_confusion_matrix(cm, f"{feature_set_name} — {model_name}\nPooled OOF confusion matrix", FIG_DIR / f"CM_{_slug(feature_set_name)}_{_slug(model_name)}.png")
            for sample_idx in range(len(y)):
                oof_rows.append({
                    "sample_index": int(sample_idx), "feature_set": feature_set_name,
                    "model": model_name, "y_true": int(y.iloc[sample_idx]),
                    "oof_probability": float(oof_prob[sample_idx]),
                    "oof_prediction_threshold_0_5": int(oof_pred[sample_idx]),
                })
            print(f"  pooled OOF AUC={observed['oof_auc']:.3f} (95% CI {ci['oof_auc_ci_lower']:.3f}–{ci['oof_auc_ci_upper']:.3f}); accuracy={observed['oof_acc']:.3f}")
        _plot_rocs(feature_set_name, roc_by_model, FIG_DIR / f"ROC_{_slug(feature_set_name)}.png")
        roc_records[feature_set_name] = roc_by_model

    results = pd.DataFrame(result_rows)
    results.to_csv(Path(OUTPUT_DIR) / "results_pca95_with_bootstrap_ci.csv", index=False)
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(Path(OUTPUT_DIR) / "pooled_oof_predictions.csv", index=False)
    _plot_auc_ci(results, FIG_DIR / "AUC_95CI_summary.png")

    # 1b) Paired DeLong comparisons across all model x feature-set combinations.
    # Each pair uses the same patients and observed labels, so these are correlated
    # ROC curves. The all-pairs table covers all 12 configurations (66 comparisons).
    print("\nRunning paired DeLong tests on pooled OOF probabilities ...")
    y_array = y.to_numpy(dtype=int)
    configurations = []
    prediction_by_configuration = {}
    auc_by_configuration = {}
    for fs in FEATURE_SET_NAMES:
        for model_name in MODEL_NAMES_ORDER:
            group = oof_df[(oof_df["feature_set"] == fs) & (oof_df["model"] == model_name)].sort_values("sample_index")
            if len(group) != len(y_array):
                raise RuntimeError(f"Expected {len(y_array)} OOF predictions for {fs} | {model_name}; found {len(group)}.")
            group_y = group["y_true"].to_numpy(dtype=int)
            if not np.array_equal(group_y, y_array):
                raise RuntimeError(f"Observed labels/order differ across OOF predictions for {fs} | {model_name}.")
            key = (fs, model_name)
            prediction_by_configuration[key] = group["oof_probability"].to_numpy(dtype=float)
            auc_by_configuration[key] = float(roc_auc_score(y_array, prediction_by_configuration[key]))
            configurations.append(key)

    comparison_rows = []
    for i in range(len(configurations)):
        fs_a, model_a = configurations[i]
        key_a = configurations[i]
        for j in range(i + 1, len(configurations)):
            fs_b, model_b = configurations[j]
            key_b = configurations[j]
            auc_a, auc_b, diff, z_stat, p_value = _delong_auc_test(
                y_array, prediction_by_configuration[key_a], prediction_by_configuration[key_b]
            )
            if fs_a == fs_b:
                comparison_type = "same_feature_set_different_model"
            elif model_a == model_b:
                comparison_type = "same_model_different_feature_set"
            else:
                comparison_type = "different_feature_set_and_model"
            comparison_rows.append({
                "comparison_type": comparison_type,
                "configuration_a": f"{fs_a} | {model_a}",
                "configuration_b": f"{fs_b} | {model_b}",
                "feature_set_a": fs_a, "model_a": model_a,
                "feature_set_b": fs_b, "model_b": model_b,
                "auc_a": auc_a, "auc_b": auc_b,
                "auc_difference_a_minus_b": diff,
                "z_statistic": z_stat,
                "p_value_delong": p_value,
            })

    delong_all = pd.DataFrame(comparison_rows)
    delong_all["p_value_holm_all_66_comparisons"] = _holm_adjust(delong_all["p_value_delong"].to_numpy())
    delong_all["significant_holm_all_66_p_lt_0_05"] = delong_all["p_value_holm_all_66_comparisons"] < 0.05
    # Additional focused-family adjustments for the prespecified comparison questions.
    delong_all["p_value_holm_within_feature_set"] = np.nan
    for fs in FEATURE_SET_NAMES:
        mask = (delong_all["comparison_type"] == "same_feature_set_different_model") & (delong_all["feature_set_a"] == fs)
        if mask.any():
            delong_all.loc[mask, "p_value_holm_within_feature_set"] = _holm_adjust(delong_all.loc[mask, "p_value_delong"].to_numpy())
    delong_all["p_value_holm_within_model"] = np.nan
    for model_name in MODEL_NAMES_ORDER:
        mask = (delong_all["comparison_type"] == "same_model_different_feature_set") & (delong_all["model_a"] == model_name)
        if mask.any():
            delong_all.loc[mask, "p_value_holm_within_model"] = _holm_adjust(delong_all.loc[mask, "p_value_delong"].to_numpy())

    delong_all.to_csv(Path(OUTPUT_DIR) / "delong_pairwise_all_models.csv", index=False)
    within_feature_set = delong_all[delong_all["comparison_type"] == "same_feature_set_different_model"].copy()
    within_feature_set.to_csv(Path(OUTPUT_DIR) / "delong_comparisons_within_feature_set.csv", index=False)
    between_feature_sets = delong_all[delong_all["comparison_type"] == "same_model_different_feature_set"].copy()
    between_feature_sets.to_csv(Path(OUTPUT_DIR) / "delong_comparisons_between_feature_sets.csv", index=False)

    configuration_labels = [f"{fs} | {model_name}" for fs, model_name in configurations]
    difference_matrix = np.zeros((len(configurations), len(configurations)), dtype=float)
    for i, key_i in enumerate(configurations):
        for j, key_j in enumerate(configurations):
            difference_matrix[i, j] = auc_by_configuration[key_i] - auc_by_configuration[key_j]
    pd.DataFrame(difference_matrix, index=configuration_labels, columns=configuration_labels).to_csv(
        Path(OUTPUT_DIR) / "delong_auc_difference_matrix.csv", index_label="configuration"
    )
    _plot_delong_auc_difference_heatmap(
        configuration_labels, difference_matrix, FIG_DIR / "DeLong_AUC_difference_heatmap.png"
    )
    print(f"  Pairwise DeLong tests completed: {len(delong_all)} comparisons across {len(configurations)} configurations.")
    print("  Global Holm correction across all pairwise tests; focused Holm corrections also saved.")

    # Long-form table: one row per model x metric, with the exact CI bounds.
    long_rows = []
    for row in results.to_dict(orient="records"):
        for metric in METRIC_COLUMNS:
            long_rows.append({
                "feature_set": row["feature_set"], "model": row["model"],
                "metric": metric, "metric_label": METRIC_LABELS[metric],
                "estimate": row[metric], "ci_lower_95": row[f"{metric}_ci_lower"],
                "ci_upper_95": row[f"{metric}_ci_upper"],
                "estimate_percent": 100 * row[metric],
                "ci_lower_95_percent": 100 * row[f"{metric}_ci_lower"],
                "ci_upper_95_percent": 100 * row[f"{metric}_ci_upper"],
            })
    pd.DataFrame(long_rows).to_csv(Path(OUTPUT_DIR) / "metric_bootstrap_cis_long.csv", index=False)

    # Manuscript-style tables (percentages, with percentile CI in each cell).
    for fs in FEATURE_SET_NAMES:
        subset = results.loc[results.feature_set == fs].copy()
        table = pd.DataFrame({"Model": subset["model"].values})
        for metric, label in [
            ("sensitivity", "Sensitivity (%)"), ("specificity", "Specificity (%)"),
            ("ppv", "PPV (%)"), ("npv", "NPV (%)"), ("oof_acc", "Accuracy (%)"),
            ("oof_auc", "AUC (%)"), ("oof_pr_auc", "PR-AUC / AP (%)"),
            ("oof_f1", "F1 (%)"), ("balanced_accuracy", "Balanced accuracy (%)"),
        ]:
            table[label + " (95% CI)"] = subset.apply(lambda r: _ci_text(r, metric, percent=True), axis=1).values
        table.to_csv(Path(OUTPUT_DIR) / f"table_{_slug(fs)}_with_95CI.csv", index=False)
        print(f"\n{fs}: pooled OOF metrics (%; 95% stratified-bootstrap CI)")
        print(table.to_string(index=False))

    # Record cleaning decisions and run settings.
    pd.DataFrame({"dropped_column": data_info["dropped_columns"]}).to_csv(Path(OUTPUT_DIR) / "dropped_columns.csv", index=False)
    pd.DataFrame(data_info["conversion_report"]).to_csv(Path(OUTPUT_DIR) / "numeric_conversion_report.csv", index=False)

    # 2) Post-hoc interpretability pass. Refit same fold pipelines after all OOF
    # metrics are finalized; fold-level importances are then averaged across folds.
    print("\nStarting post-hoc back-mapped feature importance and MLP SHAP pass ...")
    importance_fold_rows = []
    mlp_shap_long_rows = []
    mlp_shap_pc_rows = []
    shap_error_rows = []

    for feature_set_name in FEATURE_SET_NAMES:
        X_set = feature_sets[feature_set_name]
        original_source_columns = list(X_set.columns)
        for model_name, base_model in _get_models().items():
            print(f"\n[Interpretability] {feature_set_name} | {model_name}")
            if model_name == "MLP" and not RUN_SHAP:
                print("  skipping MLP SHAP/refit because --skip-shap was set")
                continue
            for fold_no, (train_idx, val_idx) in enumerate(folds, start=1):
                X_train, X_val = X_set.iloc[train_idx], X_set.iloc[val_idx]
                y_train = y.iloc[train_idx]
                pipe = _build_pipeline(X_train, clone(base_model))
                pipe.fit(X_train, y_train)
                pre = pipe.named_steps["preprocess"]
                pca = pipe.named_steps["pca"]
                estimator = pipe.named_steps["model"]
                encoded_source_names = _encoded_to_source_feature_map(pipe, X_train)
                loading_weights = _component_loading_weights(pca)
                n_components = loading_weights.shape[0]

                if model_name != "MLP":
                    pc_importance = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
                    if len(pc_importance) != n_components:
                        raise RuntimeError(f"{model_name} importance dimension {len(pc_importance)} does not match PCA components {n_components}.")
                    encoded_importance = pc_importance @ loading_weights
                    source_imp = _aggregate_encoded_to_sources(encoded_importance.reshape(1, -1), encoded_source_names, original_source_columns).ravel()
                    source_imp = np.maximum(source_imp, 0.0)
                    total = float(source_imp.sum())
                    normalized_imp = source_imp / total if total > 0 else np.zeros_like(source_imp)
                    for feature, raw_imp, norm_imp in zip(original_source_columns, source_imp, normalized_imp):
                        importance_fold_rows.append({
                            "feature_set": feature_set_name, "model": model_name, "fold": fold_no,
                            "feature": str(feature), "importance": float(raw_imp),
                            "normalized_importance": float(norm_imp),
                            "importance_method": "native tree importance in PCA space, distributed by absolute PCA loading",
                        })
                elif RUN_SHAP:
                    # Explain validation-fold MLP probabilities using a background sampled
                    # only from the training fold; SHAP is computed in that fold's PCA space.
                    try:
                        X_train_pre = pre.transform(X_train)
                        X_val_pre = pre.transform(X_val)
                        X_train_pc = pca.transform(X_train_pre)
                        X_val_pc = pca.transform(X_val_pre)
                        n_background = min(SHAP_BACKGROUND_SIZE, len(X_train_pc))
                        rng = np.random.default_rng(RANDOM_STATE + fold_no * 100 + FEATURE_SET_NAMES.index(feature_set_name))
                        bg_idx = rng.choice(len(X_train_pc), size=n_background, replace=False)
                        background = np.asarray(X_train_pc[bg_idx], dtype=float)
                        model_fn = lambda z, _est=estimator: _est.predict_proba(np.asarray(z, dtype=float))[:, 1]
                        explainer = shap.KernelExplainer(model_fn, background, link="identity")
                        shap_pc = _get_binary_shap_matrix(explainer, np.asarray(X_val_pc, dtype=float), SHAP_NSAMPLES)
                        # Approximate projection: distribute each PC attribution across encoded
                        # variables proportional to abs(loadings), then sum one-hot categories.
                        shap_encoded = shap_pc @ loading_weights
                        shap_source = _aggregate_encoded_to_sources(shap_encoded, encoded_source_names, original_source_columns)
                        prob_val = estimator.predict_proba(X_val_pc)[:, 1]
                        for local_i, global_i in enumerate(val_idx):
                            for pc_i in range(n_components):
                                mlp_shap_pc_rows.append({
                                    "sample_index": int(global_i), "y_true": int(y.iloc[global_i]),
                                    "feature_set": feature_set_name, "fold": fold_no,
                                    "principal_component": f"PC{pc_i+1}",
                                    "shap_value_probability_scale": float(shap_pc[local_i, pc_i]),
                                })
                            for feat_i, feature in enumerate(original_source_columns):
                                mlp_shap_long_rows.append({
                                    "sample_index": int(global_i), "y_true": int(y.iloc[global_i]),
                                    "feature_set": feature_set_name, "fold": fold_no,
                                    "feature": str(feature),
                                    "shap_value": float(shap_source[local_i, feat_i]),
                                    "oof_probability_from_interpretability_refit": float(prob_val[local_i]),
                                    "attribution_note": "PCA-loading-based back-projection of MLP SHAP; approximate, not exact raw-feature SHAP",
                                })
                        # Mean absolute attribution per validation fold; average these fold-level
                        # global importances after all folds to avoid mixing PCs across folds.
                        source_imp = np.mean(np.abs(shap_source), axis=0)
                        total = float(source_imp.sum())
                        normalized_imp = source_imp / total if total > 0 else np.zeros_like(source_imp)
                        for feature, raw_imp, norm_imp in zip(original_source_columns, source_imp, normalized_imp):
                            importance_fold_rows.append({
                                "feature_set": feature_set_name, "model": model_name, "fold": fold_no,
                                "feature": str(feature), "importance": float(raw_imp),
                                "normalized_importance": float(norm_imp),
                                "importance_method": "mean absolute MLP SHAP in validation fold, back-projected via absolute PCA loadings",
                            })
                        print(f"  fold {fold_no}/{N_SPLITS}: SHAP complete ({len(val_idx)} validation cases, {n_components} PCs)")
                    except Exception as exc:
                        shap_error_rows.append({
                            "feature_set": feature_set_name, "model": model_name, "fold": fold_no,
                            "error": repr(exc),
                        })
                        print(f"  WARNING: SHAP failed for {feature_set_name}, fold {fold_no}: {exc}")

            if model_name != "MLP":
                print(f"  fold-averaged native importance projected to {len(original_source_columns)} source features.")

    importance_by_fold = pd.DataFrame(importance_fold_rows)
    importance_by_fold.to_csv(Path(OUTPUT_DIR) / "backmapped_feature_importance_by_fold.csv", index=False)
    if not importance_by_fold.empty:
        importance_mean = (
            importance_by_fold.groupby(["feature_set", "model", "feature", "importance_method"], as_index=False)
            .agg(mean_importance=("importance", "mean"), sd_importance=("importance", "std"),
                 mean_normalized_importance=("normalized_importance", "mean"),
                 sd_normalized_importance=("normalized_importance", "std"), n_folds=("fold", "nunique"))
        )
        importance_mean["sd_importance"] = importance_mean["sd_importance"].fillna(0.0)
        importance_mean["sd_normalized_importance"] = importance_mean["sd_normalized_importance"].fillna(0.0)
        importance_mean = importance_mean.sort_values(["feature_set", "model", "mean_normalized_importance"], ascending=[True, True, False])
        importance_mean.to_csv(Path(OUTPUT_DIR) / "backmapped_feature_importance_mean_across_folds.csv", index=False)

        # Top-feature figures for each feature set and classifier.
        for fs in FEATURE_SET_NAMES:
            for model_name in MODEL_NAMES_ORDER:
                d = importance_mean[(importance_mean.feature_set == fs) & (importance_mean.model == model_name)].nlargest(TOP_N_FEATURES, "mean_normalized_importance")
                if not d.empty:
                    _plot_importance(d, f"{fs} — {model_name}: top back-mapped features", FIG_DIR / f"FI_{_slug(fs)}_{_slug(model_name)}_top{TOP_N_FEATURES}.png")

    if mlp_shap_long_rows:
        mlp_shap_df = pd.DataFrame(mlp_shap_long_rows)
        mlp_shap_df.to_csv(Path(OUTPUT_DIR) / "mlp_shap_backmapped_oof_attributions.csv", index=False)
        pd.DataFrame(mlp_shap_pc_rows).to_csv(Path(OUTPUT_DIR) / "mlp_shap_pca_space_oof_attributions.csv", index=False)
        for fs in FEATURE_SET_NAMES:
            d = mlp_shap_df[mlp_shap_df.feature_set == fs].copy()
            if not d.empty:
                _plot_shap_beeswarm(
                    d,
                    f"{fs} MLP — back-mapped SHAP summary (approximate PCA projection)",
                    FIG_DIR / f"SHAP_MLP_{_slug(fs)}_backmapped_summary.png",
                )
    else:
        pd.DataFrame(columns=["sample_index", "y_true", "feature_set", "fold", "feature", "shap_value"]).to_csv(
            Path(OUTPUT_DIR) / "mlp_shap_backmapped_oof_attributions.csv", index=False)
    if shap_error_rows:
        pd.DataFrame(shap_error_rows).to_csv(Path(OUTPUT_DIR) / "shap_errors.csv", index=False)
        print(f"WARNING: SHAP encountered {len(shap_error_rows)} fold-level errors; details are in shap_errors.csv")

    # Metadata for reproducibility and interpreting the uncertainty estimates.
    metadata = {
        "pipeline": "ARaclepius/tbi-hemorrhage-expansion-radiomics; single-cell Colab adaptation",
        "dataset_path": str(CSV_PATH), "output_dir": str(OUTPUT_DIR),
        "random_state": RANDOM_STATE, "n_splits": N_SPLITS, "pca_variance": PCA_VARIANCE,
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_ci": "stratified percentile 95% CI over pooled OOF (y_true, probability) pairs",
        "classification_threshold": 0.5,
        "feature_sets": {name: int(df.shape[1]) for name, df in feature_sets.items()},
        "n_patients": int(data_info["n_patients"]), "class_counts": data_info["class_counts"],
        "n_rows_removed_missing_target": data_info["n_rows_removed_missing_target"],
        "target_column_used": data_info["target_column_used"],
        "dropped_columns": data_info["dropped_columns"],
        "conversion_report": data_info["conversion_report"],
        "pca_components_per_fold": fold_pca_counts,
        "models": {
            "Random Forest": {"n_estimators": 100},
            "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 0.8, "colsample_bytree": 0.8},
            "MLP": {"hidden_layer_sizes": [100], "max_iter": 1000},
        },
        "statistical_comparisons": {
            "method": "Paired DeLong test for correlated ROC AUCs using pooled out-of-fold probabilities",
            "scope": "All 66 pairwise comparisons among 12 model x feature-set configurations",
            "multiple_testing": "Holm family-wise error correction across all 66 comparisons, plus focused Holm corrections within each feature set (model comparisons) and within each model (feature-set comparisons)",
            "alpha": 0.05,
            "caveat": "Standard DeLong p-values are approximate for pooled OOF predictions because fitted models across CV folds use overlapping training sets; interpret as internal, exploratory comparisons, not external validation.",
        },
        "interpretability": {
            "shap_enabled": bool(RUN_SHAP),
            "tree_models": "native PCA-space feature_importances_ allocated to source features by absolute PCA-loading proportions; then grouped over one-hot categories",
            "mlp": "Kernel SHAP in PCA space on each held-out fold; attribution is approximately back-projected to source variables with absolute PCA loadings",
            "shap_kernel_l1_reg": "num_features(15)",
            "warning": "Back-projected attributions are approximate and should not be described as exact raw-feature SHAP values.",
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "imbalanced_learn": imblearn.__version__,
            "xgboost": xgboost.__version__, "shap": shap.__version__,
        },
        "runtime_seconds": round(time.time() - start_time, 2),
    }
    with open(Path(OUTPUT_DIR) / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print(f"Patients: {len(y)} | feature/model combinations: {len(results)} | stratified bootstrap replicates/model: {N_BOOTSTRAP}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Key files:")
    for filename in [
        "results_pca95_with_bootstrap_ci.csv", "metric_bootstrap_cis_long.csv",
        "pooled_oof_predictions.csv", "delong_pairwise_all_models.csv",
        "delong_comparisons_within_feature_set.csv", "delong_comparisons_between_feature_sets.csv",
        "delong_auc_difference_matrix.csv", "backmapped_feature_importance_mean_across_folds.csv",
        "backmapped_feature_importance_by_fold.csv", "mlp_shap_backmapped_oof_attributions.csv",
        "run_metadata.json",
    ]:
        full_path = Path(OUTPUT_DIR) / filename
        if full_path.exists():
            print("  -", full_path)
    print("Main figures:")
    for filename in ["AUC_95CI_summary.png", "DeLong_AUC_difference_heatmap.png", "ROC_radiomics.png", "ROC_clinical.png", "ROC_combined.png"]:
        full_path = FIG_DIR / filename
        if full_path.exists():
            print("  -", full_path)
    print(f"Elapsed: {time.time() - start_time:.1f} seconds")
    print("Reminder: internal cross-validation only; CIs quantify resampling uncertainty conditional on pooled OOF predictions, not external validity.")
    print("DeLong caveat: tests compare correlated pooled OOF ROC curves; overlapping CV training sets make p-values approximate.")



def main(argv=None):
    """Command-line entry point for the confidence-interval/interpretability pipeline."""
    parser = argparse.ArgumentParser(
        description="TBI hemorrhage-expansion radiomics: pooled OOF metrics, bootstrap CIs, DeLong tests and interpretability."
    )
    parser.add_argument("--data", default="data/TBI_final_corrected.csv", help="Input CSV; must contain a 'rebleeding' target column.")
    parser.add_argument("--output", default="results", help="Directory for CSV tables, figures and run metadata.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=2000, help="Stratified percentile-bootstrap replicates for 95%% CIs.")
    parser.add_argument("--shap-nsamples", type=int, default=100, help="Kernel SHAP approximation budget per validation case.")
    parser.add_argument("--shap-background-size", type=int, default=10, help="Training-fold cases used as SHAP background.")
    parser.add_argument("--top-n-features", type=int, default=20)
    parser.add_argument("--skip-shap", action="store_true", help="Skip MLP SHAP (faster smoke tests); tree-model feature importances are still computed.")
    args = parser.parse_args(argv)
    if args.n_splits < 2:
        parser.error("--n-splits must be at least 2")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if not 0 < args.pca_variance < 1:
        parser.error("--pca-variance must be strictly between 0 and 1")
    if args.shap_nsamples < 1 or args.shap_background_size < 1 or args.top_n_features < 1:
        parser.error("SHAP sample/background sizes and top-n-features must be positive")

    global CSV_PATH, OUTPUT_DIR, RANDOM_STATE, N_SPLITS, PCA_VARIANCE
    global N_BOOTSTRAP, SHAP_NSAMPLES, SHAP_BACKGROUND_SIZE, TOP_N_FEATURES, RUN_SHAP
    CSV_PATH = str(args.data)
    OUTPUT_DIR = str(args.output)
    RANDOM_STATE = args.random_state
    N_SPLITS = args.n_splits
    PCA_VARIANCE = args.pca_variance
    N_BOOTSTRAP = args.bootstrap
    SHAP_NSAMPLES = args.shap_nsamples
    SHAP_BACKGROUND_SIZE = args.shap_background_size
    TOP_N_FEATURES = args.top_n_features
    RUN_SHAP = not args.skip_shap
    _run()


if __name__ == "__main__":
    main()
