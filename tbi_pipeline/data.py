"""Loading, target normalisation and column cleaning."""
import re
import numpy as np
import pandas as pd

from . import config as C


def load_raw(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    unnamed = [c for c in df.columns if str(c).strip().lower().startswith("unnamed:")]
    return df.drop(columns=unnamed, errors="ignore")


def find_target_column(df, target=C.TARGET_COLUMN):
    for col in df.columns:
        if str(col).strip().lower() == target:
            return col
    raise ValueError(f"Could not find target column '{target}'.")


def normalize_target(value):
    """Map heterogeneous labels to {0, 1, NaN}.

    NOTE: free-text labels that start with 'until' / 'between' / 'after'
    (timing descriptions) and any other non-empty unrecognised label are
    treated as positive (expansion), as in the manuscript analysis.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 0 if float(value) == 0 else 1
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    if text in C.NEGATIVE_LABELS:
        return 0
    if text in C.POSITIVE_LABELS:
        return 1
    if text.startswith(("until", "between", "after")):
        return 1
    return 1 if text != "" else np.nan


def should_drop_column(column):
    name = str(column).strip()
    if name.lower() in C.EXACT_EXCLUSIONS:
        return True
    return any(re.search(p, name, flags=re.IGNORECASE) for p in C.REGEX_EXCLUSIONS)


def clean_numeric_like_columns(X, threshold=0.80):
    """Convert string columns that are >=80% numeric ('13,342' -> 13342).

    Unparseable entries in converted columns become NaN (and are imputed later).
    """
    X = X.copy()
    report = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        raw = X[col]
        cleaned = raw.astype("string").str.strip().str.replace(",", "", regex=False)
        converted = pd.to_numeric(cleaned, errors="coerce")
        non_missing = raw.notna().sum()
        if non_missing == 0:
            continue
        rate = converted.notna().sum() / non_missing
        if rate >= threshold:
            invalid = int((raw.notna() & converted.isna()).sum())
            X[col] = converted.astype(float)
            report.append({"column": col, "type": "numeric",
                           "conversion_rate": float(rate), "invalid_to_nan": invalid})
        else:
            report.append({"column": col, "type": "categorical",
                           "conversion_rate": float(rate), "invalid_to_nan": 0})
    return X, report


def prepare_dataset(path, verbose=True):
    """Return (X, y, info) ready for feature-set splitting."""
    df = load_raw(path)
    target_col = find_target_column(df)
    y = df[target_col].map(normalize_target)

    missing = y.isna()
    if missing.any():
        if verbose:
            print(f"Rows with missing target removed: {int(missing.sum())}")
        df = df.loc[~missing].reset_index(drop=True)
        y = y.loc[~missing].reset_index(drop=True)
    y = y.astype(int)

    X = df.drop(columns=[target_col]).copy()
    dropped = [c for c in X.columns if should_drop_column(c)]
    X = X.drop(columns=dropped)

    all_nan = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=all_nan)
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant)
    dropped = list(dict.fromkeys(dropped + all_nan + constant))

    X, conversion_report = clean_numeric_like_columns(X)

    info = {
        "n_patients": int(len(X)),
        "class_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "dropped_columns": dropped,
        "conversion_report": conversion_report,
    }
    if verbose:
        print(f"Patients: {info['n_patients']} | class counts: {info['class_counts']}")
        print(f"Expected cohort: {C.EXPECTED_CLASS_COUNTS}")
        print(f"Dropped {len(dropped)} columns; X shape: {X.shape}")
        for r in conversion_report:
            if r["type"] == "numeric" and r["invalid_to_nan"] > 0:
                print(f"  invalid numeric -> NaN: {r['column']} ({r['invalid_to_nan']})")
    return X, y, info
