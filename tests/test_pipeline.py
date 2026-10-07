import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_synthetic_data import make
from tbi_pipeline.data import normalize_target, should_drop_column, prepare_dataset
from tbi_pipeline.features import build_feature_sets, is_radiomics_feature


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "syn.csv"
    make().to_csv(p, index=False)
    return p


def test_normalize_target():
    assert normalize_target(0) == 0 and normalize_target(1) == 1
    assert normalize_target("No") == 0 and normalize_target("Yes") == 1
    assert normalize_target("until 24h") == 1
    assert np.isnan(normalize_target(np.nan))


def test_exclusions():
    for c in ["Surgery", "Outcome", "Patient", "Length of hospitalization(days) in ICU"]:
        assert should_drop_column(c)
    assert not should_drop_column("Age")


def test_prepare_and_feature_sets(csv):
    X, y, info = prepare_dataset(csv, verbose=False)
    assert info["class_counts"] == {0: 46, 1: 40}
    assert "Surgery" not in X.columns and "Patient" not in X.columns
    assert pd.api.types.is_numeric_dtype(X["Platelets"])
    fs = build_feature_sets(X, verbose=False)
    assert fs["Radiomics"].shape[1] + fs["Clinical"].shape[1] == fs["Combined"].shape[1]


def test_end_to_end_one_model(csv):
    pytest.importorskip("imblearn"); pytest.importorskip("xgboost")
    from sklearn.ensemble import RandomForestClassifier
    from tbi_pipeline.evaluation import cross_validate_oof, make_folds
    X, y, _ = prepare_dataset(csv, verbose=False)
    m, _, pcs, prob = cross_validate_oof(
        build_feature_sets(X, verbose=False)["Combined"], y,
        RandomForestClassifier(n_estimators=20, random_state=0), make_folds(X, y))
    assert 0 <= m["oof_auc"] <= 1 and len(pcs) == 5 and len(prob) == len(y)
