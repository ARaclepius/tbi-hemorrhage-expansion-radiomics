"""Model zoo and leakage-safe pipeline: preprocess -> oversample -> PCA -> classifier.

Every step is fitted on the training fold only; the validation fold is merely
transformed and scored. It never influences imputation, scaling, one-hot
encoding, oversampling or PCA.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config as C


def _onehot_kwargs():
    try:
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return {"handle_unknown": "ignore", "sparse_output": False}
    except TypeError:  # scikit-learn < 1.2
        return {"handle_unknown": "ignore", "sparse": False}


def get_models(seed=C.RANDOM_STATE):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic",
            eval_metric="logloss", random_state=seed, n_jobs=-1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=1000, random_state=seed),
    }


def build_pipeline(X_train, classifier, seed=C.RANDOM_STATE,
                   pca_variance=C.PCA_VARIANCE):
    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**_onehot_kwargs())),
        ("scaler", StandardScaler()),
    ])
    preprocess = ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)],
        remainder="drop")

    return Pipeline([
        ("preprocess", preprocess),
        ("sampler", RandomOverSampler(random_state=seed)),
        ("pca", PCA(n_components=pca_variance, svd_solver="full", random_state=seed)),
        ("model", classifier),
    ])
