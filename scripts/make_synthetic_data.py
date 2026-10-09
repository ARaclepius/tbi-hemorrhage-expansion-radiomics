"""Generate a SYNTHETIC dataset with the same schema style as the real data.

For smoke tests / CI / reviewers only. Contains no patient information and
carries no clinical signal beyond weak random noise.
"""
import argparse
import numpy as np
import pandas as pd


def make(n=86, n_pos=40, n_radiomics=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * (n - n_pos)); rng.shuffle(y)
    df = pd.DataFrame({"Patient": [f"P{i:03d}" for i in range(n)]})
    df["Age"] = rng.integers(18, 90, n)
    df["Sex"] = rng.choice(["M", "F"], n)
    df["GCS"] = rng.integers(3, 16, n)
    df["INR"] = rng.normal(1.1, 0.2, n).round(2)
    df["Platelets"] = [f"{v:,}" for v in rng.integers(100000, 400000, n)]  # '13,342'-style strings
    df.loc[rng.choice(n, 5, replace=False), "INR"] = np.nan
    classes = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"]
    for i in range(n_radiomics):
        c = classes[i % len(classes)]
        df[f"original_{c}_feature{i}"] = rng.normal(0, 1, n) + 0.3 * y * (i < 5)
    # columns that must be excluded
    df["Surgery"] = rng.choice([0, 1], n)
    df["Outcome"] = rng.choice([0, 1], n)
    df["Length of hospitalization(days) in ICU"] = rng.integers(1, 30, n)
    df["rebleeding"] = y
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_example.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make(seed=a.seed).to_csv(a.out, index=False)
    print("wrote", a.out)
