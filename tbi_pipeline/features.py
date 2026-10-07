"""Split the cleaned matrix into the three feature sets."""
from . import config as C


def is_radiomics_feature(column):
    name = str(column).lower()
    if any(tag in name for tag in ("original__", "wavelet__", "original_", "wavelet_")):
        return True
    return any(f"__{c}__" in name or f"_{c}_" in name for c in C.RADIOMICS_CLASSES)


def build_feature_sets(X, verbose=True):
    radiomics = [c for c in X.columns if is_radiomics_feature(c)]
    clinical = [c for c in X.columns if c not in radiomics]
    sets = {
        "Radiomics": X[radiomics].copy(),
        "Clinical": X[clinical].copy(),
        "Combined": X.copy(),
    }
    if verbose:
        for name, Xm in sets.items():
            print(f"  {name:10s}: {Xm.shape[0]} patients x {Xm.shape[1]} features")
        if len(radiomics) != C.EXPECTED_N_RADIOMICS:
            print(f"WARNING: detected {len(radiomics)} radiomics features; "
                  f"analysis expects {C.EXPECTED_N_RADIOMICS}.")
    return sets
