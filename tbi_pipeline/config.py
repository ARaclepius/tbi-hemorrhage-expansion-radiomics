"""Central configuration. Every constant that affects results lives here."""

RANDOM_STATE = 42
N_SPLITS = 5            # Stratified K-fold
PCA_VARIANCE = 0.95     # Fraction of variance retained by PCA (fitted inside each training fold)
TARGET_COLUMN = "rebleeding"

# Analysed cohort (used only for a sanity-check message)
EXPECTED_CLASS_COUNTS = {0: 46, 1: 40}   # 46 no expansion / 40 expansion
EXPECTED_N_RADIOMICS = 107   # PyRadiomics "original" features (14 shape, 18 first-order, 75 texture)

FEATURE_SET_NAMES = ["Radiomics", "Clinical", "Combined"]

# Columns removed because they are identifiers, outcomes, post-baseline
# information, or would leak the target.
EXACT_EXCLUSIONS = {
    "surgery", "outcome",
    "lenght of hospitalization(days) in icu",
    "length of hospitalization(days) in icu",
    "icu length of stay",
    "rebleeding on control ct scans",
    "timestamp", "patient", "patient id", "patient_id", "patient_name",
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

NEGATIVE_LABELS = {"0", "no", "n", "negative", "no rebleeding", "none",
                   "absent", "no expansion", "false"}
POSITIVE_LABELS = {"1", "yes", "y", "positive", "rebleeding", "expansion",
                   "hemorrhage expansion", "present", "true"}
