# Data availability

The clinical dataset (`TBI_final.csv`, n = 86 patients: 40 expansion, 46 no expansion) contains patient-level
information and **is not distributed in this repository**
(ethics approval / privacy restrictions).

Available from the corresponding author on reasonable request, subject to
institutional approval (Hamadan University of Medical Sciences).

## Expected format

One row per patient (radiomic and clinical tables already joined on the patient identifier), UTF-8 CSV, with:

| Column type | Description |
|---|---|
| `rebleeding` | Target. 0/1 (or yes/no). Required. |
| Radiomics (n = 107) | PyRadiomics *original* features, e.g. `original__shape__Elongation`, `original__firstorder__10Percentile` (14 shape, 18 first-order, 24 GLCM, 16 GLRLM, 16 GLSZM, 14 GLDM, 5 NGTDM) |
| Clinical / lab / demographic | Any other baseline column (numeric or categorical) |
| Excluded automatically | Patient identifiers, timestamps, surgery, outcome, ICU stay, mortality, control-CT / follow-up variables (see `tbi_pipeline/config.py`) |

Numbers stored as text with thousands separators (`"13,342"`) are converted
automatically.

## Synthetic example

`python scripts/make_synthetic_data.py` creates `data/synthetic_example.csv`
(no real patient information) so the pipeline can be run and tested end to end.
Metrics on it are meaningless.

## Cleaning rules applied by the code

- Columns containing outcome or post-admission information (surgery, in-hospital
  outcome, ICU stay, follow-up/control CT, timestamps, identifiers) are excluded
  from all predictor matrices.
- Text columns that are >= 80% numeric are converted; entries that cannot be
  parsed (e.g. `5..84`) become missing and are median-imputed inside each
  training fold. Repair such entries upstream if you want them recovered.
