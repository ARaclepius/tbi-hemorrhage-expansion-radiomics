# Reference results

Fixed outputs of the final analysis (PCA 95% models, 5-fold stratified CV, n = 86).
Aggregate numbers only - no patient-level data.

## Model results (main content)

| Path | Content |
|---|---|
| `results_pca95.csv` | All 12 models (3 feature sets × 4 classifiers): AUC, PR-AUC, accuracy, F1, sensitivity, specificity, PPV, NPV, balanced accuracy, confusion-matrix counts, mean PCA components |
| `tables/table_radiomics.csv` · `table_clinical.csv` · `table_combined.csv` | Manuscript-style performance tables (percent) |
| `figures/CM_*.png` | The 12 pooled out-of-fold confusion matrices |

`python run_pipeline.py` regenerates the same files in `results/`, including ROC curves.
To check that your run matches this reference:

```bash
python scripts/compare_with_reference.py results/results_pca95.csv
```

## Supplementary (secondary)

`supplementary_statistics/` holds the univariate baseline and chi-square tables of the
clinical study. They are provided for transparency only; the code in this repository
covers the machine-learning analysis, not these tests.
