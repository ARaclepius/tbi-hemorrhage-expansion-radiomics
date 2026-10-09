# Changelog

## 1.2.0 — confidence intervals, interpretable ML, and continuous integration

This release expands the existing PCA-based TBI hemorrhage-expansion prediction pipeline while retaining its three feature sets and four classifier families.

- Added stratified-bootstrap 95% confidence intervals for nine metrics evaluated using pooled out-of-fold predictions.
- Added paired DeLong AUC comparisons across all 12 model/feature-set configurations and Holm-adjusted p-values.
- Added fold-wise tree-model feature importance mapped approximately from PCA components to source features, with fold-averaged summaries.
- Added fold-specific MLP Kernel SHAP in PCA space and an explicitly approximate back-projection to source-feature names.
- Expanded run metadata and command-line options for data/output paths, cross-validation, PCA, bootstrap, and SHAP settings.
- Retained the one-cell Google Colab notebook and added a local CLI entry point.
- Added unit tests for bootstrap intervals, DeLong calculations, Holm correction, PCA-loading feature mapping, and Kernel SHAP helper behavior.
- Extended GitHub Actions to run tests and an end-to-end synthetic-data smoke workflow.
- Added aggregate reference outputs and updated documentation, figures, and methodological caveats.
- Preserved the existing DOI: https://doi.org/10.5281/zenodo.23166853

## 1.0.1 and earlier

Earlier PCA-based pipeline with three feature sets, four classifiers, stratified five-fold cross-validation, and pooled out-of-fold metrics.
