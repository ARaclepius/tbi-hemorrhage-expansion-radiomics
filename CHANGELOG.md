# Changelog

## 2.0.0 — enhanced confidence intervals and interpretability

- Added stratified-bootstrap 95% confidence intervals for nine pooled OOF metrics.
- Added paired DeLong AUC comparisons across all 12 model/feature-set configurations and Holm-adjusted p-values.
- Added fold-wise, PCA-loading-based back-mapping of tree-model importance and fold-averaged source-feature summaries.
- Added fold-specific MLP Kernel SHAP in PCA space and explicitly labelled approximate source-feature attribution.
- Replaced the default entry point with the expanded analysis runner and added CLI options for data/output paths, CV, PCA, bootstrap and SHAP settings.
- Kept the provided one-cell Colab notebook in `notebooks/`.
- Added unit tests for uncertainty, AUC comparison, multiple-testing correction and loading-based feature mapping.
- Extended GitHub Actions to run tests and an end-to-end synthetic-data smoke workflow.
- Added aggregate reference outputs and updated documentation, figures and method caveats.

## 1.x

Earlier PCA-based pipeline with three feature sets, four classifiers, stratified five-fold CV and pooled OOF metrics.
