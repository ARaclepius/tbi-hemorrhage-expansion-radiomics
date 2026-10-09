# Reference outputs

`results_pca95.csv`, the original tables, confusion matrices and supplementary statistics retain the baseline results that shipped with v1.x.

The enhanced pipeline's aggregate outputs from the supplied 86-patient run are in [`ci_interpretability/`](ci_interpretability/). They include pooled OOF point estimates with stratified-bootstrap 95% CIs, DeLong comparisons with Holm correction, feature-importance aggregates and generated figures. Per-patient OOF predictions and case-level SHAP CSVs are deliberately not included here.

The confidence intervals are conditional on the pooled OOF prediction pairs. The paired DeLong analyses are exploratory because cross-validation training sets overlap. See the top-level README for methods, interpretability caveats and instructions for reproducing the outputs.
