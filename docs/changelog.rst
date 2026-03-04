Changelog
=========

v0.1.0 (2026/03/04)
--------------

Initial release.

- LCA/LPA via StepMix with automatic model selection and configurable cluster-size constraints
- Train/test split before preprocessing; full-cohort refit after selection
- Outcome analysis (logistic regression, chi-square/Fisher), survival analysis (Cox PH, log-rank), and multistate modelling (transition HRs, Monte Carlo state occupation)
- FDR correction (Benjamini-Hochberg) across all comparisons
- Consensus clustering stability and internal validation
- Feature characterisation with effect sizes (Cohen's d, Cramer's V)
- Interactive HTML report, forest plots, KM/NA curves, state occupation plots
- Colorblind-safe visualisations, CLI with config profiles, artifact caching
