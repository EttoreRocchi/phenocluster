Changelog
=========

v0.1.1 (2026/03/12)
--------------------

Refactored
~~~~~~~~~~

- Split large modules (``config.py``, ``pipeline.py``, ``evaluation/data_quality.py``) into organized subpackages.
- Reorganized visualization into domain-specific visualizers with a unified ``Visualizer`` interface.
- Improved structure and readability of multistate analysis, stability analysis, grid search, and CLI validation code.
- Removed decorative separator comments from source and test files.

Tests
~~~~~

- Reorganized tests to mirror the project structure and added shared fixtures.
- Expanded test coverage.

Fixed
~~~~~

- Minor fixes in encoder, preprocessing, external validation, and feature selection discovered during testing.


v0.1.0 (2026/03/04)
--------------------

Initial release.

- LCA/LPA via StepMix with automatic model selection and configurable cluster-size constraints
- Train/test split before preprocessing; full-cohort refit after selection
- Outcome analysis (logistic regression, chi-square/Fisher), survival analysis (Cox PH, log-rank), and multistate modelling
- Consensus clustering stability and internal validation
- Feature characterisation with effect sizes (Cohen's d, Cramer's V)
- Interactive HTML report, forest plots, KM/NA curves, state occupation plots
