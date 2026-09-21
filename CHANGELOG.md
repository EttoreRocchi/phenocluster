# Changelog

All notable changes to phenocluster will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-21

### Fixed

- Phenotype renumbering by size is now applied to the fitted model, not only to the label array: the final model is wrapped in `phenocluster.core.PhenotypeOrderedModel` so `predict` and `predict_proba` share the label space of the derivation labels.
- Categorical labels are dtype-stable across cohorts via `phenocluster.data.encoder.canonical_labels`, applied on the label, one-hot, and frequency paths at fit and transform.

### Added

- Per-patient phenotype assignments for validation cohorts in `data/generalizability/phenotypes_<label>.csv`, also exposed as `CohortReport.assignments`.
- Validation cohort schema check via `phenocluster.evaluation.generalizability.check_cohort_schema`, reported in `CohortReport.schema_check` and `CohortReport.warnings`.

## [0.3.0] - 2026-05-07

### Added

- Temporal and multi-site generalizability stage: `GeneralizationStage`, pluggable splitters in `phenocluster.data.splitting`, and `GeneralizabilityEvaluator` with apply-only and refit-and-match modes.
- Per-split derivation refit via `generalizability.training_scope` (default `"per_split"`).
- Per-split feature-selector refit via `generalizability.feature_selector_scope` (default `"auto"`).
- External cohort CSVs via `generalizability.external_cohorts`, with `results/external_cohorts_results.json`.
- Outcome-target collision detection via `feature_selection.error_on_outcome_collision`.
- Calibration, drift, and concordance metrics: Brier / ECE / reliability curves, PSI / KS / chi-square drift, Pearson, Spearman, Lin's CCC, and per-phenotype Wald delta tests with BH-FDR.
- Streamlit dashboard via `phenocluster dashboard <results_dir>`; install with `pip install 'phenocluster[dashboard]'`.
- HTML report toggle: `generate_html_report` config flag and `--html-report / --no-html-report`.

### Changed

- `DataSplitConfig` and `DataSplitResult` extended for temporal and group-based splitting.
- `DataSplitter` aliased to `RandomSplitter`.
- `ExternalValidator` cohort comparison lifted into `phenocluster.evaluation.generalizability.prevalence`.

## [0.2.0] - 2026-04-10

### Added

- `cli.py` split into a `phenocluster.cli` package.
- New commands `phenocluster list-profiles` and `phenocluster show-profile <name>`.
- Grambsch-Therneau global Schoenfeld test in `SurvivalAnalyzer`.
- Little's MCAR test via a proper EM loop.
- Stratification provenance on `DataSplitResult`.

### Changed

- `FeatureCharacterizer` reports Hedges' g* instead of pooled Cohen's d.
- Multistate `n_sims` is the total simulation budget, divided across patients.

### Fixed

- `OutcomeAnalyzer` no longer raises on non-float outcome columns.

## [0.1.1] - 2026-03-12

### Changed

- Split `config.py`, `pipeline.py`, and `evaluation/data_quality.py` into subpackages.
- Reorganized visualization into domain-specific visualizers with a unified `Visualizer` interface.
- Tests reorganized to mirror the project structure; coverage expanded.

### Fixed

- Minor fixes in encoder, preprocessing, external validation, and feature selection.

## [0.1.0] - 2025-12-01

### Added

- Initial release.

