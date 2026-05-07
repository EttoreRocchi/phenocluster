# Changelog

## [0.3.0] - 2026-05-07

### Added

- **Temporal and multi-site generalizability stage.** New `GeneralizationStage` runs after the main pipeline, building validation cohorts via pluggable splitters (`RandomSplitter`, `TemporalSplitter`, `HoldoutGroupSplitter`, `LeaveOneGroupOutSplitter`) under `phenocluster.data.splitting`. `GeneralizabilityEvaluator` supports apply-only and refit-and-match modes, with Hungarian alignment for unequal cluster counts and ARI / NMI / matched accuracy.
- **Per-split derivation refit (default).** `generalizability.training_scope: "per_split" | "global"` (default `"per_split"`) fits a fresh `DataPreprocessor` and StepMix on derivation rows only for each in-CSV split, leaving the full-cohort model untouched. Each cohort report carries `fit_mode` and `derivation_only_ari`. Per-split derivation labels also feed a `derivation_only_outcomes` block used in cross-cohort concordance.
- **Per-split feature-selector refit.** `generalizability.feature_selector_scope: "auto" | "global" | "per_split"` (default `"auto"`) refits the selector on derivation rows when safe (unsupervised methods, or supervised LASSO whose target is not also a concordance outcome). Cohort reports carry `feature_selector_mode`.
- **External cohort CSVs.** `generalizability.external_cohorts: [{path, label, kind}]` scores one or more separate CSVs through the global model and routes them into the temporal / multi-site / external bucket. New `results/external_cohorts_results.json`.
- **Outcome-target collision detection.** `feature_selection.error_on_outcome_collision` (default `false`) warns at config-validation time when `feature_selection.target_column` overlaps `outcome.outcome_columns` or a survival `time_column` / `event_column`; setting it to `true` promotes the warning to a hard `ValueError`.
- **Calibration, drift, and concordance metrics.** Brier / ECE / reliability curves (refit-and-match only); per-feature PSI, KS, chi-square drift table with `top_drifted` helper; Pearson, Spearman, Lin's CCC, sign agreement, and per-phenotype Wald delta tests with BH-FDR correction.
- **Streamlit dashboard.** New `phenocluster dashboard <results_dir>` CLI command for interactively exploring saved outputs. Optional dependency; install with `pip install 'phenocluster[dashboard]'`.
- **HTML report toggle.** `generate_html_report` config flag and `--html-report / --no-html-report` CLI flag on `phenocluster run`.

### Changed

- **`DataSplitConfig` and `DataSplitResult` extended** to support temporal and group-based splitting strategies; defaults preserve v0.2.0 random-split behavior.
- **`DataSplitter` aliased to `RandomSplitter`** so existing imports keep working.
- **`ExternalValidator` refactor** - cluster-distribution and chi-square cohort comparison lifted into `phenocluster.evaluation.generalizability.prevalence` as a single source of truth.


## [0.2.0] - 2026-04-10

### Added

- **CLI refactor** - Split the `cli.py` file into a `phenocluster.cli` package.
- **New CLI commands and flags** - `phenocluster list-profiles` and `phenocluster show-profile <name>`.
- **Grambsch-Therneau global Schoenfeld test** - `SurvivalAnalyzer` now reports a global chi-squared alongside per-covariate raw Schoenfeld p-values.
- **Little's MCAR test via EM** - `littles_mcar_test` now estimates MVN parameters under MAR with a proper EM loop.
- **Stratification provenance** - `DataSplitResult` exposes `stratification_used` and `stratification_fallback_reason` so downstream reports can tell whether a requested stratified split actually happened.

### Changed

- **Effect sizes with Hedges' g*** - `FeatureCharacterizer` replaces pooled Cohen's d with Hedges' g* using the small-sample correction. New result fields: `effect_size_metric`, `rest_std`, `average_std`, `welch_df`.
- **Multistate simulation budget** - `n_sims` is treated as the total simulation budget for the phenotype and divided across patients with ceiling division, instead of being multiplied by the cohort size.

### Fixed

- `OutcomeAnalyzer` no longer raises on non-float outcome columns.


## [0.1.1] - 2026-03-12

### Refactored

- Split large modules (`config.py`, `pipeline.py`, `evaluation/data_quality.py`) into organized subpackages.
- Reorganized visualization into domain-specific visualizers with a unified `Visualizer` interface.
- Improved structure and readability of multistate analysis, stability analysis, grid search, and CLI validation code.
- Removed decorative separator comments from source and test files.

### Tests

- Reorganized tests to mirror the project structure and added shared fixtures.
- Expanded test coverage

### Fixed

* Minor fixes in encoder, preprocessing, external validation, and feature selection discovered during testing.


## [0.1.0] - 2025-12-01

- Initial release
