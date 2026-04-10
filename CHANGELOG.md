# Changelog

## [0.2.0] - 2026-04-10

### Added

- **CLI refactor** - Split the `cli.py` file into a `phenocluster.cli` package.
- **New CLI commands and flags** - `phenocluster list-profiles` and `phenocluster show-profile <name>`.
- **Grambsch-Therneau global Schoenfeld test** - `SurvivalAnalyzer` now reports a global χ² alongside per-covariate raw Schoenfeld p-values.
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
