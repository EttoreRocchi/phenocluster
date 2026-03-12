# Changelog

## [0.1.1] - 2026-03-12

### Refactored

- Split monolithic `config.py` into `config/` subpackage (`base`, `data`, `model`, `analysis`, `multistate`, `output`)
- Split monolithic `pipeline.py` into `pipeline/` subpackage (`orchestrator`, `context`, `quality`, `warnings`, `stages/`)
- Split `evaluation/data_quality.py` into `evaluation/data_quality/` subpackage (`assessor`, `figures`, `mcar`)
- Reorganized `visualization/plots.py` — extracted per-domain visualizers (`_cluster_distribution`, `_cluster_heatmap`, `_cluster_quality`, `_survival`, `_outcome`, `_multistate`) with a unified `Visualizer` facade
- Refactored `evaluation/multistate/analyzer.py` for clearer separation of concerns
- Refactored `evaluation/stability.py` — extracted standalone functions for parallel iteration
- Refactored `model_selection/grid_search.py` for improved readability
- Cleaned up `cli.py` — extracted validation helpers (`_validate_structure`, `_validate_multistate_structure`, `_validate_against_data`)
- Removed decorative separator comments (`# ---`, `# ===`) from source and test files

### Tests

- Reorganized flat `tests/` directory into subdirectories mirroring source package structure (`tests/cli/`, `tests/data/`, `tests/evaluation/`, `tests/feature_selection/`, `tests/pipeline/`, `tests/utils/`, `tests/visualization/`)
- Added shared test fixtures in `tests/conftest.py` (`minimal_config`, `sample_dataframe`, `sample_labels`, `config_with_survival`, `config_with_multistate`)
- Extended test coverage from 48% to 68% (90 → 241 tests) with new test files:
  - `test_cli.py` — CLI validation functions and typer commands
  - `test_stability.py` — label alignment, consensus iterations, stability analyzer
  - `test_outcome_analysis.py` — outcome analysis with real statsmodels
  - `test_visualization_base.py` — base visualizer helpers and pure functions
  - `test_visualization_plots.py` — all 6 visualizer classes and facade
  - `test_multistate_analyzer.py` — multistate orchestrator and pathway analysis
  - `test_data_modules.py` — encoder, imputer, outlier handler, data splitter
  - `test_feature_selection.py` — variance, correlation, lasso, mutual info, mixed selector
  - `test_reporting.py` — HTML report generation and helper functions
  - `test_data_quality.py` — data quality assessor and Little's MCAR test

### Fixed

- Minor fixes in encoder, preprocessor, external validation, and feature selection modules discovered during testing

## [0.1.0] - 2025-12-01

- Initial release
