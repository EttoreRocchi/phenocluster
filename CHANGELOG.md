# Changelog

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
