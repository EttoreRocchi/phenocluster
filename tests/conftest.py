"""Shared test fixtures and safety nets."""

import pytest

from phenocluster.utils.logging import PhenoClusterLogger


@pytest.fixture(autouse=True)
def _clean_logger_cache():
    """Clear the logger cache between tests to prevent cross-test contamination.

    Without this, a logger created with log_to_file=True in one test would be
    reused (from the class-level cache) in subsequent tests that expect different
    logging config.
    """
    PhenoClusterLogger._loggers.clear()
    yield
    PhenoClusterLogger._loggers.clear()


@pytest.fixture(autouse=True)
def _no_results_dir_in_project(tmp_path, monkeypatch):
    """Prevent tests from creating a 'results/' directory in the project root.

    Monkeypatches LoggingConfig.log_to_file default to False so that any
    PhenoClusterConfig created without explicit log_to_file=True won't
    trigger directory creation via get_logger().
    """
    from phenocluster.config import LoggingConfig

    original = LoggingConfig.__dataclass_fields__["log_to_file"].default
    LoggingConfig.__dataclass_fields__["log_to_file"].default = False
    yield
    LoggingConfig.__dataclass_fields__["log_to_file"].default = original
