"""Shared test fixtures and safety nets."""

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import PhenoClusterConfig
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


@pytest.fixture
def minimal_config():
    """Minimal PhenoClusterConfig for unit tests."""
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {
                "continuous_columns": ["x1", "x2", "x3"],
                "categorical_columns": ["cat1"],
                "split": {},
            },
            "preprocessing": {},
            "model": {"n_clusters": 3},
            "outcome": {"enabled": True, "outcome_columns": ["outcome1"]},
            "inference": {"enabled": True, "fdr_correction": True},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


@pytest.fixture
def sample_dataframe():
    """60-row DataFrame with continuous, categorical, outcome, and survival columns."""
    rng = np.random.RandomState(42)
    n = 60
    return pd.DataFrame(
        {
            "x1": rng.randn(n),
            "x2": rng.randn(n),
            "x3": rng.randn(n),
            "cat1": rng.choice(["A", "B", "C"], n),
            "outcome1": rng.choice([0, 1], n),
            "time": rng.exponential(10, n),
            "event": rng.choice([0, 1], n),
        }
    )


@pytest.fixture
def sample_labels():
    """Cluster labels for 60 samples across 3 clusters."""
    return np.array([i % 3 for i in range(60)])


@pytest.fixture
def config_with_survival():
    """Config with survival targets defined."""
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {"continuous_columns": ["x1"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": 3},
            "outcome": {"enabled": False},
            "survival": {
                "enabled": True,
                "targets": [
                    {"name": "os", "time_column": "time", "event_column": "event"},
                ],
            },
            "inference": {"enabled": True},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


@pytest.fixture
def config_with_multistate():
    """Config with multistate states and transitions."""
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {"continuous_columns": ["x1"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": 3},
            "outcome": {"enabled": False},
            "multistate": {
                "enabled": True,
                "states": [
                    {"id": 0, "name": "initial", "state_type": "initial"},
                    {
                        "id": 1,
                        "name": "event",
                        "state_type": "transient",
                        "event_column": "event",
                        "time_column": "time",
                    },
                    {
                        "id": 2,
                        "name": "death",
                        "state_type": "absorbing",
                        "event_column": "death",
                        "time_column": "time_death",
                    },
                ],
                "transitions": [
                    {"name": "to_event", "from_state": 0, "to_state": 1},
                    {"name": "to_death", "from_state": 0, "to_state": 2},
                ],
            },
            "inference": {"enabled": True},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )
