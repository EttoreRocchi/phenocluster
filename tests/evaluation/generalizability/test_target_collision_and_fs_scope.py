"""Tests for target_column collision detection and feature_selector_scope."""

import warnings as warnings_module

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import (
    GeneralizabilityConfig,
    PhenoClusterConfig,
    TemporalSpec,
)
from phenocluster.evaluation.generalizability.derivation_fit import (
    _resolve_feature_selector_for_split,
)


def _config_with_outcome_and_lasso_target(target: str, error: bool = False):
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "x", "random_state": 0},
            "data": {
                "continuous_columns": ["c1", "c2"],
                "categorical_columns": [],
                "split": {"test_size": 0.2},
            },
            "preprocessing": {
                "categorical_encoding": {"method": "label"},
                "imputation": {"enabled": False},
                "outlier": {"enabled": False},
                "feature_selection": {
                    "enabled": True,
                    "method": "lasso",
                    "target_column": target,
                    "error_on_outcome_collision": error,
                },
            },
            "model": {"n_clusters": 2},
            "outcome": {
                "enabled": True,
                "outcome_columns": ["mortality_30d", "readmission_30d"],
            },
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


class TestTargetColumnCollision:
    def test_collision_with_outcome_warns_by_default(self):
        cfg = _config_with_outcome_and_lasso_target("mortality_30d", error=False)
        with warnings_module.catch_warnings(record=True) as captured:
            warnings_module.simplefilter("always")
            cfg.validate()
        msgs = [str(w.message) for w in captured]
        assert any("mortality_30d" in m and "outcome" in m for m in msgs)

    def test_collision_with_outcome_raises_when_strict(self):
        cfg = _config_with_outcome_and_lasso_target("mortality_30d", error=True)
        with pytest.raises(ValueError, match="mortality_30d"):
            cfg.validate()

    def test_no_collision_silent(self):
        cfg = _config_with_outcome_and_lasso_target("baseline_severity", error=False)
        with warnings_module.catch_warnings(record=True) as captured:
            warnings_module.simplefilter("always")
            cfg.validate()
        assert not any("baseline_severity" in str(w.message) for w in captured)

    def test_outcome_like_columns_includes_survival(self):
        cfg = PhenoClusterConfig.from_dict(
            {
                "global": {"project_name": "x", "random_state": 0},
                "data": {
                    "continuous_columns": ["c1"],
                    "categorical_columns": [],
                    "split": {"test_size": 0.2},
                },
                "preprocessing": {
                    "categorical_encoding": {"method": "label"},
                    "imputation": {"enabled": False},
                    "outlier": {"enabled": False},
                },
                "model": {"n_clusters": 2},
                "outcome": {"enabled": False},
                "survival": {
                    "enabled": True,
                    "targets": [
                        {
                            "name": "OS",
                            "time_column": "time_to_death",
                            "event_column": "death",
                        }
                    ],
                },
                "logging": {"level": "WARNING", "log_to_file": False},
            }
        )
        cols = cfg.outcome_like_columns()
        assert "time_to_death" in cols
        assert "death" in cols


class TestFeatureSelectorScopeConfig:
    def test_default_is_auto(self):
        cfg = GeneralizabilityConfig(
            enabled=True,
            temporal=TemporalSpec(time_column="t", time_cutoff="2020-01-01"),
        )
        assert cfg.feature_selector_scope == "auto"

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError, match="feature_selector_scope"):
            GeneralizabilityConfig(
                enabled=True,
                feature_selector_scope="bogus",
                temporal=TemporalSpec(time_column="t", time_cutoff="2020-01-01"),
            )


class TestResolveSelectorForSplit:
    def _cfg_with_variance_selector(self):
        return PhenoClusterConfig.from_dict(
            {
                "global": {"project_name": "x", "random_state": 0},
                "data": {
                    "continuous_columns": ["c1", "c2", "c3"],
                    "categorical_columns": [],
                    "split": {"test_size": 0.2},
                },
                "preprocessing": {
                    "categorical_encoding": {"method": "label"},
                    "imputation": {"enabled": False},
                    "outlier": {"enabled": False},
                    "feature_selection": {
                        "enabled": True,
                        "method": "variance",
                        "variance_threshold": 0.0,
                    },
                },
                "model": {"n_clusters": 2},
                "outcome": {"enabled": False},
                "logging": {"level": "WARNING", "log_to_file": False},
            }
        )

    def test_global_scope_returns_global_selector(self):
        cfg = self._cfg_with_variance_selector()
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"c1": rng.normal(size=50), "c2": rng.normal(size=50), "c3": rng.normal(size=50)}
        )
        sentinel = object()
        result, mode, notes = _resolve_feature_selector_for_split(
            cfg, sentinel, df, feature_selector_scope="global"
        )
        assert result is sentinel
        assert mode == "global_reused"
        assert notes == []

    def test_disabled_returns_none(self):
        cfg = PhenoClusterConfig.from_dict(
            {
                "global": {"project_name": "x", "random_state": 0},
                "data": {
                    "continuous_columns": ["c1"],
                    "categorical_columns": [],
                    "split": {"test_size": 0.2},
                },
                "preprocessing": {
                    "categorical_encoding": {"method": "label"},
                    "imputation": {"enabled": False},
                    "outlier": {"enabled": False},
                },
                "model": {"n_clusters": 2},
                "outcome": {"enabled": False},
                "logging": {"level": "WARNING", "log_to_file": False},
            }
        )
        df = pd.DataFrame({"c1": [1, 2, 3]})
        result, mode, _ = _resolve_feature_selector_for_split(
            cfg, None, df, feature_selector_scope="auto"
        )
        assert result is None
        assert mode == "none"

    def test_supervised_target_collision_falls_back_in_auto(self):
        cfg = _config_with_outcome_and_lasso_target("mortality_30d", error=False)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "c1": rng.normal(size=50),
                "c2": rng.normal(size=50),
                "mortality_30d": rng.integers(0, 2, size=50),
            }
        )
        sentinel = object()
        result, mode, notes = _resolve_feature_selector_for_split(
            cfg, sentinel, df, feature_selector_scope="auto"
        )
        assert result is sentinel
        assert mode == "global_reused_with_warning"
        assert any("mortality_30d" in n for n in notes)

    def test_supervised_target_collision_raises_in_strict_per_split(self):
        cfg = _config_with_outcome_and_lasso_target("mortality_30d", error=False)
        df = pd.DataFrame({"mortality_30d": [0, 1, 0, 1]})
        sentinel = object()
        with pytest.raises(ValueError, match="mortality_30d"):
            _resolve_feature_selector_for_split(
                cfg, sentinel, df, feature_selector_scope="per_split"
            )
