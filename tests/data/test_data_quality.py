"""Tests for data quality assessment module."""

import numpy as np
import pandas as pd

from phenocluster.config import PhenoClusterConfig
from phenocluster.evaluation.data_quality import DataQualityAssessor, littles_mcar_test


def _make_config():
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {
                "continuous_columns": ["c1", "c2", "c3"],
                "categorical_columns": [],
                "split": {},
            },
            "preprocessing": {},
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
            "data_quality": {
                "enabled": True,
                "missing_threshold": 0.15,
                "correlation_threshold": 0.9,
                "variance_threshold": 0.01,
            },
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


def _sample_df(n=100, seed=42, add_missing=False):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "c1": rng.randn(n),
            "c2": rng.randn(n) * 2 + 5,
            "c3": rng.randn(n),
        }
    )
    if add_missing:
        df.loc[0:4, "c1"] = np.nan
        df.loc[10:14, "c2"] = np.nan
    return df


class TestDataQualityAssessor:
    def test_assess_complete_data(self):
        cfg = _make_config()
        assessor = DataQualityAssessor(cfg)
        df = _sample_df()
        result = assessor.assess_data_quality(df)
        assert isinstance(result, dict)
        assert "summary" in result

    def test_assess_with_missing(self):
        cfg = _make_config()
        assessor = DataQualityAssessor(cfg)
        df = _sample_df(add_missing=True)
        result = assessor.assess_data_quality(df)
        assert isinstance(result, dict)
        missing = result.get("missing_data", {})
        assert isinstance(missing, dict)

    def test_assess_correlation(self):
        cfg = _make_config()
        assessor = DataQualityAssessor(cfg)
        rng = np.random.RandomState(42)
        n = 100
        x = rng.randn(n)
        df = pd.DataFrame(
            {
                "c1": x,
                "c2": x + rng.randn(n) * 0.01,  # highly correlated
                "c3": rng.randn(n),
            }
        )
        result = assessor.assess_data_quality(df)
        assert isinstance(result, dict)

    def test_assess_variance(self):
        cfg = _make_config()
        assessor = DataQualityAssessor(cfg)
        df = _sample_df()
        df["constant"] = 1.0
        result = assessor.assess_data_quality(df)
        assert isinstance(result, dict)


class TestLittlesMCARTest:
    def test_complete_data(self):
        df = _sample_df()
        result = littles_mcar_test(df)
        # With no missing data, function should handle gracefully
        assert isinstance(result, dict)

    def test_with_missing(self):
        df = _sample_df(add_missing=True)
        result = littles_mcar_test(df)
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "chi_square" in result
        assert "is_mcar" in result

    def test_all_missing_column(self):
        df = _sample_df()
        df["all_nan"] = np.nan
        result = littles_mcar_test(df)
        assert isinstance(result, dict)
