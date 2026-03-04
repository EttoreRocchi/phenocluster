"""Tests for survival analysis -- edge cases and guard conditions."""

import numpy as np
import pandas as pd


def _make_config(n_clusters=2, inference_enabled=True):
    """Create a minimal PhenoClusterConfig for survival tests."""
    from phenocluster.config import PhenoClusterConfig

    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {"continuous_columns": ["x"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": n_clusters},
            "outcome": {"enabled": False},
            "inference": {"enabled": inference_enabled},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


def _make_survival_data(n=50, n_events=None, seed=42):
    """Create synthetic survival data with controlled event count."""
    rng = np.random.RandomState(seed)
    times = rng.exponential(10, size=n)
    events = np.zeros(n, dtype=int)
    if n_events is not None and n_events > 0:
        event_idx = rng.choice(n, size=min(n_events, n), replace=False)
        events[event_idx] = 1
    phenotypes = np.array([i % 2 for i in range(n)])
    df = pd.DataFrame({"time": times, "event": events, "phenotype": phenotypes})
    return df


class TestCoxEventsGuard:
    """Test the < 2 events guard in _fit_cox_model."""

    def _call_fit_cox(self, df, n_clusters=2, ref=0):
        """Helper to call the standalone _fit_cox_model function."""
        import logging

        from phenocluster.evaluation.survival import _fit_cox_model

        config = _make_config(n_clusters=n_clusters)
        logger = logging.getLogger("test_survival")
        return _fit_cox_model(
            df,
            "time",
            "event",
            n_clusters,
            ref,
            config.inference,
            1000,
            logger,
        )

    def test_zero_events_returns_empty(self):
        """All censored data should return empty comparison dict."""
        df = _make_survival_data(n=30, n_events=0)
        comparison = self._call_fit_cox(df)
        assert comparison == {}

    def test_one_event_returns_empty(self):
        """Single event should trigger the < 2 events guard."""
        df = _make_survival_data(n=30, n_events=1)
        comparison = self._call_fit_cox(df)
        assert comparison == {}

    def test_sufficient_events_returns_hr(self):
        """With enough events, Cox PH should return HR estimates."""
        df = _make_survival_data(n=100, n_events=30)
        comparison = self._call_fit_cox(df)

        # Should have at least one comparison key
        assert len(comparison) > 0

        # Check structure of the HR result
        for key, result in comparison.items():
            assert "HR" in result
            assert "CI_lower" in result
            assert "CI_upper" in result
            assert "p_value" in result
            assert result["HR"] > 0
            assert result["CI_lower"] <= result["HR"] <= result["CI_upper"]


class TestAnalyzeSurvivalResultKeys:
    """Test that analyze_survival returns the expected result structure."""

    def test_comparison_key_in_results(self):
        """Results dict should use 'comparison' key (not 'bayesian_comparison')."""
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        config = _make_config()
        analyzer = SurvivalAnalyzer(config, n_clusters=2)
        df = _make_survival_data(n=100, n_events=30)
        labels = df["phenotype"].values

        results = analyzer.analyze_survival(df, labels, "time", "event")

        assert "comparison" in results
        assert "bayesian_comparison" not in results
        assert "survival_data" in results
        assert "median_survival" in results
        assert "logrank_p_value" in results
