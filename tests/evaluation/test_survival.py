"""Tests for survival analysis -- edge cases and guard conditions."""

import numpy as np
import pandas as pd
import pytest


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


class TestAnalyzeSurvivalEdgeCases:
    """Coverage for guard branches in analyze_survival."""

    def test_missing_time_column_raises(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer(_make_config(), n_clusters=2)
        df = _make_survival_data(n=20, n_events=5).drop(columns=["time"])
        with pytest.raises(ValueError):
            analyzer.analyze_survival(df, df["phenotype"].values, "time", "event")

    def test_no_valid_rows_returns_empty(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer(_make_config(), n_clusters=2)
        df = _make_survival_data(n=20, n_events=5)
        df.loc[:, "time"] = np.nan
        results = analyzer.analyze_survival(df, df["phenotype"].values, "time", "event")
        assert results == {}


class TestWeightedSurvival:
    """Tests for analyze_weighted_survival."""

    @pytest.mark.filterwarnings("ignore::lifelines.exceptions.StatisticalWarning")
    def test_weighted_survival_returns_results(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer(_make_config(), n_clusters=2)
        df = _make_survival_data(n=80, n_events=30)
        rng = np.random.RandomState(0)
        probs = rng.dirichlet([3, 3], size=len(df))
        results = analyzer.analyze_weighted_survival(df, probs, "time", "event")
        for key in (
            "weighted_km",
            "comparison",
            "median_survival",
            "time_column",
            "event_column",
            "analysis_type",
        ):
            assert key in results
        assert results["analysis_type"] == "weighted"

    def test_mismatched_rows_raises(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer(_make_config(), n_clusters=2)
        df = _make_survival_data(n=10, n_events=3)
        with pytest.raises(ValueError):
            analyzer.analyze_weighted_survival(df, np.ones((5, 2)), "time", "event")

    def test_mismatched_clusters_raises(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer(_make_config(), n_clusters=2)
        df = _make_survival_data(n=20, n_events=5)
        with pytest.raises(ValueError):
            analyzer.analyze_weighted_survival(df, np.ones((20, 3)), "time", "event")

    def test_no_valid_rows_returns_empty(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer(_make_config(), n_clusters=2)
        df = _make_survival_data(n=20, n_events=5)
        df.loc[:, "time"] = np.nan
        results = analyzer.analyze_weighted_survival(df, np.ones((20, 2)) / 2.0, "time", "event")
        assert results == {}


class TestSurvivalAtTimes:
    def test_returns_interpolated_values(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        km = {
            0: {
                "timeline": np.array([0.0, 5.0, 10.0]),
                "survival_function": np.array([1.0, 0.7, 0.4]),
            }
        }
        out = SurvivalAnalyzer._survival_at_times(km, np.array([3.0, 7.0]))
        assert 0 in out
        assert len(out[0]) == 2


class TestPHCheck:
    def test_check_ph_runs(self):
        from phenocluster.evaluation.survival import SurvivalAnalyzer

        analyzer = SurvivalAnalyzer(_make_config(), n_clusters=2)
        df = _make_survival_data(n=80, n_events=30)
        df["phenotype"] = np.array([i % 2 for i in range(len(df))])
        ph = analyzer._check_ph(df, "time", "event")
        assert ph is None or isinstance(ph, dict)


class TestGrambschTherneauGlobal:
    """Spot-check the global Schoenfeld-residual PH test against lifelines."""

    def test_matches_lifelines_single_covariate(self):
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test

        from phenocluster.evaluation.survival import _grambsch_therneau_global

        rng = np.random.default_rng(0)
        n = 800
        x = rng.normal(size=n)
        times = -np.log(rng.uniform(size=n)) / np.exp(0.5 * x) + 0.5
        events = (rng.uniform(size=n) < 0.7).astype(int)
        df = pd.DataFrame({"t": times, "e": events, "x": x})
        cph = CoxPHFitter()
        cph.fit(df, duration_col="t", event_col="e", show_progress=False)

        lif = proportional_hazard_test(cph, df, time_transform="log")
        gt = _grambsch_therneau_global(cph, df, time_col="t", transform="log")
        assert gt is not None
        assert np.isclose(
            gt["test_statistic"],
            lif.summary["test_statistic"].iloc[0],
            rtol=5e-3,
        )

    def test_handles_zero_event_times(self):
        from lifelines import CoxPHFitter

        from phenocluster.evaluation.survival import _grambsch_therneau_global

        df = pd.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                "e": [1] * 10,
                "x": [0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.7, -0.8, 0.9, -0.3],
            }
        )
        cph = CoxPHFitter()
        cph.fit(df, duration_col="t", event_col="e", show_progress=False)
        gt = _grambsch_therneau_global(cph, df, time_col="t", transform="log")
        assert gt is not None
        assert np.isfinite(gt["test_statistic"])
        assert 0.0 <= gt["p_value"] <= 1.0
