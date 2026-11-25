"""Tests for survival analysis -- edge cases and guard conditions."""

from pathlib import Path

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


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def two_covariate_data():
    """40 subjects, binary x1 and continuous x2, no delayed entry."""
    return pd.read_csv(FIXTURES / "schoenfeld_two_covariates.csv")


@pytest.fixture
def delayed_entry_data():
    """40 subjects with delayed entry (start, stop] and tied event times."""
    return pd.read_csv(FIXTURES / "schoenfeld_delayed_entry.csv")


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

    def test_per_covariate_and_global_share_km_transform(self, two_covariate_data):
        import logging

        from phenocluster.evaluation.survival import _check_proportional_hazards

        ref = two_covariate_data
        df = pd.DataFrame({"t": ref["t"], "e": ref["e"], "phenotype": ref["x1"].astype(int)})
        ph = _check_proportional_hazards(df, "t", "e", 2, 0, logging.getLogger("test"))
        assert ph["tested"]
        assert len(ph["per_covariate"]) == 1
        assert ph["global_test"]["transform"] == "km"
        # With a single covariate the global and per-covariate statistics
        # coincide only when both tests use the same time transform.
        assert np.isclose(
            ph["global_test"]["test_statistic"],
            ph["per_covariate"][0]["test_statistic"],
            rtol=1e-8,
        )


class TestKMTimeTransform:
    def test_left_continuous_with_ties_and_censoring(self):
        from phenocluster.evaluation.survival import _km_time_transform

        out = _km_time_transform(pd.Series([1.0, 2.0, 2.0, 3.0, 4.0]), pd.Series([1, 1, 1, 0, 1]))
        assert isinstance(out, pd.Series)
        np.testing.assert_allclose(out.to_numpy(), [0.0, 0.2, 0.2, 0.6, 0.6])

    def test_weights(self):
        from phenocluster.evaluation.survival import _km_time_transform

        out = _km_time_transform(
            [1.0, 2.0, 2.0, 3.0, 4.0], [1, 1, 1, 0, 1], weights=[2.0, 1.0, 1.0, 1.0, 1.0]
        )
        np.testing.assert_allclose(out.to_numpy(), [0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3])

    def test_delayed_entry(self):
        from phenocluster.evaluation.survival import _km_time_transform

        durations, events = [2.0, 3.0, 4.0], [1, 1, 1]
        without = _km_time_transform(durations, events)
        with_entry = _km_time_transform(durations, events, entries=[0.0, 1.0, 2.5])
        np.testing.assert_allclose(without.to_numpy(), [0.0, 1 / 3, 2 / 3])
        np.testing.assert_allclose(with_entry.to_numpy(), [0.0, 0.5, 0.75])


class TestSchoenfeldPHTests:
    """Schoenfeld PH tests checked against R and lifelines."""

    @staticmethod
    def _fit(df, **kwargs):
        from lifelines import CoxPHFitter

        cph = CoxPHFitter()
        cph.fit(df, show_progress=False, **kwargs)
        return cph

    def test_matches_r_reference_two_covariates(self, two_covariate_data):
        """Reference: survival::cox.zph before version 3.0 recomputed in R
        (survival 3.8.12) from the coxph Schoenfeld residuals of the same
        data, Efron ties, Kaplan-Meier transform."""
        from phenocluster.evaluation.survival import _schoenfeld_ph_tests

        df = two_covariate_data
        cph = self._fit(df, duration_col="t", event_col="e")
        tests = _schoenfeld_ph_tests(cph, df, "t", "e")
        np.testing.assert_allclose(
            [c["test_statistic"] for c in tests["per_covariate"]], [9.261274, 0.011564], rtol=1e-4
        )
        assert tests["global_test"]["df"] == 2
        assert np.isclose(tests["global_test"]["test_statistic"], 9.342859, rtol=1e-4)
        assert tests["global_test"]["transform"] == "km"

    def test_matches_r_reference_with_delayed_entry(self, delayed_entry_data):
        """Reference: as above on Surv(start, stop, event) ~ x + cluster(id),
        with model-based variance and left-truncated Kaplan-Meier."""
        from phenocluster.evaluation.survival import _schoenfeld_ph_tests

        df = delayed_entry_data
        cph = self._fit(
            df, duration_col="stop", event_col="event", entry_col="start", cluster_col="id"
        )
        tests = _schoenfeld_ph_tests(cph, df, "stop", "event", entry_col="start")
        assert np.isclose(tests["per_covariate"][0]["test_statistic"], 2.629771, rtol=1e-4)
        assert np.isclose(tests["global_test"]["test_statistic"], 2.629771, rtol=1e-4)

    def test_matches_lifelines_without_entry(self, two_covariate_data):
        from lifelines.statistics import proportional_hazard_test

        from phenocluster.evaluation.survival import (
            _km_time_transform,
            _schoenfeld_ph_tests,
            _schoenfeld_residuals,
        )

        df = two_covariate_data
        cph = self._fit(df, duration_col="t", event_col="e")

        residuals, _ = _schoenfeld_residuals(cph, df, "t", "e")
        lifelines_residuals = cph.compute_residuals(df, kind="schoenfeld").to_numpy()
        np.testing.assert_allclose(residuals, lifelines_residuals, atol=1e-10)

        tests = _schoenfeld_ph_tests(cph, df, "t", "e")
        lif = proportional_hazard_test(cph, df, time_transform=_km_time_transform)
        np.testing.assert_allclose(
            [c["test_statistic"] for c in tests["per_covariate"]],
            lif.summary["test_statistic"].to_numpy(),
            rtol=1e-8,
        )

    def test_binary_covariate_global_matches_lifelines(self):
        """With one covariate the global statistic equals the per-covariate
        one; a binary covariate (std far from 1) exposes any rescaling of the
        variance matrix."""
        from lifelines.statistics import proportional_hazard_test

        from phenocluster.evaluation.survival import _km_time_transform, _schoenfeld_ph_tests

        rng = np.random.default_rng(1)
        n = 600
        x = (rng.uniform(size=n) < 0.3).astype(float)
        times = rng.weibull(np.where(x == 1, 0.6, 1.6))
        events = (rng.uniform(size=n) < 0.8).astype(int)
        df = pd.DataFrame({"t": times, "e": events, "x": x})
        cph = self._fit(df, duration_col="t", event_col="e")

        tests = _schoenfeld_ph_tests(cph, df, "t", "e")
        lif = proportional_hazard_test(cph, df, time_transform=_km_time_transform)
        assert np.isclose(
            tests["global_test"]["test_statistic"],
            lif.summary["test_statistic"].iloc[0],
            rtol=1e-8,
        )

    def test_invariant_to_covariate_scale(self, two_covariate_data):
        from phenocluster.evaluation.survival import _schoenfeld_ph_tests

        df = two_covariate_data
        rescaled = df.assign(x1=df["x1"] * 10.0, x2=df["x2"] * 0.01)
        results = []
        for frame in (df, rescaled):
            cph = self._fit(frame, duration_col="t", event_col="e")
            results.append(_schoenfeld_ph_tests(cph, frame, "t", "e"))
        assert np.isclose(
            results[0]["global_test"]["test_statistic"],
            results[1]["global_test"]["test_statistic"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            [c["test_statistic"] for c in results[0]["per_covariate"]],
            [c["test_statistic"] for c in results[1]["per_covariate"]],
            rtol=1e-6,
        )

    def test_handles_zero_event_times(self):
        from phenocluster.evaluation.survival import _schoenfeld_ph_tests

        df = pd.DataFrame(
            {
                "t": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                "e": [1] * 10,
                "x": [0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.7, -0.8, 0.9, -0.3],
            }
        )
        cph = self._fit(df, duration_col="t", event_col="e")
        tests = _schoenfeld_ph_tests(cph, df, "t", "e")
        assert tests is not None
        assert np.isfinite(tests["global_test"]["test_statistic"])
        assert 0.0 <= tests["global_test"]["p_value"] <= 1.0
