"""Tests for feature selection module."""

import numpy as np
import pandas as pd
import pytest

from phenocluster.feature_selection.correlation import CorrelationSelector
from phenocluster.feature_selection.variance import VarianceSelector


def _sample_df(n=100, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n) * 2 + 5,
            "f3": rng.randn(n),
            "constant": np.ones(n),
        }
    )


class TestVarianceSelector:
    def test_removes_constant(self):
        sel = VarianceSelector(variance_threshold=0.01)
        df = _sample_df()
        sel.fit(df)
        selected = sel.get_selected_features()
        assert "constant" not in selected
        assert "f1" in selected

    def test_threshold(self):
        sel = VarianceSelector(variance_threshold=100.0)
        df = _sample_df()
        sel.fit(df)
        # Very high threshold should remove most features
        selected = sel.get_selected_features()
        assert len(selected) < 4

    def test_transform_output(self):
        sel = VarianceSelector(variance_threshold=0.01)
        df = _sample_df()
        result = sel.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert "constant" not in result.columns


class TestCorrelationSelector:
    def test_removes_correlated(self):
        rng = np.random.RandomState(42)
        n = 100
        x = rng.randn(n)
        df = pd.DataFrame(
            {
                "a": x,
                "b": x + rng.randn(n) * 0.01,  # nearly identical to a
                "c": rng.randn(n),
            }
        )
        sel = CorrelationSelector(threshold=0.95)
        sel.fit(df)
        selected = sel.get_selected_features()
        # Either a or b should be removed, but not both
        assert len(selected) == 2

    def test_no_removal_for_uncorrelated(self):
        df = _sample_df()[["f1", "f2", "f3"]]
        sel = CorrelationSelector(threshold=0.95)
        sel.fit(df)
        selected = sel.get_selected_features()
        assert len(selected) == 3

    def test_correlated_pairs_attribute(self):
        rng = np.random.RandomState(42)
        n = 100
        x = rng.randn(n)
        df = pd.DataFrame({"a": x, "b": x + rng.randn(n) * 0.001, "c": rng.randn(n)})
        sel = CorrelationSelector(threshold=0.95)
        sel.fit(df)
        assert sel.correlated_pairs_ is not None


class TestLassoSelector:
    def test_requires_target(self):
        from phenocluster.feature_selection.lasso import LassoSelector

        sel = LassoSelector(random_state=42)
        df = _sample_df()[["f1", "f2", "f3"]]
        with pytest.raises((ValueError, TypeError)):
            sel.fit(df)

    def test_selects_features(self):
        from phenocluster.feature_selection.lasso import LassoSelector

        rng = np.random.RandomState(42)
        n = 100
        x1 = rng.randn(n)
        df = pd.DataFrame(
            {
                "useful": x1,
                "noise1": rng.randn(n),
                "noise2": rng.randn(n),
            }
        )
        y = x1 * 2 + rng.randn(n) * 0.1
        sel = LassoSelector(random_state=42)
        sel.fit(df, y)
        selected = sel.get_selected_features()
        assert len(selected) >= 1


class TestMutualInfoSelector:
    def test_requires_target(self):
        from phenocluster.feature_selection.mutual_info import MutualInfoSelector

        sel = MutualInfoSelector(random_state=42)
        df = _sample_df()[["f1", "f2", "f3"]]
        with pytest.raises((ValueError, TypeError)):
            sel.fit(df)

    def test_selects_features(self):
        from phenocluster.feature_selection.mutual_info import MutualInfoSelector

        rng = np.random.RandomState(42)
        n = 200
        x1 = rng.randn(n)
        df = pd.DataFrame(
            {
                "useful": x1,
                "noise1": rng.randn(n),
                "noise2": rng.randn(n),
            }
        )
        y = (x1 > 0).astype(float)
        sel = MutualInfoSelector(n_features=2, random_state=42)
        sel.fit(df, y)
        selected = sel.get_selected_features()
        assert len(selected) == 2


class TestMixedDataFeatureSelector:
    def test_fit_transform_returns_dataframe(self):
        from phenocluster.config import FeatureSelectionConfig
        from phenocluster.feature_selection.mixed_selector import MixedDataFeatureSelector

        config = FeatureSelectionConfig(
            enabled=True,
            method="variance",
            variance_threshold=0.01,
        )
        df = _sample_df()
        sel = MixedDataFeatureSelector(config, continuous_cols=["f1", "f2", "f3", "constant"])
        result = sel.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert "constant" not in result.columns

    def test_get_selected_features(self):
        from phenocluster.config import FeatureSelectionConfig
        from phenocluster.feature_selection.mixed_selector import MixedDataFeatureSelector

        config = FeatureSelectionConfig(
            enabled=True,
            method="variance",
            variance_threshold=0.01,
        )
        df = _sample_df()
        sel = MixedDataFeatureSelector(config, continuous_cols=["f1", "f2", "f3", "constant"])
        sel.fit(df)
        features = sel.get_selected_features()
        assert isinstance(features, list)
        assert "constant" not in features
