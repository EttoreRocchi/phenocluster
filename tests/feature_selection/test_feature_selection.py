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


class TestBaseFeatureSelectorErrors:
    """Error paths on BaseFeatureSelector before/after fit."""

    def test_transform_before_fit_raises(self):
        from phenocluster.core.exceptions import ModelNotFittedError

        sel = VarianceSelector()
        with pytest.raises(ModelNotFittedError):
            sel.transform(_sample_df())

    def test_get_selected_before_fit_raises(self):
        from phenocluster.core.exceptions import ModelNotFittedError

        sel = VarianceSelector()
        with pytest.raises(ModelNotFittedError):
            sel.get_selected_features()

    def test_get_removed_before_fit_raises(self):
        from phenocluster.core.exceptions import ModelNotFittedError

        sel = VarianceSelector()
        with pytest.raises(ModelNotFittedError):
            sel.get_removed_features()

    def test_get_scores_before_fit_raises(self):
        from phenocluster.core.exceptions import ModelNotFittedError

        sel = VarianceSelector()
        with pytest.raises(ModelNotFittedError):
            sel.get_feature_scores()

    def test_n_selected_zero_before_fit(self):
        sel = VarianceSelector()
        assert sel.n_selected == 0
        assert sel.n_removed == 0

    def test_n_selected_after_fit(self):
        sel = VarianceSelector(variance_threshold=0.01)
        sel.fit(_sample_df())
        assert sel.n_selected + sel.n_removed == 4

    def test_transform_missing_column_raises(self):
        from phenocluster.core.exceptions import FeatureSelectionError

        sel = VarianceSelector(variance_threshold=0.01)
        df = _sample_df()
        sel.fit(df)
        smaller = df.drop(columns=["f1"])
        with pytest.raises(FeatureSelectionError):
            sel.transform(smaller)


class TestVarianceCategoricalPath:
    def test_dominant_category_removed(self):
        df = pd.DataFrame(
            {
                "x": np.linspace(0, 10, 100),
                "skewed": ["A"] * 99 + ["B"],
            }
        )
        sel = VarianceSelector(frequency_threshold=0.95)
        sel.fit(df, continuous_cols=["x"], categorical_cols=["skewed"])
        assert "skewed" in sel.get_removed_features()

    def test_balanced_categorical_kept(self):
        df = pd.DataFrame(
            {
                "x": np.linspace(0, 10, 100),
                "balanced": ["A"] * 50 + ["B"] * 50,
            }
        )
        sel = VarianceSelector(frequency_threshold=0.95)
        sel.fit(df, continuous_cols=["x"], categorical_cols=["balanced"])
        assert "balanced" in sel.get_selected_features()

    def test_empty_categorical_removed(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "empty": [np.nan, np.nan, np.nan]})
        df["empty"] = df["empty"].astype("object")
        sel = VarianceSelector(frequency_threshold=0.99)
        sel.fit(df, continuous_cols=["x"], categorical_cols=["empty"])
        scores = sel.get_feature_scores()
        assert "empty" not in sel.get_selected_features() or scores["empty"] == 0.0

    def test_auto_detection(self):
        df = pd.DataFrame(
            {
                "num": np.linspace(0, 1, 50),
                "cat": ["A"] * 25 + ["B"] * 25,
            }
        )
        sel = VarianceSelector(variance_threshold=0.0)
        sel.fit(df)
        assert "num" in sel.get_selected_features()


class TestLassoExtras:
    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_classification_path(self):
        from phenocluster.feature_selection.lasso import LassoSelector

        rng = np.random.RandomState(0)
        n = 80
        x = rng.randn(n)
        df = pd.DataFrame(
            {
                "useful": x,
                "noise": rng.randn(n),
            }
        )
        y = (x > 0).astype(int)
        sel = LassoSelector(random_state=0, cv=3)
        sel.fit(df, y)
        assert sel.alpha_ is not None

    def test_n_features_caps(self):
        from phenocluster.feature_selection.lasso import LassoSelector

        rng = np.random.RandomState(0)
        n = 60
        x = rng.randn(n)
        df = pd.DataFrame(
            {
                "a": x,
                "b": rng.randn(n),
                "c": rng.randn(n),
            }
        )
        y = x * 2 + rng.randn(n) * 0.1
        sel = LassoSelector(random_state=0, n_features=2, cv=3)
        sel.fit(df, y)
        assert len(sel.get_selected_features()) <= 2

    def test_no_numeric_features_raises(self):
        from phenocluster.core.exceptions import FeatureSelectionError
        from phenocluster.feature_selection.lasso import LassoSelector

        df = pd.DataFrame({"cat": ["a", "b", "c", "d"]})
        sel = LassoSelector(random_state=0, cv=2)
        with pytest.raises(FeatureSelectionError):
            sel.fit(df, np.array([0, 1, 0, 1]))

    def test_non_numeric_columns_kept_with_unit_score(self):
        from phenocluster.feature_selection.lasso import LassoSelector

        rng = np.random.RandomState(0)
        n = 60
        x = rng.randn(n)
        df = pd.DataFrame(
            {
                "x": x,
                "label": rng.choice(["A", "B"], n),
            }
        )
        y = x * 2 + rng.randn(n) * 0.1
        sel = LassoSelector(random_state=0, cv=3)
        sel.fit(df, y)
        scores = sel.get_feature_scores()
        assert scores["label"] == 1.0


class TestMutualInfoExtras:
    def test_percentile_path(self):
        from phenocluster.feature_selection.mutual_info import MutualInfoSelector

        rng = np.random.RandomState(0)
        n = 80
        x = rng.randn(n)
        df = pd.DataFrame(
            {
                "useful": x,
                "noise1": rng.randn(n),
                "noise2": rng.randn(n),
                "noise3": rng.randn(n),
            }
        )
        y = (x > 0).astype(float)
        sel = MutualInfoSelector(percentile=25.0, random_state=0)
        sel.fit(df, y)
        assert len(sel.get_selected_features()) <= 4

    def test_handles_nan_with_imputation(self):
        from phenocluster.feature_selection.mutual_info import MutualInfoSelector

        rng = np.random.RandomState(0)
        n = 80
        df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
        df.loc[5:10, "a"] = np.nan
        y = rng.choice([0, 1], n).astype(float)
        sel = MutualInfoSelector(n_features=1, random_state=0)
        sel.fit(df, y)
        assert sel.mi_scores_ is not None


class TestMixedSelectorExtras:
    def test_disabled_keeps_all(self):
        from phenocluster.config import FeatureSelectionConfig
        from phenocluster.feature_selection.mixed_selector import MixedDataFeatureSelector

        config = FeatureSelectionConfig(enabled=False, method="variance")
        df = _sample_df()
        sel = MixedDataFeatureSelector(config)
        sel.fit(df)
        assert sel.get_selected_features() == list(df.columns)
        scores = sel.get_feature_scores()
        assert all(v == 1.0 for v in scores.values())

    def test_require_target_raises(self):
        from phenocluster.config import FeatureSelectionConfig
        from phenocluster.core.exceptions import FeatureSelectionError
        from phenocluster.feature_selection.mixed_selector import MixedDataFeatureSelector

        config = FeatureSelectionConfig(enabled=True, method="lasso", target_column="y")
        df = _sample_df()
        sel = MixedDataFeatureSelector(config)
        with pytest.raises(FeatureSelectionError):
            sel.fit(df, y=None)

    def test_selection_report_keys(self):
        from phenocluster.config import FeatureSelectionConfig
        from phenocluster.feature_selection.mixed_selector import MixedDataFeatureSelector

        config = FeatureSelectionConfig(enabled=True, method="variance", variance_threshold=0.01)
        sel = MixedDataFeatureSelector(config, continuous_cols=["f1", "f2", "f3", "constant"])
        sel.fit(_sample_df())
        report = sel.get_selection_report()
        for key in (
            "method",
            "n_original",
            "n_selected",
            "n_removed",
            "selection_ratio",
            "selected_features",
            "removed_features",
            "feature_scores",
        ):
            assert key in report

    def test_selection_report_before_fit_raises(self):
        from phenocluster.config import FeatureSelectionConfig
        from phenocluster.core.exceptions import ModelNotFittedError
        from phenocluster.feature_selection.mixed_selector import MixedDataFeatureSelector

        config = FeatureSelectionConfig(enabled=True, method="variance")
        sel = MixedDataFeatureSelector(config)
        with pytest.raises(ModelNotFittedError):
            sel.get_selection_report()
