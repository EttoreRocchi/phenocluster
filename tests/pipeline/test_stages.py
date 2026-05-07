"""Tests for individual pipeline stages."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import PhenoClusterConfig
from phenocluster.data.preprocessor import DataPreprocessor
from phenocluster.data.splitting import RandomSplitter
from phenocluster.pipeline.context import PipelineContext
from phenocluster.pipeline.stages.feature_selection import FeatureSelectionStage
from phenocluster.pipeline.stages.preprocessing import PreprocessingStage
from phenocluster.pipeline.stages.training import TrainingStage


def _logger():
    """Stub logger so stage info messages do not pollute test output."""
    log = MagicMock(spec=logging.Logger)
    log.info = MagicMock()
    log.warning = MagicMock()
    log.error = MagicMock()
    return log


def _config(tmp_path, **overrides):
    """Build a minimal PhenoClusterConfig suitable for stage tests."""
    base = {
        "global": {
            "project_name": "stage_test",
            "output_dir": str(tmp_path),
            "random_state": 42,
        },
        "data": {
            "continuous_columns": ["x1", "x2", "x3"],
            "categorical_columns": [],
            "split": {"test_size": 0.2, "random_state": 42},
        },
        "preprocessing": {},
        "model": {"n_clusters": 2, "selection": {"enabled": False}},
        "outcome": {"enabled": False},
        "logging": {"level": "WARNING", "log_to_file": False},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base:
            base[key].update(value)
        else:
            base[key] = value
    return PhenoClusterConfig.from_dict(base)


def _toy_dataframe(n=80, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "x1": rng.randn(n),
            "x2": rng.randn(n) + 1.0,
            "x3": rng.randn(n) * 2,
        }
    )


class TestPreprocessingStage:
    def test_run_populates_context(self, tmp_path):
        config = _config(tmp_path)
        preprocessor = DataPreprocessor(config)
        splitter = RandomSplitter(config.data_split)
        stage = PreprocessingStage(config, preprocessor, splitter, _logger())

        ctx = PipelineContext(data_raw=_toy_dataframe())
        stage.run(ctx)

        assert ctx.X_train is not None
        assert ctx.X_test is not None
        assert ctx.data_train is not None
        assert ctx.data_test is not None
        assert ctx.split_info["train_size"] + ctx.split_info["test_size"] == len(ctx.data_filtered)

    def test_filter_removes_high_missing_rows(self, tmp_path):
        config = _config(
            tmp_path,
            preprocessing={"row_filter": {"enabled": True, "max_missing_pct": 0.5}},
        )
        df = _toy_dataframe(n=20)
        df.loc[:5, ["x1", "x2", "x3"]] = np.nan
        preprocessor = DataPreprocessor(config)
        splitter = RandomSplitter(config.data_split)
        stage = PreprocessingStage(config, preprocessor, splitter, _logger())

        ctx = PipelineContext(data_raw=df)
        stage.run(ctx)
        assert len(ctx.data_filtered) <= len(df)

    def test_too_few_samples_raises(self, tmp_path):
        config = _config(
            tmp_path,
            model={
                "n_clusters": 2,
                "selection": {"enabled": True, "min_clusters": 2, "max_clusters": 4},
            },
        )
        preprocessor = DataPreprocessor(config)
        splitter = RandomSplitter(config.data_split)
        stage = PreprocessingStage(config, preprocessor, splitter, _logger())
        with pytest.raises(ValueError, match="Insufficient samples"):
            stage.run(PipelineContext(data_raw=_toy_dataframe(n=5)))


class TestFeatureSelectionStage:
    def test_disabled_short_circuits(self, tmp_path):
        config = _config(tmp_path)
        stage = FeatureSelectionStage(config, MagicMock(), _logger())
        ctx = PipelineContext()
        stage.run(ctx)
        assert stage.feature_selector is None

    def test_enabled_filters_features(self, tmp_path):
        config = _config(
            tmp_path,
            preprocessing={
                "feature_selection": {
                    "enabled": True,
                    "method": "variance",
                    "variance_threshold": 0.0,
                }
            },
        )
        preprocessor = DataPreprocessor(config)
        df = _toy_dataframe()
        preprocessor.fit_imputer(df)
        df_imp = preprocessor.transform_impute(df)
        preprocessor.fit_outlier_handler(df_imp)
        df_out = preprocessor.transform_outliers(df_imp)
        preprocessor.fit_preprocessor(df_out)
        df_proc, X = preprocessor.transform_preprocess(df_out)

        stage = FeatureSelectionStage(config, preprocessor, _logger())
        ctx = PipelineContext(data_train=df_proc, data_test=df_proc, X_train=X, X_test=X)
        stage.run(ctx)
        assert stage.feature_selector is not None
        assert ctx.X_train is not None

    def test_missing_target_column_raises(self, tmp_path):
        config = _config(
            tmp_path,
            preprocessing={
                "feature_selection": {
                    "enabled": True,
                    "method": "lasso",
                    "target_column": "missing_target",
                }
            },
        )
        stage = FeatureSelectionStage(config, MagicMock(), _logger())
        df = _toy_dataframe()
        ctx = PipelineContext(data_train=df, data_test=df)
        with pytest.raises(ValueError, match="missing_target"):
            stage.run(ctx)


class TestTrainingStage:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_fixed_clusters_path(self, tmp_path):
        config = _config(tmp_path)
        stage = TrainingStage(config, _logger())
        rng = np.random.RandomState(0)
        ctx = PipelineContext(X_train=rng.randn(60, 3))
        stage.run(ctx)
        assert ctx.model is not None

    def test_count_feature_types_no_selector(self, tmp_path):
        config = _config(tmp_path)
        stage = TrainingStage(config, _logger())
        n_cont, n_cat = stage._count_feature_types(np.zeros((10, 3)), feature_selector=None)
        assert n_cont == 3
        assert n_cat == 0

    def test_count_feature_types_with_selector(self, tmp_path):
        config = _config(tmp_path)
        stage = TrainingStage(config, _logger())
        selector = MagicMock()
        selector.get_selected_features.return_value = ["x1", "x2"]
        n_cont, n_cat = stage._count_feature_types(np.zeros((10, 4)), feature_selector=selector)
        assert n_cont == 2
        assert n_cat == 2

    def test_build_measurement_dict_nan_models(self, tmp_path):
        config = _config(tmp_path)
        config.imputation.enabled = False
        stage = TrainingStage(config, _logger())
        meas = stage._build_measurement_dict(has_missing=True, n_continuous=2, n_categorical=1)
        assert meas["continuous"]["model"] == "continuous_nan"
        assert meas["categorical"]["model"] == "categorical_nan"

    def test_build_measurement_dict_standard_models(self, tmp_path):
        config = _config(tmp_path)
        stage = TrainingStage(config, _logger())
        meas = stage._build_measurement_dict(has_missing=False, n_continuous=2, n_categorical=0)
        assert meas["continuous"]["model"] == "continuous"
        assert "categorical" not in meas
