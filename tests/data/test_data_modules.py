"""Tests for data processing modules (encoder, imputer, outlier, splitter)."""

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import PhenoClusterConfig


def _make_config(**overrides):
    base = {
        "global": {"project_name": "test", "random_state": 42},
        "data": {
            "continuous_columns": ["c1", "c2", "c3"],
            "categorical_columns": ["cat1"],
            "split": {"test_size": 0.3},
        },
        "preprocessing": {
            "categorical_encoding": {"method": "label"},
            "imputation": {"enabled": True, "method": "iterative"},
            "outlier": {"enabled": True, "method": "winsorize", "winsorize_limits": [0.05, 0.05]},
        },
        "model": {"n_clusters": 2},
        "outcome": {"enabled": False},
        "logging": {"level": "WARNING", "log_to_file": False},
    }
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base:
            base[k].update(v)
        else:
            base[k] = v
    return PhenoClusterConfig.from_dict(base)


def _sample_df(n=60, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "c1": rng.randn(n),
            "c2": rng.randn(n) * 2 + 5,
            "c3": rng.randn(n),
            "cat1": rng.choice(["A", "B", "C"], n),
        }
    )


class TestEncoder:
    def test_fit_label_encoding(self):
        from phenocluster.data.encoder import Encoder

        cfg = _make_config()
        enc = Encoder(cfg)
        df = _sample_df()
        enc.fit(df)
        assert enc.feature_columns is not None

    def test_transform_label_encoding(self):
        from phenocluster.data.encoder import Encoder

        cfg = _make_config()
        enc = Encoder(cfg)
        df = _sample_df()
        enc.fit(df)
        result = enc.transform(df)
        assert isinstance(result, pd.DataFrame)
        # transform should return a processed DataFrame
        assert len(result) == len(df)

    def test_fit_onehot_encoding(self):
        from phenocluster.data.encoder import Encoder

        cfg = _make_config(
            preprocessing={
                "categorical_encoding": {"method": "onehot"},
                "imputation": {"enabled": False},
                "outlier": {"enabled": False},
            }
        )
        enc = Encoder(cfg)
        df = _sample_df()
        enc.fit(df)
        result = enc.transform(df)
        # One-hot should create multiple columns
        assert result.shape[1] >= 5  # 3 continuous + at least 2 one-hot

    def test_fit_no_categoricals(self):
        from phenocluster.data.encoder import Encoder

        cfg = _make_config(
            data={
                "continuous_columns": ["c1", "c2"],
                "categorical_columns": [],
                "split": {"test_size": 0.3},
            }
        )
        enc = Encoder(cfg)
        df = _sample_df()[["c1", "c2"]]
        enc.fit(df)
        result = enc.transform(df)
        assert list(result.columns) == ["c1", "c2"]

    def test_get_feature_matrix(self):
        from phenocluster.data.encoder import Encoder

        cfg = _make_config()
        enc = Encoder(cfg)
        df = _sample_df()
        enc.fit(df)
        X = enc.get_feature_matrix(df)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(df)


class TestImputer:
    def test_fit_iterative(self):
        from phenocluster.data.imputer import Imputer

        cfg = _make_config()
        imp = Imputer(cfg)
        df = _sample_df()
        # Insert some missing values
        df.loc[0, "c1"] = np.nan
        df.loc[5, "c2"] = np.nan
        imp.fit(df)
        result = imp.transform(df)
        assert result["c1"].isna().sum() == 0
        assert result["c2"].isna().sum() == 0

    def test_no_missing_skips(self):
        from phenocluster.data.imputer import Imputer

        cfg = _make_config()
        imp = Imputer(cfg)
        df = _sample_df()
        imp.fit(df)
        result = imp.transform(df)
        pd.testing.assert_frame_equal(result, df)

    def test_disabled_returns_unchanged(self):
        from phenocluster.data.imputer import Imputer

        cfg = _make_config(
            preprocessing={
                "imputation": {"enabled": False},
                "categorical_encoding": {"method": "label"},
                "outlier": {"enabled": False},
            }
        )
        imp = Imputer(cfg)
        df = _sample_df()
        df.loc[0, "c1"] = np.nan
        imp.fit(df)
        result = imp.transform(df)
        assert np.isnan(result.loc[0, "c1"])

    def test_fit_knn(self):
        from phenocluster.data.imputer import Imputer

        cfg = _make_config(
            preprocessing={
                "imputation": {"enabled": True, "method": "knn"},
                "categorical_encoding": {"method": "label"},
                "outlier": {"enabled": False},
            }
        )
        imp = Imputer(cfg)
        df = _sample_df()
        df.loc[0, "c1"] = np.nan
        imp.fit(df)
        result = imp.transform(df)
        assert result["c1"].isna().sum() == 0


class TestOutlierHandler:
    def test_winsorize_fit_transform(self):
        from phenocluster.data.outlier_handler import OutlierHandler

        cfg = _make_config()
        handler = OutlierHandler(cfg)
        df = _sample_df()
        # Add extreme outlier
        df.loc[0, "c1"] = 100.0
        handler.fit(df)
        result = handler.transform(df)
        # After winsorization, extreme value should be clipped
        assert result.loc[0, "c1"] < 100.0

    def test_disabled_returns_unchanged(self):
        from phenocluster.data.outlier_handler import OutlierHandler

        cfg = _make_config(
            preprocessing={
                "outlier": {"enabled": False},
                "categorical_encoding": {"method": "label"},
                "imputation": {"enabled": False},
            }
        )
        handler = OutlierHandler(cfg)
        df = _sample_df()
        df.loc[0, "c1"] = 100.0
        handler.fit(df)
        result = handler.transform(df)
        assert result.loc[0, "c1"] == 100.0

    def test_isolation_forest(self):
        from phenocluster.data.outlier_handler import OutlierHandler

        cfg = _make_config(
            preprocessing={
                "outlier": {"enabled": True, "method": "isolation_forest", "contamination": 0.1},
                "categorical_encoding": {"method": "label"},
                "imputation": {"enabled": False},
            }
        )
        handler = OutlierHandler(cfg)
        df = _sample_df(n=100)
        handler.fit(df)
        result = handler.transform(df)
        assert isinstance(result, pd.DataFrame)


class TestDataSplitter:
    def test_split_basic_sizes(self):
        from phenocluster.data.splitter import DataSplitter

        cfg = _make_config()
        splitter = DataSplitter(cfg.data_split)
        df = _sample_df(n=100)
        result = splitter.split(df)
        assert len(result.train) + len(result.test) == 100
        assert len(result.test) == pytest.approx(30, abs=5)

    def test_split_stratified(self):
        from phenocluster.data.splitter import DataSplitter

        cfg = _make_config(
            data={
                "continuous_columns": ["c1"],
                "categorical_columns": [],
                "split": {"test_size": 0.3, "stratify_by": "group"},
            }
        )
        splitter = DataSplitter(cfg.data_split)
        df = _sample_df(n=100)
        df["group"] = np.array([i % 2 for i in range(100)])
        result = splitter.split(df, stratify_column="group")
        assert len(result.train) + len(result.test) == 100

    def test_split_returns_indices(self):
        from phenocluster.data.splitter import DataSplitter

        cfg = _make_config()
        splitter = DataSplitter(cfg.data_split)
        df = _sample_df(n=100)
        result = splitter.split(df)
        assert result.train_indices is not None
        assert result.test_indices is not None
        assert len(result.train_indices) == len(result.train)
