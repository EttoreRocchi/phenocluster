"""Tests for the categorical encoder fit/transform."""

import warnings

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import PhenoClusterConfig
from phenocluster.data.encoder import Encoder


def _config(method, tmp_path, categorical=None, continuous=None, handle_unknown="ignore"):
    """Build an encoder-friendly PhenoClusterConfig with the requested method."""
    return PhenoClusterConfig.from_dict(
        {
            "global": {
                "project_name": "enc",
                "output_dir": str(tmp_path),
                "random_state": 42,
            },
            "data": {
                "continuous_columns": continuous or ["x"],
                "categorical_columns": categorical or ["cat"],
                "split": {},
            },
            "preprocessing": {
                "categorical_encoding": {
                    "method": method,
                    "handle_unknown": handle_unknown,
                }
            },
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


class TestEncoderLabel:
    def test_fit_and_transform(self, tmp_path):
        cfg = _config("label", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "B", "A"]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        assert "cat_encoded" in out.columns
        assert not out["cat_encoded"].isna().any()

    def test_unknown_category_mapped_to_mode(self, tmp_path):
        cfg = _config("label", tmp_path)
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "A", "B"]})
        test = pd.DataFrame({"x": [4.0], "cat": ["C"]})
        enc = Encoder(cfg)
        enc.fit(train)
        out = enc.transform(test)
        assert pd.notna(out["cat_encoded"].iloc[0])

    def test_nan_preserved(self, tmp_path):
        cfg = _config("label", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "B", np.nan]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        assert np.isnan(out["cat_encoded"].iloc[2])


class TestEncoderOnehot:
    def test_fit_creates_per_category_columns(self, tmp_path):
        cfg = _config("onehot", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "B", "A"]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        assert "cat_A" in out.columns
        assert "cat_B" in out.columns

    def test_nan_becomes_missing_category(self, tmp_path):
        cfg = _config("onehot", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0], "cat": ["A", np.nan]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        assert "cat__MISSING" in out.columns


class TestEncoderFrequency:
    def test_fit_then_transform(self, tmp_path):
        cfg = _config("frequency", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "cat": ["A", "A", "A", "B"]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        assert "cat_encoded" in out.columns
        assert out.loc[0, "cat_encoded"] == pytest.approx(0.75)

    def test_unknown_category_warns(self, tmp_path):
        cfg = _config("frequency", tmp_path)
        train = pd.DataFrame({"x": [1.0, 2.0], "cat": ["A", "A"]})
        test = pd.DataFrame({"x": [3.0], "cat": ["Z"]})
        enc = Encoder(cfg)
        enc.fit(train)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = enc.transform(test)
        assert any("unknown" in str(w.message).lower() for w in caught)
        assert out["cat_encoded"].iloc[0] == 0.0


class TestEncoderNotFitted:
    def test_transform_before_fit_raises(self, tmp_path):
        cfg = _config("label", tmp_path)
        enc = Encoder(cfg)
        with pytest.raises(RuntimeError):
            enc.transform(pd.DataFrame({"x": [1.0], "cat": ["A"]}))


class TestGetFeatureMatrix:
    def test_label(self, tmp_path):
        cfg = _config("label", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0], "cat": ["A", "B"]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        mat = enc.get_feature_matrix(out)
        assert mat.shape == (2, 2)

    def test_onehot(self, tmp_path):
        cfg = _config("onehot", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0], "cat": ["A", "B"]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        mat = enc.get_feature_matrix(out)
        assert mat.shape[1] >= 3

    def test_frequency(self, tmp_path):
        cfg = _config("frequency", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0], "cat": ["A", "B"]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        mat = enc.get_feature_matrix(out)
        assert mat.shape == (2, 2)


class TestStripCategoricals:
    def test_whitespace_and_quotes_stripped(self, tmp_path):
        cfg = _config("label", tmp_path)
        df = pd.DataFrame({"x": [1.0, 2.0], "cat": ["  'A' ", '"B"']})
        enc = Encoder(cfg)
        enc.fit(df)
        assert "A" in enc.label_encoders["cat"].classes_
        assert "B" in enc.label_encoders["cat"].classes_


class TestEncoderNoCategorical:
    def test_skips_when_no_categorical(self, tmp_path):
        cfg = _config("label", tmp_path, categorical=[], continuous=["x"])
        df = pd.DataFrame({"x": [1.0, 2.0]})
        enc = Encoder(cfg)
        enc.fit(df)
        out = enc.transform(df)
        assert list(out.columns) == ["x"]
