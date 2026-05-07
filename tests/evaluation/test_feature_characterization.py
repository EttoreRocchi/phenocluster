"""Unit tests for the FeatureCharacterizer effect-size helpers."""

import numpy as np
import pandas as pd
from scipy import stats

from phenocluster.config import PhenoClusterConfig
from phenocluster.evaluation.feature_characterization import FeatureCharacterizer


def _make_config(continuous=("x",), categorical=()):
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {
                "continuous_columns": list(continuous),
                "categorical_columns": list(categorical),
                "split": {},
            },
            "preprocessing": {},
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
            "inference": {"enabled": True, "fdr_correction": False},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


class TestHedgesGStar:
    def test_near_zero_for_random_labels(self):
        cfg = _make_config()
        fc = FeatureCharacterizer(cfg, n_clusters=2)
        rng = np.random.default_rng(0)
        x = rng.normal(size=4000)
        df = pd.DataFrame({"x": x})
        labels = rng.integers(0, 2, size=4000)
        res = fc.compute_feature_importance(df, labels)
        assert abs(res["continuous"]["x"][0]["effect_size"]) < 0.1

    def test_sign_matches_direction(self):
        cfg = _make_config()
        fc = FeatureCharacterizer(cfg, n_clusters=2)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"x": np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(2.0, 1.0, 100)])}
        )
        labels = np.array([0] * 100 + [1] * 100)
        res = fc.compute_feature_importance(df, labels)
        e0 = res["continuous"]["x"][0]
        e1 = res["continuous"]["x"][1]
        assert e0["effect_size"] < 0
        assert e1["effect_size"] > 0
        assert e0["direction"] == "lower"
        assert e1["direction"] == "higher"


class TestCramersV:
    def test_2x2_equals_phi_no_yates(self):
        cat = np.array(["a", "a", "b", "b", "a", "b", "a", "b", "a", "b"])
        mask = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0]).astype(bool)
        v, _ = FeatureCharacterizer._compute_cramers_v(cat, mask)
        contingency = pd.crosstab(cat, mask.astype(int)).values
        chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
        expected = np.sqrt(chi2 / len(cat))
        assert np.isclose(v, expected, rtol=1e-6)

    def test_returns_zero_for_empty(self):
        v, p = FeatureCharacterizer._compute_cramers_v(
            np.array([], dtype=object), np.array([], dtype=bool)
        )
        assert v == 0.0
        assert p is None
