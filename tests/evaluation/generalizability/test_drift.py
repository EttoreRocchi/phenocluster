"""Unit tests for distribution drift helpers."""

import numpy as np
import pandas as pd

from phenocluster.evaluation.generalizability.drift import (
    categorical_drift,
    feature_drift,
    population_stability_index,
    top_drifted,
)


def test_psi_identical_distribution_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    b = rng.normal(size=2000)
    psi, _ = population_stability_index(a, b, n_bins=10)
    assert psi < 0.05


def test_psi_shifted_distribution_large():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, size=2000)
    b = rng.normal(loc=2.0, size=2000)
    psi, _ = population_stability_index(a, b, n_bins=10)
    assert psi > 0.25


def test_categorical_drift_identical_p_above_threshold():
    rng = np.random.default_rng(2)
    cats = rng.choice(["A", "B", "C"], size=1000, p=[0.5, 0.3, 0.2])
    a = pd.Series(cats[:500])
    b = pd.Series(cats[500:])
    chi2_stat, chi2_p, psi = categorical_drift(a, b)
    assert chi2_p is not None and chi2_p > 0.05
    assert psi < 0.05


def test_categorical_drift_shifted_low_p():
    a = pd.Series(np.repeat(["A", "B"], [400, 100]))
    b = pd.Series(np.repeat(["A", "B"], [100, 400]))
    chi2_stat, chi2_p, psi = categorical_drift(a, b)
    assert chi2_p is not None and chi2_p < 0.001
    assert psi > 0.5


def test_feature_drift_table_columns():
    rng = np.random.default_rng(3)
    deriv = pd.DataFrame(
        {
            "x": rng.normal(size=500),
            "y": rng.normal(size=500),
            "g": rng.choice(["A", "B"], size=500),
        }
    )
    val = pd.DataFrame(
        {
            "x": rng.normal(loc=1.5, size=300),
            "y": rng.normal(size=300),
            "g": rng.choice(["A", "B"], size=300),
        }
    )
    out = feature_drift(deriv, val, continuous_cols=["x", "y"], categorical_cols=["g"])
    expected_cols = {
        "feature",
        "kind",
        "psi",
        "ks_stat",
        "ks_p",
        "chi2_stat",
        "chi2_p",
        "n_deriv",
        "n_val",
        "missing_diff",
    }
    assert expected_cols.issubset(set(out.columns))
    assert set(out["feature"]) == {"x", "y", "g"}
    psi_x = float(out.loc[out["feature"] == "x", "psi"].iloc[0])
    psi_y = float(out.loc[out["feature"] == "y", "psi"].iloc[0])
    assert psi_x > psi_y


def test_top_drifted_returns_top_k_by_abs():
    df = pd.DataFrame({"feature": list("abcd"), "psi": [0.1, 0.5, -0.3, 0.05]})
    top = top_drifted(df, k=2, by="psi")
    assert list(top["feature"]) == ["b", "c"]


def test_psi_exact_match_is_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    psi, _ = population_stability_index(x.copy(), x.copy(), n_bins=10)
    assert psi < 1e-9


def test_psi_empty_bin_bounded_by_eps():
    rng = np.random.default_rng(0)
    deriv = rng.uniform(0, 10, size=1000)
    val = rng.uniform(0, 9, size=1000)
    psi, _ = population_stability_index(deriv, val, n_bins=10)
    assert 0 < psi < 0.5
