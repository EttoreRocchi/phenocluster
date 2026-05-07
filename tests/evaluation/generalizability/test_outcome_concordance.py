"""Unit tests for outcome-association concordance."""

import numpy as np

from phenocluster.evaluation.generalizability.outcome_concordance import (
    DEFAULT_Z,
    compare_outcomes,
    compare_survival,
    lin_ccc,
)


def _make_outcome_block(point: float, ratio_to_se: float = 6.0):
    """Build a single phenotype result whose 95% Wald CI implies a known SE."""
    log_p = float(np.log(point))
    se = abs(log_p) / max(ratio_to_se, 1e-3) if log_p != 0 else 0.1
    ci_lo = float(np.exp(log_p - DEFAULT_Z * se))
    ci_hi = float(np.exp(log_p + DEFAULT_Z * se))
    return {"OR": float(point), "CI_lower": ci_lo, "CI_upper": ci_hi, "p_value": 0.01}


def test_perfect_concordance_high_ccc():
    points = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    deriv_outcome = {f"p{i}": _make_outcome_block(p) for i, p in enumerate(points)}
    val_outcome = {f"p{i}": _make_outcome_block(p) for i, p in enumerate(points)}
    deriv = {"death": deriv_outcome}
    val = {"death": val_outcome}
    result = compare_outcomes(deriv, val)
    summary = result["death"]["summary"]
    assert summary["pearson_r"] > 0.99
    assert summary["lin_ccc"] > 0.99
    assert summary["sign_agreement"] == 1.0


def test_anti_correlated_negative_pearson():
    points_d = [0.5, 0.8, 1.5, 2.0]
    points_v = [2.0, 1.5, 0.8, 0.5]
    deriv = {"death": {f"p{i}": _make_outcome_block(p) for i, p in enumerate(points_d)}}
    val = {"death": {f"p{i}": _make_outcome_block(p) for i, p in enumerate(points_v)}}
    summary = compare_outcomes(deriv, val)["death"]["summary"]
    assert summary["pearson_r"] < -0.9


def test_per_phenotype_delta_test_keys_and_fdr():
    points_d = [0.5, 0.8, 1.5, 2.0]
    points_v = [0.55, 0.82, 1.4, 2.1]
    deriv = {"death": {f"p{i}": _make_outcome_block(p) for i, p in enumerate(points_d)}}
    val = {"death": {f"p{i}": _make_outcome_block(p) for i, p in enumerate(points_v)}}
    rows = compare_outcomes(deriv, val)["death"]["per_phenotype"]
    assert len(rows) == 4
    for row in rows:
        assert {"phenotype", "delta_log_effect", "p_value", "p_value_fdr"}.issubset(row)


def test_lin_ccc_simple_cases():
    assert lin_ccc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) > 0.99
    assert lin_ccc(np.array([1.0, 2.0]), np.array([2.0, 1.0])) < 0.0


def test_compare_survival_uses_HR_key():
    deriv = {
        "los": {
            "1_vs_0": {"HR": 1.5, "CI_lower": 1.2, "CI_upper": 1.8, "p_value": 0.01},
            "2_vs_0": {"HR": 0.5, "CI_lower": 0.3, "CI_upper": 0.8, "p_value": 0.01},
        }
    }
    val = {
        "los": {
            "1_vs_0": {"HR": 1.4, "CI_lower": 1.0, "CI_upper": 1.9, "p_value": 0.04},
            "2_vs_0": {"HR": 0.55, "CI_lower": 0.3, "CI_upper": 0.9, "p_value": 0.02},
        }
    }
    summary = compare_survival(deriv, val)["los"]["summary"]
    assert summary["pearson_r"] > 0.95
    assert summary["sign_agreement"] == 1.0


def test_wald_p_value_does_not_underflow_for_large_z():
    deriv = {"y": {0: {"OR": 1.0, "CI_lower": 0.95, "CI_upper": 1.05, "p_value": 0.5}}}
    val = {"y": {0: {"OR": 5.0, "CI_lower": 4.0, "CI_upper": 6.0, "p_value": 1e-30}}}
    out = compare_outcomes(deriv, val)
    p = out["y"]["per_phenotype"][0]["p_value"]
    assert p > 0.0
    assert p < 1e-30
    assert np.isfinite(p)
