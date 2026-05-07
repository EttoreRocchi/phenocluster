"""Unit tests for the prevalence helpers."""

import numpy as np

from phenocluster.evaluation.generalizability.prevalence import (
    chi2_cohort_comparison,
    chi2_homogeneity,
    cluster_distribution,
)


def test_cluster_distribution_basic():
    labels = np.array([0, 0, 0, 1, 1, 2])
    dist = cluster_distribution(labels)
    assert dist[0]["count"] == 3
    assert dist[1]["count"] == 2
    assert dist[2]["count"] == 1
    assert dist[0]["percentage"] == 50.0


def test_cluster_distribution_empty():
    assert cluster_distribution(np.array([], dtype=int)) == {}


def test_chi2_cohort_comparison_returns_p_value():
    deriv = {0: {"n_positive": 30, "n_total": 100}, 1: {"n_positive": 20, "n_total": 80}}
    val = {0: {"n_positive": 35, "n_total": 90}, 1: {"n_positive": 18, "n_total": 70}}
    result = chi2_cohort_comparison(deriv, val)
    assert result is not None
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["statistic"] >= 0.0


def test_chi2_cohort_comparison_no_overlap():
    deriv = {0: {"n_positive": 5, "n_total": 10}}
    val = {1: {"n_positive": 5, "n_total": 10}}
    assert chi2_cohort_comparison(deriv, val) is None


def test_chi2_cohort_comparison_zero_totals():
    deriv = {0: {"n_positive": 0, "n_total": 0}}
    val = {0: {"n_positive": 0, "n_total": 0}}
    assert chi2_cohort_comparison(deriv, val) is None


def test_chi2_homogeneity_detects_drift():
    deriv = {0: {"count": 100}, 1: {"count": 60}, 2: {"count": 40}}
    val = {0: {"count": 50}, 1: {"count": 40}, 2: {"count": 60}}
    res = chi2_homogeneity(deriv, val)
    assert res is not None
    assert res["p_value"] < 0.01
    assert res["statistic"] > 0
    assert res["df"] == 2


def test_chi2_homogeneity_identical_distributions():
    same = {0: {"count": 50}, 1: {"count": 30}, 2: {"count": 20}}
    res = chi2_homogeneity(same, same)
    assert res is not None
    assert res["statistic"] == 0.0
    assert res["p_value"] == 1.0


def test_chi2_homogeneity_handles_no_overlap():
    deriv = {0: {"count": 10}}
    val = {1: {"count": 10}}
    assert chi2_homogeneity(deriv, val) is None


def test_chi2_homogeneity_handles_empty():
    assert chi2_homogeneity({}, {}) is None
