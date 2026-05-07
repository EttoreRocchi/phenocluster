"""Tests for the evaluation.generalizability submodules."""

import numpy as np
import pandas as pd
import pytest

from phenocluster.evaluation.generalizability.calibration import (
    brier_multiclass,
    compute_calibration_block,
    expected_calibration_error,
    reliability_curve,
)
from phenocluster.evaluation.generalizability.drift import (
    categorical_drift,
    feature_drift,
    population_stability_index,
    top_drifted,
)
from phenocluster.evaluation.generalizability.outcome_concordance import (
    compare_outcomes,
    compare_survival,
    lin_ccc,
)


def _identity_proba(n=60, k=3, seed=0):
    """Posteriors aligned with cycling labels (high diagonal)."""
    rng = np.random.RandomState(seed)
    proba = np.full((n, k), 0.05)
    labels = np.array([i % k for i in range(n)])
    proba[np.arange(n), labels] = 0.9
    proba += rng.uniform(0, 0.01, proba.shape)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba, labels


class TestCalibration:
    def test_brier_multiclass(self):
        proba, labels = _identity_proba()
        out = brier_multiclass(proba, labels)
        assert "per_class" in out
        assert "mean" in out
        assert 0.0 <= out["mean"] <= 1.0

    def test_ece_uniform(self):
        proba, labels = _identity_proba()
        ece = expected_calibration_error(proba, labels, n_bins=5, strategy="uniform")
        assert 0.0 <= ece <= 1.0

    def test_ece_quantile(self):
        proba, labels = _identity_proba()
        ece = expected_calibration_error(proba, labels, n_bins=5, strategy="quantile")
        assert 0.0 <= ece <= 1.0

    def test_ece_unknown_strategy_raises(self):
        proba, labels = _identity_proba()
        with pytest.raises(ValueError):
            expected_calibration_error(proba, labels, strategy="bogus")

    def test_reliability_curve(self):
        proba, labels = _identity_proba()
        curve = reliability_curve(proba, labels, n_bins=5)
        for key in (
            "bin_confidences",
            "bin_accuracies",
            "bin_counts",
            "bin_edges_lo",
            "bin_edges_hi",
        ):
            assert key in curve

    def test_compute_calibration_block(self):
        proba, labels = _identity_proba()
        block = compute_calibration_block(proba, labels)
        assert block["n_bins"] == 10
        assert "brier" in block
        assert "ece" in block

    def test_calibration_block_empty(self):
        assert compute_calibration_block(np.array([]).reshape(0, 0), np.array([])) is None

    def test_invalid_proba_shape(self):
        with pytest.raises(ValueError):
            brier_multiclass(np.array([0.5]), np.array([0]))

    def test_invalid_labels_shape(self):
        with pytest.raises(ValueError):
            brier_multiclass(np.zeros((4, 2)), np.zeros((4, 2)))

    def test_mismatched_rows(self):
        with pytest.raises(ValueError):
            brier_multiclass(np.zeros((5, 2)), np.zeros(4))


class TestDrift:
    def test_psi_no_drift(self):
        rng = np.random.RandomState(0)
        a = rng.randn(200)
        psi, edges = population_stability_index(a, a.copy())
        assert psi == pytest.approx(0.0, abs=0.05)
        assert len(edges) >= 3

    def test_psi_strong_drift(self):
        rng = np.random.RandomState(0)
        a = rng.randn(200)
        b = rng.randn(200) + 5
        psi, _ = population_stability_index(a, b)
        assert psi > 0.5

    def test_psi_empty_returns_nan(self):
        psi, _ = population_stability_index(np.array([]), np.array([1.0]))
        assert np.isnan(psi)

    def test_categorical_drift_returns_chi2(self):
        a = pd.Series(["A"] * 50 + ["B"] * 50)
        b = pd.Series(["A"] * 30 + ["B"] * 70)
        chi2, p, psi = categorical_drift(a, b)
        assert chi2 is not None
        assert 0.0 <= p <= 1.0

    def test_categorical_drift_empty(self):
        a = pd.Series([], dtype="object")
        b = pd.Series(["A"], dtype="object")
        chi2, p, psi = categorical_drift(a, b)
        assert chi2 is None
        assert p is None
        assert np.isnan(psi)

    def test_feature_drift_table(self):
        rng = np.random.RandomState(0)
        deriv = pd.DataFrame(
            {
                "x": rng.randn(80),
                "g": rng.choice(["A", "B"], 80),
            }
        )
        val = pd.DataFrame(
            {
                "x": rng.randn(80) + 0.5,
                "g": rng.choice(["A", "B"], 80, p=[0.3, 0.7]),
            }
        )
        report = feature_drift(deriv, val, ["x"], ["g"])
        assert {"feature", "kind", "psi"}.issubset(set(report.columns))
        assert len(report) == 2

    def test_top_drifted_returns_subset(self):
        rng = np.random.RandomState(0)
        deriv = pd.DataFrame(
            {
                "x1": rng.randn(50),
                "x2": rng.randn(50),
            }
        )
        val = pd.DataFrame(
            {
                "x1": rng.randn(50) + 3,
                "x2": rng.randn(50),
            }
        )
        report = feature_drift(deriv, val, ["x1", "x2"], [])
        top = top_drifted(report, k=1)
        assert len(top) == 1

    def test_top_drifted_empty_passthrough(self):
        empty = pd.DataFrame()
        assert top_drifted(empty, k=5).empty


class TestOutcomeConcordance:
    def test_lin_ccc_perfect(self):
        x = np.array([0.1, 0.5, 1.0])
        assert lin_ccc(x, x.copy()) == pytest.approx(1.0, abs=1e-6)

    def test_lin_ccc_short(self):
        assert np.isnan(lin_ccc(np.array([0.5]), np.array([0.5])))

    def test_lin_ccc_zero_denominator(self):
        x = np.array([0.5, 0.5, 0.5])
        assert np.isnan(lin_ccc(x, x))

    def test_compare_outcomes_returns_summary(self):
        deriv = {
            "mortality": {
                0: {"OR": 1.5, "CI_lower": 1.1, "CI_upper": 2.0},
                1: {"OR": 0.7, "CI_lower": 0.5, "CI_upper": 1.0},
                2: {"OR": 2.5, "CI_lower": 1.5, "CI_upper": 4.1},
            }
        }
        val = {
            "mortality": {
                0: {"OR": 1.4, "CI_lower": 1.0, "CI_upper": 1.9},
                1: {"OR": 0.8, "CI_lower": 0.6, "CI_upper": 1.1},
                2: {"OR": 2.3, "CI_lower": 1.4, "CI_upper": 3.8},
            }
        }
        out = compare_outcomes(deriv, val)
        assert "mortality" in out
        block = out["mortality"]
        assert "summary" in block and "per_phenotype" in block
        assert isinstance(block["per_phenotype"], list)
        assert len(block["per_phenotype"]) == 3

    def test_compare_outcomes_skip_invalid(self):
        deriv = {
            "x": {0: {"OR": -1.0, "CI_lower": -2, "CI_upper": 0}},
        }
        val = {
            "x": {0: {"OR": 1.0, "CI_lower": 0.5, "CI_upper": 2.0}},
        }
        out = compare_outcomes(deriv, val)
        assert out["x"]["per_phenotype"] == []

    def test_compare_survival_uses_hr(self):
        deriv = {
            "ms": {
                0: {"HR": 1.0, "CI_lower": 0.6, "CI_upper": 1.6},
                1: {"HR": 2.0, "CI_lower": 1.2, "CI_upper": 3.2},
            }
        }
        val = {
            "ms": {
                0: {"HR": 1.1, "CI_lower": 0.7, "CI_upper": 1.7},
                1: {"HR": 1.8, "CI_lower": 1.0, "CI_upper": 3.0},
            }
        }
        out = compare_survival(deriv, val)
        assert "ms" in out


class TestRefitValidator:
    def test_hungarian_alignment_exact_match(self):
        from phenocluster.evaluation.generalizability.refit_validator import (
            hungarian_alignment,
        )

        deriv = np.array([0, 0, 1, 1, 2, 2])
        refit = np.array([0, 0, 1, 1, 2, 2])
        out = hungarian_alignment(deriv, refit, 3, 3)
        assert out["mapping"] == {0: 0, 1: 1, 2: 2}
        assert list(out["aligned_labels"]) == [0, 0, 1, 1, 2, 2]

    def test_hungarian_alignment_permutation(self):
        from phenocluster.evaluation.generalizability.refit_validator import (
            hungarian_alignment,
        )

        deriv = np.array([0, 0, 1, 1, 2, 2])
        refit = np.array([2, 2, 0, 0, 1, 1])
        out = hungarian_alignment(deriv, refit, 3, 3)
        assert list(out["aligned_labels"]) == [0, 0, 1, 1, 2, 2]

    def test_hungarian_alignment_padded(self):
        from phenocluster.evaluation.generalizability.refit_validator import (
            hungarian_alignment,
        )

        deriv = np.array([0, 0, 1, 1])
        refit = np.array([0, 1, 0, 1])
        out = hungarian_alignment(deriv, refit, 2, 3)
        assert "unmatched_validation_clusters" in out

    def test_to_json_safe_strips_arrays(self):
        from phenocluster.evaluation.generalizability.refit_validator import (
            to_json_safe,
        )

        result = {
            "refit_labels_raw": np.array([0, 1]),
            "refit_proba": np.zeros((2, 2)),
            "aligned_labels": np.array([0, 1]),
            "ari": 0.8,
        }
        out = to_json_safe(result)
        assert "refit_labels_raw" not in out
        assert "refit_proba" not in out
        assert "aligned_labels" not in out
        assert out["ari"] == 0.8

    def test_to_json_safe_passthrough_empty(self):
        from phenocluster.evaluation.generalizability.refit_validator import (
            to_json_safe,
        )

        assert to_json_safe({}) == {}

    def test_refit_skipped_for_small_cohort(self):
        from unittest.mock import MagicMock

        from phenocluster.evaluation.generalizability.refit_validator import (
            refit_and_match,
        )

        log = MagicMock()
        out = refit_and_match(
            X_val=np.zeros((5, 3)),
            derivation_labels_on_val=np.zeros(5, dtype=int),
            reference_model=MagicMock(),
            n_clusters=2,
            min_validation_size_for_refit=10,
            logger=log,
        )
        assert out is None
        log.warning.assert_called()


class TestApplyValidatorSafeWrapper:
    def test_safe_apply_returns_none_on_failure(self):
        from unittest.mock import MagicMock

        from phenocluster.evaluation.generalizability.apply_validator import (
            safe_apply_to_cohort,
        )

        log = MagicMock()
        bad = MagicMock()
        bad.transform_impute.side_effect = RuntimeError("boom")
        out = safe_apply_to_cohort(
            pd.DataFrame({"x": [1.0]}),
            logger=log,
            label="x",
            model=MagicMock(),
            preprocessor=bad,
            feature_selector=None,
            continuous_columns=["x"],
            categorical_columns=[],
        )
        assert out is None
        log.warning.assert_called()
