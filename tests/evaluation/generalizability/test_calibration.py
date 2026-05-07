"""Unit tests for calibration metrics on synthetic posteriors."""

import numpy as np

from phenocluster.evaluation.generalizability.calibration import (
    brier_multiclass,
    compute_calibration_block,
    expected_calibration_error,
    reliability_curve,
)


def _well_calibrated_proba(rng, n=600):
    """Generate proba where the predicted-class confidence equals empirical accuracy."""
    proba = np.zeros((n, 3))
    confidences = rng.uniform(0.4, 0.99, size=n)
    classes = rng.integers(0, 3, size=n)
    correct = rng.uniform(size=n) < confidences
    labels = np.where(correct, classes, (classes + 1 + rng.integers(0, 2, size=n)) % 3)
    for i in range(n):
        residual = (1.0 - confidences[i]) / 2.0
        for k in range(3):
            proba[i, k] = confidences[i] if k == classes[i] else residual
    return proba, labels


def test_well_calibrated_low_ece():
    rng = np.random.default_rng(0)
    proba, labels = _well_calibrated_proba(rng, n=2000)
    ece = expected_calibration_error(proba, labels, n_bins=10)
    assert ece < 0.05


def test_overconfident_high_ece():
    rng = np.random.default_rng(1)
    n = 1000
    proba = np.zeros((n, 2))
    proba[:, 0] = 0.99
    proba[:, 1] = 0.01
    labels = (rng.uniform(size=n) < 0.5).astype(int)
    ece = expected_calibration_error(proba, labels, n_bins=10)
    assert ece > 0.3


def test_brier_multiclass_shape():
    rng = np.random.default_rng(2)
    proba = np.eye(3)[rng.integers(0, 3, size=50)]
    labels = rng.integers(0, 3, size=50)
    out = brier_multiclass(proba, labels)
    assert set(out["per_class"].keys()) == {"0", "1", "2"}
    assert out["mean"] >= 0.0


def test_reliability_curve_keys():
    rng = np.random.default_rng(3)
    proba, labels = _well_calibrated_proba(rng, n=300)
    curve = reliability_curve(proba, labels, n_bins=5)
    assert set(curve.keys()) >= {"bin_confidences", "bin_accuracies", "bin_counts"}
    assert sum(curve["bin_counts"]) == len(labels)


def test_compute_calibration_block_returns_none_for_empty():
    assert compute_calibration_block(np.empty((0, 2)), np.array([], dtype=int)) is None


def test_compute_calibration_block_flags_degenerate_quantile_bins():
    """Apply-only mode produces near-constant top-class confidences. The
    quantile binning collapses; the block must record the fallback so the
    cohort report can warn the user (finding #14).
    """
    n = 200
    proba = np.zeros((n, 2))
    proba[:, 0] = 1.0
    labels = np.zeros(n, dtype=int)
    block = compute_calibration_block(proba, labels, n_bins=10, strategy="quantile")
    assert block is not None
    assert block["degenerate_quantile_bins"] is True
    # ECE remains finite under the uniform fallback.
    assert np.isfinite(block["ece"])


def test_compute_calibration_block_no_flag_for_healthy_distribution():
    rng = np.random.default_rng(42)
    proba, labels = _well_calibrated_proba(rng, n=500)
    block = compute_calibration_block(proba, labels, n_bins=10, strategy="quantile")
    assert block is not None
    assert block["degenerate_quantile_bins"] is False
