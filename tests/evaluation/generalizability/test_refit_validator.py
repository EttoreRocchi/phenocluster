"""Unit tests for the Hungarian alignment helper used by refit_and_match."""

import numpy as np

from phenocluster.evaluation.generalizability.refit_validator import (
    hungarian_alignment,
    to_json_safe,
)


def test_permutation_recovered_via_hungarian():
    rng = np.random.default_rng(0)
    deriv = rng.integers(0, 3, size=400)
    permutation = {0: 2, 1: 0, 2: 1}
    refit = np.array([permutation[label] for label in deriv])
    out = hungarian_alignment(deriv, refit, n_deriv_clusters=3, n_val_clusters=3)
    assert (out["aligned_labels"] == deriv).all()
    assert out["unmatched_derivation_clusters"] == []
    assert out["unmatched_validation_clusters"] == []
    inverse = {v: k for k, v in permutation.items()}
    assert out["mapping"] == inverse


def test_extra_validation_cluster_marked_unmatched():
    rng = np.random.default_rng(1)
    deriv = rng.integers(0, 3, size=300)
    refit = deriv.copy()
    refit[:50] = 3
    out = hungarian_alignment(deriv, refit, n_deriv_clusters=3, n_val_clusters=4)
    assert 3 in out["unmatched_validation_clusters"]
    assert out["unmatched_derivation_clusters"] == []


def test_to_json_safe_strips_arrays():
    payload = {
        "ari": 0.9,
        "nmi": 0.85,
        "matched_accuracy": 0.95,
        "refit_labels_raw": np.array([0, 1, 2]),
        "refit_proba": np.zeros((3, 3)),
        "aligned_labels": np.array([0, 1, 2]),
        "mapping": {0: 0, 1: 1, 2: 2},
        "unmatched_derivation_clusters": [],
        "unmatched_validation_clusters": [],
    }
    out = to_json_safe(payload)
    assert "refit_labels_raw" not in out
    assert "refit_proba" not in out
    assert "aligned_labels" not in out
    assert out["ari"] == 0.9
