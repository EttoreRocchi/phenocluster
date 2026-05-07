"""
PhenoCluster Posterior Calibration
==================================

Brier score, expected calibration error and reliability curve helpers for
posterior probabilities produced by a fitted StepMix model on a validation
cohort.

These metrics are only meaningful when the validation cohort has been
re-fit and aligned: the predicted-class posterior is then compared to the
empirical agreement with the refit-and-aligned labels (proxy ground truth).
In apply-only mode (no refit) the predicted class always equals
``argmax(proba)`` and the metrics are degenerate; callers should gate
accordingly.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def _validate_inputs(proba: np.ndarray, labels: np.ndarray) -> None:
    if proba.ndim != 2:
        raise ValueError("proba must be a 2D array of shape (n_samples, n_classes)")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array of shape (n_samples,)")
    if proba.shape[0] != labels.shape[0]:
        raise ValueError("proba and labels must have the same number of rows")


def brier_multiclass(proba: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """One-vs-rest Brier score per class plus the unweighted mean.

    The Brier score for class ``k`` is the mean squared error between
    ``proba[:, k]`` and the indicator ``labels == k``.
    """
    _validate_inputs(proba, labels)
    n_classes = proba.shape[1]
    per_class = {}
    scores = []
    for k in range(n_classes):
        target = (labels == k).astype(float)
        bs = float(np.mean((proba[:, k] - target) ** 2))
        per_class[str(k)] = bs
        scores.append(bs)
    return {"per_class": per_class, "mean": float(np.mean(scores))}


def expected_calibration_error(
    proba: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Top-label expected calibration error (ECE).

    Parameters
    ----------
    proba : np.ndarray
        Posterior matrix of shape ``(n_samples, n_classes)``.
    labels : np.ndarray
        Reference labels (typically refit-aligned).
    n_bins : int, default 10
        Number of bins.
    strategy : {"quantile", "uniform"}, default "quantile"
        ``"quantile"`` uses equal-mass bins on the predicted-class posterior
        (robust to skewed distributions). ``"uniform"`` uses equal-width bins
        on ``[0, 1]``.
    """
    _validate_inputs(proba, labels)
    pred = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(labels)), pred]
    correct = (pred == labels).astype(float)

    edges, _ = _bin_edges(confidence, n_bins, strategy)
    bin_ids = np.digitize(confidence, edges[1:-1], right=True)

    total = float(len(confidence))
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        weight = float(mask.sum()) / total
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += weight * abs(avg_conf - avg_acc)
    return float(ece)


def reliability_curve(
    proba: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Dict[str, List[float]]:
    """Top-label reliability curve data (per-bin confidence, accuracy, count)."""
    _validate_inputs(proba, labels)
    pred = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(labels)), pred]
    correct = (pred == labels).astype(float)

    edges, _ = _bin_edges(confidence, n_bins, strategy)
    bin_ids = np.digitize(confidence, edges[1:-1], right=True)

    bin_conf: List[float] = []
    bin_acc: List[float] = []
    bin_count: List[int] = []
    bin_lo: List[float] = []
    bin_hi: List[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        n = int(mask.sum())
        bin_count.append(n)
        bin_lo.append(float(edges[b]))
        bin_hi.append(float(edges[b + 1]))
        if n == 0:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
            continue
        bin_conf.append(float(confidence[mask].mean()))
        bin_acc.append(float(correct[mask].mean()))

    return {
        "bin_confidences": bin_conf,
        "bin_accuracies": bin_acc,
        "bin_counts": bin_count,
        "bin_edges_lo": bin_lo,
        "bin_edges_hi": bin_hi,
    }


def _bin_edges(values: np.ndarray, n_bins: int, strategy: str) -> Tuple[np.ndarray, bool]:
    """Return ``(edges, fell_back)``.

    ``fell_back`` is ``True`` when quantile binning collapsed (degenerate
    distribution: all-equal confidences or fewer than three distinct
    quantile edges) and the function returned uniform ``[0, 1]`` bins
    instead. Callers can attach this signal to a cohort warning so
    downstream consumers (dashboard, report) understand why the reliability
    curve looks flat.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1), False
    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(values, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
        if len(edges) < 3 or edges[-1] - edges[0] < 1e-9:
            return np.linspace(0.0, 1.0, n_bins + 1), True
        if len(edges) < n_bins + 1:
            edges = np.linspace(edges[0], edges[-1], n_bins + 1)
        return edges, False
    raise ValueError(f"Unknown calibration binning strategy '{strategy}'")


def compute_calibration_block(
    proba: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Optional[Dict[str, object]]:
    """Convenience: bundle Brier, ECE and reliability into one dict.

    Returns ``None`` when ``proba`` or ``labels`` are empty. The
    ``degenerate_quantile_bins`` field is ``True`` when quantile binning
    fell back to uniform bins because the predicted-class confidence
    distribution was effectively constant; callers can lift this onto a
    cohort warning.
    """
    if len(labels) == 0 or proba.size == 0:
        return None
    pred = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(labels)), pred]
    _, fell_back = _bin_edges(confidence, n_bins, strategy)
    return {
        "brier": brier_multiclass(proba, labels),
        "ece": expected_calibration_error(proba, labels, n_bins=n_bins, strategy=strategy),
        "reliability": reliability_curve(proba, labels, n_bins=n_bins, strategy=strategy),
        "n_bins": int(n_bins),
        "strategy": strategy,
        "degenerate_quantile_bins": bool(fell_back),
    }
