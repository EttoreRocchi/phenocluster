"""
PhenoCluster Refit-and-Match Validator
======================================

Refits the LCA model from scratch on a validation cohort and matches the
resulting clusters against the derivation-applied labels using a padded
Hungarian assignment. Returns label-agreement metrics (ARI, NMI,
Hungarian-matched accuracy) plus the cluster mapping itself.

ARI and NMI are computed on the **raw** (pre-alignment) labels because
both metrics are permutation-invariant; alignment is only needed for
plotting/clinical narrative and for computing matched accuracy.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _build_padded_confusion(
    derivation_labels: np.ndarray,
    refit_labels: np.ndarray,
    n_deriv: int,
    n_val: int,
) -> np.ndarray:
    size = max(n_deriv, n_val)
    confusion = np.zeros((size, size), dtype=int)
    deriv_arr = np.asarray(derivation_labels)
    refit_arr = np.asarray(refit_labels)
    valid = (deriv_arr >= 0) & (deriv_arr < n_deriv) & (refit_arr >= 0) & (refit_arr < n_val)
    np.add.at(confusion, (deriv_arr[valid], refit_arr[valid]), 1)
    return confusion


def hungarian_alignment(
    derivation_labels: np.ndarray,
    refit_labels: np.ndarray,
    n_deriv_clusters: int,
    n_val_clusters: int,
) -> Dict[str, Any]:
    """Maximise total overlap between two clusterings via Hungarian assignment.

    Returns
    -------
    dict
        ``mapping`` (refit -> derivation), ``aligned_labels`` (refit labels
        relabelled to derivation space), ``unmatched_derivation_clusters``,
        ``unmatched_validation_clusters``.
    """
    confusion = _build_padded_confusion(
        derivation_labels, refit_labels, n_deriv_clusters, n_val_clusters
    )
    row_ind, col_ind = linear_sum_assignment(-confusion)

    real_derivation = set(range(n_deriv_clusters))
    real_validation = set(range(n_val_clusters))
    mapping: Dict[int, int] = {}
    matched_deriv = set()
    matched_val = set()
    for r, c in zip(row_ind, col_ind):
        if c in real_validation and r in real_derivation:
            mapping[int(c)] = int(r)
            matched_deriv.add(int(r))
            matched_val.add(int(c))

    aligned = np.array([mapping.get(int(lab), int(lab) + n_deriv_clusters) for lab in refit_labels])
    unmatched_deriv = sorted(real_derivation - matched_deriv)
    unmatched_val = sorted(real_validation - matched_val)
    return {
        "mapping": mapping,
        "aligned_labels": aligned,
        "unmatched_derivation_clusters": unmatched_deriv,
        "unmatched_validation_clusters": unmatched_val,
    }


def _build_refit_model(reference_model, n_components: int, random_state: int):
    """Construct a fresh estimator with the same hyperparameters as ``reference_model``."""
    cls = type(reference_model)
    if hasattr(reference_model, "get_params"):
        params = dict(reference_model.get_params(deep=False))
    else:
        params = {}
    params["n_components"] = n_components
    params["random_state"] = random_state
    try:
        return cls(**params)
    except TypeError:
        return cls(n_components=n_components, random_state=random_state)


def refit_and_match(
    X_val: np.ndarray,
    derivation_labels_on_val: np.ndarray,
    *,
    reference_model,
    n_clusters: int,
    random_state: int = 42,
    min_validation_size_for_refit: int = 100,
    logger=None,
) -> Optional[Dict[str, Any]]:
    """Refit the LCA model on validation data and align labels with the derivation cohort.

    Parameters
    ----------
    X_val : np.ndarray
        Pre-processed validation feature matrix.
    derivation_labels_on_val : np.ndarray
        Predicted labels of the *derivation* model applied to ``X_val``.
        Used as the reference clustering for alignment.
    reference_model
        The fitted derivation StepMix-like model. A fresh estimator with the
        same hyperparameters is constructed for the refit.
    n_clusters : int
        Number of clusters to fit (mirrors the derivation cohort).
    random_state : int
        RNG seed for the refit.
    min_validation_size_for_refit : int
        Skip the refit (and return ``None``) when the validation cohort is
        smaller than this threshold; refitting tiny cohorts is unstable and
        produces meaningless ARI/NMI.
    logger
        Optional logger for skip/warn messages.

    Returns
    -------
    dict or None
        Returns ``None`` when the cohort is too small to refit reliably.
    """
    n_val = len(X_val)
    if n_val < min_validation_size_for_refit:
        if logger is not None:
            logger.warning(
                f"Generalizability: skipping refit (n={n_val} < "
                f"min_validation_size_for_refit={min_validation_size_for_refit})"
            )
        return None

    refit = _build_refit_model(reference_model, n_clusters, random_state)
    refit.fit(X_val)
    refit_labels = np.asarray(refit.predict(X_val))
    refit_proba = np.asarray(refit.predict_proba(X_val))
    n_val_clusters = refit_proba.shape[1]

    ari = float(adjusted_rand_score(derivation_labels_on_val, refit_labels))
    nmi = float(normalized_mutual_info_score(derivation_labels_on_val, refit_labels))

    alignment = hungarian_alignment(
        derivation_labels_on_val,
        refit_labels,
        n_deriv_clusters=int(n_clusters),
        n_val_clusters=int(n_val_clusters),
    )
    matched = (alignment["aligned_labels"] == derivation_labels_on_val).mean()
    matched_accuracy = float(matched)

    return {
        "refit_labels_raw": refit_labels,
        "refit_proba": refit_proba,
        "aligned_labels": alignment["aligned_labels"],
        "ari": ari,
        "nmi": nmi,
        "matched_accuracy": matched_accuracy,
        "mapping": alignment["mapping"],
        "unmatched_derivation_clusters": alignment["unmatched_derivation_clusters"],
        "unmatched_validation_clusters": alignment["unmatched_validation_clusters"],
        "n_validation_clusters": int(n_val_clusters),
        "log_likelihood": float(refit.score(X_val)),
    }


def to_json_safe(refit_result: Dict[str, Any]) -> Dict[str, Any]:
    """Strip numpy arrays from a refit result for JSON serialization."""
    if not refit_result:
        return refit_result
    out = dict(refit_result)
    out.pop("refit_labels_raw", None)
    out.pop("refit_proba", None)
    out.pop("aligned_labels", None)
    return out
