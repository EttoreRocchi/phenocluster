"""Classification quality metrics utility."""

import numpy as np


def compute_classification_quality(proba: np.ndarray, labels: np.ndarray) -> dict:
    """Compute classification quality metrics from posterior probabilities.

    Parameters
    ----------
    proba : np.ndarray
        N x K matrix of posterior probabilities.
    labels : np.ndarray
        Length-N array of modal class assignments.

    Returns
    -------
    dict
        Classification quality metrics including overall_avepp,
        overall_relative_entropy, n_clusters, n_samples,
        assignment_confidence, and per_phenotype breakdown.
    """
    if len(proba) == 0:
        return {}
    n_clusters = proba.shape[1]
    max_proba = np.max(proba, axis=1)
    eps = 1e-15
    sample_entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    max_entropy = np.log(n_clusters)
    rel_entropy = sample_entropy / max_entropy if max_entropy > 0 else sample_entropy

    result = {
        "overall_avepp": float(np.mean(max_proba)),
        "overall_relative_entropy": float(np.mean(rel_entropy)),
        "n_clusters": n_clusters,
        "n_samples": len(labels),
        "assignment_confidence": {
            "above_90": float(np.mean(max_proba > 0.9) * 100),
            "above_80": float(np.mean(max_proba > 0.8) * 100),
            "above_70": float(np.mean(max_proba > 0.7) * 100),
            "below_70": float(np.mean(max_proba <= 0.7) * 100),
        },
        "per_phenotype": {},
    }

    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cluster_proba = max_proba[mask]
            cluster_entropy = rel_entropy[mask]
            result["per_phenotype"][int(k)] = {
                "n": int(mask.sum()),
                "avepp": float(np.mean(cluster_proba)),
                "avepp_std": float(np.std(cluster_proba)),
                "min_pp": float(np.min(cluster_proba)),
                "median_pp": float(np.median(cluster_proba)),
                "mean_entropy": float(np.mean(cluster_entropy)),
            }

    return result
