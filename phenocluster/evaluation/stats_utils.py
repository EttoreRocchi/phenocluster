"""
Statistical Inference Utilities
================================

Shared utilities for frequentist statistical inference, including
FDR correction for multiple comparisons.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import false_discovery_control


def apply_fdr_correction(p_values: List[Optional[float]]) -> List[Optional[float]]:
    """Apply Benjamini-Hochberg FDR correction to a list of p-values.

    Parameters
    ----------
    p_values : List[Optional[float]]
        Raw p-values (None/NaN entries are preserved unchanged).

    Returns
    -------
    List[Optional[float]]
        FDR-corrected q-values in original order.
    """
    p_arr = np.array([p if p is not None else np.nan for p in p_values], dtype=float)
    valid_mask = ~np.isnan(p_arr)
    if int(valid_mask.sum()) == 0:
        return list(p_values)

    adjusted = false_discovery_control(p_arr[valid_mask], method="bh")

    result = p_arr.copy()
    result[valid_mask] = adjusted
    return [None if np.isnan(v) else float(v) for v in result]


def create_phenotype_dummies(
    labels: np.ndarray, n_clusters: int, reference: int = 0
) -> Tuple[np.ndarray, List[int]]:
    """Create dummy variables excluding the reference phenotype.

    Parameters
    ----------
    labels : np.ndarray
        Cluster/phenotype labels for each observation.
    n_clusters : int
        Total number of clusters.
    reference : int
        Phenotype ID to use as the reference (excluded from dummies).

    Returns
    -------
    dummies : np.ndarray, shape (n_samples, n_clusters - 1)
        Dummy-coded matrix.
    non_ref_ids : List[int]
        Phenotype IDs corresponding to each dummy column.
    """
    non_ref = [k for k in range(n_clusters) if k != reference]
    dummies = np.zeros((len(labels), len(non_ref)))
    for i, k in enumerate(non_ref):
        dummies[:, i] = (labels == k).astype(float)
    return dummies, non_ref
