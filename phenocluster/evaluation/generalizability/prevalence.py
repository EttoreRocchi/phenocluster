"""
PhenoCluster Phenotype Prevalence Helpers
=========================================

Shared primitives for comparing phenotype distributions across cohorts.
Lifted out of :mod:`phenocluster.evaluation.external_validation` so the
same helpers serve external, temporal and multi-site validation.
"""

from typing import Dict, Optional

import numpy as np
from scipy.stats import chi2_contingency


def cluster_distribution(labels: np.ndarray) -> Dict[int, Dict]:
    """Compute count and percentage for each phenotype label."""
    unique, counts = np.unique(labels, return_counts=True)
    total = int(len(labels))
    if total == 0:
        return {}
    return {
        int(cluster_id): {
            "count": int(count),
            "percentage": float(count / total * 100.0),
        }
        for cluster_id, count in zip(unique, counts)
    }


def chi2_homogeneity(
    deriv_distribution: Dict[int, Dict],
    val_distribution: Dict[int, Dict],
) -> Optional[Dict]:
    """
    Run a chi-square test of homogeneity comparing two cohort phenotype
    distributions.

    Builds a 2 x K contingency table from the per-cluster ``count`` entries
    of ``cluster_distribution(...)`` (rows are derivation vs validation,
    columns are clusters).

    Parameters
    ----------
    deriv_distribution, val_distribution : dict
        Mapping ``{cluster_id: {"count": int, ...}}`` as returned by
        :func:`cluster_distribution`.

    Returns
    -------
    dict or None
        ``{"statistic", "p_value", "df"}``. Returns ``None`` when the table
        cannot be built (no shared clusters, empty cohort, all-zero column).
    """
    cluster_ids = sorted(set(deriv_distribution.keys()) & set(val_distribution.keys()))
    if not cluster_ids:
        return None
    deriv_row = [int(deriv_distribution[k].get("count", 0)) for k in cluster_ids]
    val_row = [int(val_distribution[k].get("count", 0)) for k in cluster_ids]
    table = np.array([deriv_row, val_row])
    if table.sum() == 0 or np.any(table.sum(axis=0) == 0):
        return None
    try:
        stat, p, dof, _ = chi2_contingency(table)
    except ValueError:
        return None
    return {"statistic": float(stat), "p_value": float(p), "df": int(dof)}


def chi2_cohort_comparison(
    deriv_counts: Dict[int, Dict],
    val_counts: Dict[int, Dict],
) -> Optional[Dict]:
    """
    Run a chi-square test comparing outcome counts across two cohorts.

    Builds a 2 x (2 * K) contingency table where each cluster contributes a
    pair of columns ``[positive, negative]`` and rows are derivation vs
    validation. Returns ``None`` if the table cannot be built (e.g., zero
    column sums) or if the test fails.
    """
    cluster_ids = sorted(set(deriv_counts.keys()) & set(val_counts.keys()))
    if not cluster_ids:
        return None

    row_deriv = []
    row_val = []
    for cid in cluster_ids:
        d = deriv_counts[cid]
        v = val_counts[cid]
        d_pos = d.get("n_positive", 0)
        d_tot = d.get("n_total", 0)
        v_pos = v.get("n_positive", 0)
        v_tot = v.get("n_total", 0)
        row_deriv.extend([d_pos, d_tot - d_pos])
        row_val.extend([v_pos, v_tot - v_pos])

    table = np.array([row_deriv, row_val])
    col_sums = table.sum(axis=0)
    if np.any(col_sums == 0) or table.sum() == 0:
        return None

    try:
        stat, p, _, _ = chi2_contingency(table)
        return {"statistic": float(stat), "p_value": float(p)}
    except ValueError:
        return None
