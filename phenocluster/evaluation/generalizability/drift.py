"""
PhenoCluster Distribution Drift
===============================

Per-feature drift summaries between a derivation cohort and a validation
cohort: Population Stability Index (PSI) for binned continuous variables,
Kolmogorov-Smirnov test for raw continuous variables, and a chi-square
test for categorical variables.

PSI is computed against equal-mass bin edges fitted on the derivation
distribution (so the validation cohort is scored against the bins it
will be reported against). Laplace smoothing with a default
``eps = 0.5 / min(n_deriv, n_val)`` (a hypothetical half-event in the
smaller cohort) keeps ``log(p_v / p_d)`` finite when a bin has zero
support in either cohort without inflating PSI on small cohorts.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def _psi_eps(n_deriv: int, n_val: int) -> float:
    """Default PSI smoothing floor that scales with the smaller cohort size."""
    return max(0.5 / max(min(n_deriv, n_val), 1), 1e-6)


def population_stability_index(
    deriv: np.ndarray,
    val: np.ndarray,
    n_bins: int = 10,
    eps: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """Compute PSI with derivation-fitted equal-mass bins.

    Parameters
    ----------
    deriv, val : np.ndarray
        Continuous values from the derivation and validation cohorts.
    n_bins : int, default 10
        Number of equal-mass bins fitted on ``deriv``.
    eps : float, optional
        Laplace smoothing floor applied to bin proportions before the log
        ratio. When ``None``, defaults to ``0.5 / min(len(deriv), len(val))``.

    Returns
    -------
    psi : float
    edges : np.ndarray
        The bin edges used (length ``n_bins + 1``).
    """
    deriv_clean = deriv[~np.isnan(deriv)]
    val_clean = val[~np.isnan(val)]
    if deriv_clean.size == 0 or val_clean.size == 0:
        return float("nan"), np.array([])

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(deriv_clean, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan"), edges

    deriv_bins = np.digitize(deriv_clean, edges[1:-1], right=True)
    val_bins = np.digitize(val_clean, edges[1:-1], right=True)

    n_actual = len(edges) - 1
    deriv_pct = np.bincount(deriv_bins, minlength=n_actual)[:n_actual] / len(deriv_clean)
    val_pct = np.bincount(val_bins, minlength=n_actual)[:n_actual] / len(val_clean)

    if eps is None:
        eps = _psi_eps(len(deriv_clean), len(val_clean))
    deriv_pct = np.maximum(deriv_pct, eps)
    val_pct = np.maximum(val_pct, eps)

    psi = float(np.sum((val_pct - deriv_pct) * np.log(val_pct / deriv_pct)))
    return psi, edges


def categorical_drift(
    deriv: pd.Series,
    val: pd.Series,
) -> Tuple[Optional[float], Optional[float], float]:
    """Chi-square test on category-frequency contingency table + PSI.

    Returns
    -------
    chi2_stat : float or None
    chi2_p : float or None
    psi : float
    """
    deriv_clean = deriv.dropna().astype("object")
    val_clean = val.dropna().astype("object")
    if deriv_clean.empty or val_clean.empty:
        return None, None, float("nan")

    categories = sorted(set(deriv_clean.unique()) | set(val_clean.unique()), key=str)
    deriv_counts = deriv_clean.value_counts().reindex(categories, fill_value=0).values
    val_counts = val_clean.value_counts().reindex(categories, fill_value=0).values

    table = np.array([deriv_counts, val_counts])
    if table.sum() == 0 or np.any(table.sum(axis=0) == 0):
        return None, None, float("nan")

    try:
        stat, p, _, _ = chi2_contingency(table)
        chi2_stat: Optional[float] = float(stat)
        chi2_p: Optional[float] = float(p)
    except ValueError:
        chi2_stat = None
        chi2_p = None

    eps = _psi_eps(int(deriv_counts.sum()), int(val_counts.sum()))
    deriv_pct = np.maximum(deriv_counts / deriv_counts.sum(), eps)
    val_pct = np.maximum(val_counts / val_counts.sum(), eps)
    psi = float(np.sum((val_pct - deriv_pct) * np.log(val_pct / deriv_pct)))
    return chi2_stat, chi2_p, psi


def feature_drift(
    deriv_df: pd.DataFrame,
    val_df: pd.DataFrame,
    continuous_cols: Iterable[str],
    categorical_cols: Iterable[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a tidy drift table with one row per feature.

    Columns:
    ``feature``, ``kind``, ``psi``, ``ks_stat``, ``ks_p``, ``chi2_stat``,
    ``chi2_p``, ``n_deriv``, ``n_val``, ``missing_diff``.
    """
    rows = []
    for feature in continuous_cols:
        if feature not in deriv_df.columns or feature not in val_df.columns:
            continue
        deriv_vals = pd.to_numeric(deriv_df[feature], errors="coerce").to_numpy(dtype=float)
        val_vals = pd.to_numeric(val_df[feature], errors="coerce").to_numpy(dtype=float)
        deriv_clean = deriv_vals[~np.isnan(deriv_vals)]
        val_clean = val_vals[~np.isnan(val_vals)]
        psi, _ = population_stability_index(deriv_vals, val_vals, n_bins=n_bins)
        if deriv_clean.size > 0 and val_clean.size > 0:
            ks = ks_2samp(deriv_clean, val_clean)
            ks_stat: Optional[float] = float(ks.statistic)
            ks_p: Optional[float] = float(ks.pvalue)
        else:
            ks_stat = None
            ks_p = None
        rows.append(
            {
                "feature": feature,
                "kind": "continuous",
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "chi2_stat": None,
                "chi2_p": None,
                "n_deriv": int(deriv_clean.size),
                "n_val": int(val_clean.size),
                "missing_diff": _missing_rate_diff(deriv_vals, val_vals),
            }
        )

    for feature in categorical_cols:
        if feature not in deriv_df.columns or feature not in val_df.columns:
            continue
        chi2_stat, chi2_p, psi = categorical_drift(deriv_df[feature], val_df[feature])
        n_deriv = int(deriv_df[feature].notna().sum())
        n_val = int(val_df[feature].notna().sum())
        rows.append(
            {
                "feature": feature,
                "kind": "categorical",
                "psi": psi,
                "ks_stat": None,
                "ks_p": None,
                "chi2_stat": chi2_stat,
                "chi2_p": chi2_p,
                "n_deriv": n_deriv,
                "n_val": n_val,
                "missing_diff": _missing_rate_diff_series(deriv_df[feature], val_df[feature]),
            }
        )

    return pd.DataFrame(rows)


def top_drifted(report: pd.DataFrame, k: int = 20, by: str = "psi") -> pd.DataFrame:
    """Return the ``k`` rows with the largest absolute value of ``by``."""
    if report.empty or by not in report.columns:
        return report
    abs_col = report[by].abs()
    return (
        report.assign(_sort=abs_col)
        .sort_values("_sort", ascending=False)
        .head(k)
        .drop(columns="_sort")
    )


def _missing_rate_diff(deriv: np.ndarray, val: np.ndarray) -> float:
    if deriv.size == 0 or val.size == 0:
        return float("nan")
    return float(np.mean(np.isnan(val))) - float(np.mean(np.isnan(deriv)))


def _missing_rate_diff_series(deriv: pd.Series, val: pd.Series) -> float:
    if len(deriv) == 0 or len(val) == 0:
        return float("nan")
    return float(val.isna().mean()) - float(deriv.isna().mean())
