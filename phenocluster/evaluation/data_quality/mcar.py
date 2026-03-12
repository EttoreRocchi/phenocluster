"""Little's MCAR (Missing Completely at Random) test."""

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def littles_mcar_test(data: pd.DataFrame) -> Dict:
    """
    Perform Little's MCAR (Missing Completely at Random) test.

    Little's MCAR test (Little, 1988) tests the null hypothesis that the
    missing data mechanism is MCAR. The test compares observed variable means
    for each missing data pattern against expected means if data were MCAR.

    Under MCAR, the test statistic follows a chi-square distribution.
    A significant p-value (typically p < 0.05) suggests data is NOT MCAR.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with potential missing values

    Returns
    -------
    Dict
        Dictionary containing:
        - chi_square: Test statistic
        - dof: Degrees of freedom
        - p_value: P-value for the test
        - is_mcar: Boolean indicating if null hypothesis (MCAR) is accepted at alpha=0.05
        - n_patterns: Number of unique missing data patterns
        - interpretation: Human-readable interpretation

    References
    ----------
    Little, R.J.A. (1988). A Test of Missing Completely at Random for
    Multivariate Data with Missing Values. Journal of the American
    Statistical Association, 83(404), 1198-1202.

    Notes
    -----
    - Requires at least 2 variables with missing data
    - Test may be unreliable with very small samples or many missing patterns
    - Non-significant result does not prove MCAR, only fails to reject it
    """
    early = _validate_mcar_input(data)
    if early is not None:
        return early

    numeric_data = data.select_dtypes(include=[np.number]).copy()
    missing_mask = numeric_data.isna()
    pattern_strings = pd.Series(
        ["".join(row) for row in missing_mask.astype(int).astype(str).values],
        index=missing_mask.index,
    )
    unique_patterns = pattern_strings.unique()

    chi_square, df = _compute_mcar_statistic(numeric_data, pattern_strings, unique_patterns)

    p_value = 1 - stats.chi2.cdf(chi_square, df)
    alpha = 0.05
    is_mcar = p_value > alpha

    if is_mcar:
        interpretation = (
            f"Little's MCAR test: chi-square({df}) = {chi_square:.3f}, p = {p_value:.4f}. "
            f"The null hypothesis (MCAR) is NOT rejected at alpha={alpha}. "
            f"Missing data appears to be Missing Completely at Random."
        )
    else:
        interpretation = (
            f"Little's MCAR test: chi-square({df}) = {chi_square:.3f}, p = {p_value:.4f}. "
            f"The null hypothesis (MCAR) IS rejected at alpha={alpha}. "
            f"Missing data is NOT Missing Completely at Random. "
            f"Consider MAR-appropriate methods (e.g., multiple imputation, maximum likelihood)."
        )

    return {
        "chi_square": float(chi_square),
        "dof": int(df),
        "p_value": float(p_value),
        "is_mcar": bool(is_mcar),
        "n_patterns": int(len(unique_patterns)),
        "interpretation": interpretation,
    }


def _validate_mcar_input(data: pd.DataFrame) -> Optional[Dict]:
    """Validate input for Little's MCAR test. Returns early-exit dict or None."""
    numeric_data = data.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": None,
            "n_patterns": 0,
            "interpretation": "No numeric columns available for MCAR test",
        }

    missing_mask = numeric_data.isna()
    n_missing_cols = missing_mask.any().sum()

    if n_missing_cols == 0:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": True,
            "n_patterns": 1,
            "interpretation": "No missing data - MCAR test not applicable",
        }

    if n_missing_cols < 2:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": None,
            "n_patterns": 1,
            "interpretation": (
                "Less than 2 variables with missing data - test requires multiple variables"
            ),
        }

    pattern_strings = pd.Series(
        ["".join(row) for row in missing_mask.astype(int).astype(str).values],
        index=missing_mask.index,
    )
    if len(pattern_strings.unique()) == 1:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": None,
            "n_patterns": 1,
            "interpretation": "Only one missing data pattern - test requires multiple patterns",
        }

    return None


def _compute_mcar_statistic(numeric_data, pattern_strings, unique_patterns):
    """Compute chi-square statistic and degrees of freedom for MCAR test."""
    n_vars = numeric_data.shape[1]
    overall_means = numeric_data.mean()
    cov_matrix = numeric_data.cov()

    try:
        np.linalg.inv(cov_matrix.values)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Covariance matrix is singular; using pseudoinverse. "
            "MCAR test results may be unreliable.",
            stacklevel=2,
        )

    chi_square = 0.0
    df_total = 0

    for pattern in unique_patterns:
        pattern_mask = pattern_strings == pattern
        n_pattern = pattern_mask.sum()
        if n_pattern < 2:
            continue

        pattern_data = numeric_data.loc[pattern_mask]
        observed_vars = [numeric_data.columns[i] for i, char in enumerate(pattern) if char == "0"]
        if len(observed_vars) < 1:
            continue

        pattern_means = pattern_data[observed_vars].mean()
        cov_sub = cov_matrix.loc[observed_vars, observed_vars]

        try:
            cov_sub_inv = np.linalg.inv(cov_sub.values)
        except np.linalg.LinAlgError:
            cov_sub_inv = np.linalg.pinv(cov_sub.values)

        mean_diff = pattern_means.values - overall_means[observed_vars].values
        chi_square += n_pattern * np.dot(np.dot(mean_diff, cov_sub_inv), mean_diff)
        df_total += len(observed_vars)

    df = max(1, df_total - n_vars)
    return chi_square, df
