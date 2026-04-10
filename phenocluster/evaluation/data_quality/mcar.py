"""Little's MCAR (Missing Completely at Random) test."""

import warnings
from typing import Dict, Optional, Tuple

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

    chi_square, df, status = _compute_mcar_statistic(numeric_data, pattern_strings, unique_patterns)

    if status != "ok":
        return {
            "chi_square": float(chi_square) if np.isfinite(chi_square) else np.nan,
            "dof": int(df) if df > 0 else 0,
            "p_value": np.nan,
            "is_mcar": None,
            "n_patterns": int(len(unique_patterns)),
            "interpretation": (
                "Little's MCAR test undefined: insufficient observed values per "
                "pattern to form a positive degrees-of-freedom statistic."
            ),
        }

    p_value = float(stats.chi2.sf(chi_square, df))
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


def _em_mvn_estimates(
    data: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Maximum-likelihood mean and covariance for multivariate normal data with missingness.

    Parameters
    ----------
    data : np.ndarray, shape (n, p)
        Numeric data with NaN entries marking missing values.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on the maximum absolute parameter change.

    Returns
    -------
    mu : np.ndarray, shape (p,)
        ML estimate of the mean.
    sigma : np.ndarray, shape (p, p)
        ML estimate of the covariance matrix.
    converged : bool
        True if EM converged within ``max_iter`` iterations.
    """
    n, p = data.shape
    obs_mask = ~np.isnan(data)

    # Initialise from column means (mean imputation) and sample covariance.
    col_means = np.nanmean(data, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    filled = np.where(obs_mask, data, col_means)
    mu = filled.mean(axis=0)
    centred = filled - mu
    sigma = (centred.T @ centred) / max(n - 1, 1)
    # Regularise on the diagonal so the initial covariance is invertible.
    eps = 1e-8 * (np.trace(sigma) / p + 1.0)
    sigma = sigma + eps * np.eye(p)

    converged = False
    for _ in range(max_iter):
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))

        for i in range(n):
            obs_i = obs_mask[i]
            mis_i = ~obs_i
            x_i = data[i].copy()

            if mis_i.any():
                if obs_i.any():
                    sigma_oo = sigma[np.ix_(obs_i, obs_i)]
                    sigma_mo = sigma[np.ix_(mis_i, obs_i)]
                    sigma_mm = sigma[np.ix_(mis_i, mis_i)]
                    try:
                        sigma_oo_inv = np.linalg.inv(sigma_oo)
                    except np.linalg.LinAlgError:
                        sigma_oo_inv = np.linalg.pinv(sigma_oo)
                    cond_mean = mu[mis_i] + sigma_mo @ sigma_oo_inv @ (x_i[obs_i] - mu[obs_i])
                    cond_cov = sigma_mm - sigma_mo @ sigma_oo_inv @ sigma_mo.T
                    x_i[mis_i] = cond_mean
                else:
                    cond_mean = mu[mis_i]
                    cond_cov = sigma[np.ix_(mis_i, mis_i)]
                    x_i[mis_i] = cond_mean

                outer = np.outer(x_i, x_i)
                ix_mm = np.ix_(mis_i, mis_i)
                outer[ix_mm] = outer[ix_mm] + cond_cov
                T2 += outer
            else:
                T2 += np.outer(x_i, x_i)

            T1 += x_i

        new_mu = T1 / n
        new_sigma = T2 / n - np.outer(new_mu, new_mu)

        delta = max(
            float(np.max(np.abs(new_mu - mu))),
            float(np.max(np.abs(new_sigma - sigma))),
        )
        mu = new_mu
        sigma = new_sigma
        if delta < tol:
            converged = True
            break

    return mu, sigma, converged


def _compute_mcar_statistic(numeric_data, pattern_strings, unique_patterns):
    """Compute Little's chi-square statistic and degrees of freedom.

    Returns
    -------
    chi_square : float
        Little's d² statistic.
    df : int
        Degrees of freedom (`Σ_g k_g - k`, where `k_g` is the number of
        observed variables in pattern `g` and `k` is the total number of
        variables).
    status : str
        `"ok"` if the statistic is well defined, otherwise a short reason.
    """
    columns = list(numeric_data.columns)
    col_to_idx = {c: i for i, c in enumerate(columns)}
    n_vars = numeric_data.shape[1]
    data_arr = numeric_data.to_numpy(dtype=float, copy=True)

    mu_hat, sigma_hat, em_converged = _em_mvn_estimates(data_arr)
    if not em_converged:
        warnings.warn(
            "EM for Little's MCAR test did not converge within the iteration "
            "limit; reported statistic uses the last EM estimate.",
            stacklevel=2,
        )

    chi_square = 0.0
    df_total = 0

    for pattern in unique_patterns:
        pattern_mask = pattern_strings == pattern
        n_pattern = int(pattern_mask.sum())
        if n_pattern < 2:
            continue

        observed_vars = [columns[i] for i, char in enumerate(pattern) if char == "0"]
        if len(observed_vars) < 1:
            continue

        obs_idx = np.array([col_to_idx[c] for c in observed_vars])
        pattern_data = data_arr[pattern_mask.values][:, obs_idx]
        pattern_means = pattern_data.mean(axis=0)

        sigma_sub = sigma_hat[np.ix_(obs_idx, obs_idx)]
        try:
            sigma_sub_inv = np.linalg.inv(sigma_sub)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Covariance submatrix is singular for one or more missing patterns; "
                "using pseudoinverse. MCAR test results may be unreliable.",
                stacklevel=2,
            )
            sigma_sub_inv = np.linalg.pinv(sigma_sub)

        mean_diff = pattern_means - mu_hat[obs_idx]
        chi_square += n_pattern * float(mean_diff @ sigma_sub_inv @ mean_diff)
        df_total += len(observed_vars)

    df = df_total - n_vars
    if df <= 0:
        return chi_square, max(df_total - n_vars, 0), "insufficient_dof"
    return chi_square, df, "ok"
