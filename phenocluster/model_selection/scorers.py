"""
PhenoCluster Custom Scorers
===========================

Scoring functions for StepMix model selection.

Individual scorer functions follow the sklearn convention (higher is better).
For information criteria (BIC, AIC, etc.) where lower is better,
scorer functions return negative values.

``get_all_criteria()`` returns values on the original scale (not negated).
"""

from typing import Any, Callable, Dict, Optional

import numpy as np

# Available criteria for model selection
AVAILABLE_CRITERIA = ["BIC", "AIC", "CAIC", "SABIC", "ICL", "ENTROPY"]

# Type alias for scorer functions
ScorerFunc = Callable[[Any, np.ndarray, Optional[np.ndarray]], float]


def _make_ic_scorer(method_name: str) -> ScorerFunc:
    """Factory for information-criterion scorers (lower IC -> higher score)."""

    def scorer(estimator: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
        try:
            score = -getattr(estimator, method_name)(X)
            return score if np.isfinite(score) else -np.inf
        except Exception:
            return -np.inf

    scorer.__name__ = f"{method_name}_score"
    scorer.__doc__ = f"Negative {method_name.upper()} score (higher is better for sklearn)."
    return scorer


bic_score = _make_ic_scorer("bic")
aic_score = _make_ic_scorer("aic")
caic_score = _make_ic_scorer("caic")
sabic_score = _make_ic_scorer("sabic")


def icl_score(estimator: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """
    Integrated Completed Likelihood (ICL) score.

    ICL = BIC + 2 * total_entropy.
    StepMix entropy(X) returns total entropy (summed over samples).
    Entropy is an additional penalty beyond BIC for poor class separation.
    Lower ICL indicates better model with clearer classes.
    Returns negative for sklearn maximization.

    Parameters
    ----------
    estimator : StepMix
        Fitted StepMix model
    X : np.ndarray
        Measurement data
    y : np.ndarray, optional
        Structural data (ignored)

    Returns
    -------
    float
        Negative ICL (higher is better for sklearn)
    """
    try:
        bic = estimator.bic(X)
        entropy = estimator.entropy(X)  # StepMix returns total entropy (summed over samples)
        score = -(bic + 2 * entropy)
        return score if np.isfinite(score) else -np.inf
    except Exception:
        return -np.inf


def relative_entropy_score(estimator: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """
    Relative entropy (class separation) score.

    Higher relative entropy (0-1) indicates better class separation.
    No negation needed as higher is already better.

    Parameters
    ----------
    estimator : StepMix
        Fitted StepMix model
    X : np.ndarray
        Measurement data
    y : np.ndarray, optional
        Structural data (ignored)

    Returns
    -------
    float
        Relative entropy (higher is better)
    """
    try:
        score = estimator.relative_entropy(X)
        return score if np.isfinite(score) else 0.0
    except Exception:
        return 0.0  # Worst possible for entropy


def create_scorer(criterion: str) -> Callable:
    """
    Create a sklearn-compatible scorer for the given criterion.

    Parameters
    ----------
    criterion : str
        One of 'BIC', 'AIC', 'CAIC', 'SABIC', 'ICL', 'entropy'

    Returns
    -------
    Callable
        Scorer function

    Raises
    ------
    ValueError
        If criterion is not recognized
    """
    criterion = criterion.upper()

    scorers = {
        "BIC": bic_score,
        "AIC": aic_score,
        "CAIC": caic_score,
        "SABIC": sabic_score,
        "ICL": icl_score,
        "ENTROPY": relative_entropy_score,
    }

    if criterion not in scorers:
        raise ValueError(f"Unknown criterion '{criterion}'. Available: {list(scorers.keys())}")

    # Return scorer function directly
    return scorers[criterion]


def get_all_criteria(estimator: Any, X: np.ndarray) -> Dict[str, float]:
    """
    Compute all available information criteria for a fitted model.

    Parameters
    ----------
    estimator : StepMix
        Fitted StepMix model
    X : np.ndarray
        Measurement data

    Returns
    -------
    Dict[str, float]
        Dictionary with all criteria values (original scale, not negated)
    """
    results = {}

    try:
        results["log_likelihood"] = estimator.score(X)
    except Exception:
        results["log_likelihood"] = None

    try:
        results["BIC"] = estimator.bic(X)
    except Exception:
        results["BIC"] = None

    try:
        results["AIC"] = estimator.aic(X)
    except Exception:
        results["AIC"] = None

    try:
        results["CAIC"] = estimator.caic(X)
    except Exception:
        results["CAIC"] = None

    try:
        results["SABIC"] = estimator.sabic(X)
    except Exception:
        results["SABIC"] = None

    try:
        entropy = estimator.entropy(X)
        results["entropy"] = entropy
        if results["BIC"] is not None:
            results["ICL"] = results["BIC"] + 2 * entropy
        else:
            results["ICL"] = None
    except Exception:
        results["entropy"] = None
        results["ICL"] = None

    try:
        rel_ent = estimator.relative_entropy(X)
        results["relative_entropy"] = rel_ent
        results["ENTROPY"] = rel_ent
    except Exception:
        results["relative_entropy"] = None
        results["ENTROPY"] = None

    try:
        results["n_parameters"] = estimator.n_parameters
    except Exception:
        results["n_parameters"] = None

    return results
