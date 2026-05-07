"""
PhenoCluster Apply-Only Validation
==================================

Apply the fitted derivation pipeline (preprocessor + feature selector + LCA
model) to a separate cohort dataframe, returning the predicted phenotype
labels and posteriors. Centralises the preprocessing replay used by both
external validation (single-cohort, manual CSV) and the iterated
generalizability evaluator (per-window or per-site).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _select_feature_matrix(
    preprocessor,
    processed_df: pd.DataFrame,
    feature_selector,
    continuous_columns: List[str],
    categorical_columns: List[str],
    fallback_X: np.ndarray,
) -> np.ndarray:
    """Reproduce the feature-selector handling from FinalizationStage."""
    if feature_selector is None:
        return fallback_X
    selected = feature_selector.get_selected_features()
    sel_cont = [c for c in continuous_columns if c in selected]
    sel_cat = [c for c in categorical_columns if c in selected]
    return preprocessor.get_feature_matrix(processed_df, sel_cont, sel_cat)


def apply_to_cohort(
    raw_df: pd.DataFrame,
    *,
    model,
    preprocessor,
    feature_selector,
    continuous_columns: List[str],
    categorical_columns: List[str],
) -> Dict[str, Any]:
    """Apply the fitted derivation pipeline to ``raw_df``.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Untransformed validation cohort (same schema as the derivation CSV).
    model
        Fitted StepMix-like object exposing ``predict``, ``predict_proba``
        and ``score``.
    preprocessor
        Fitted :class:`DataPreprocessor` from the derivation cohort.
    feature_selector
        Fitted feature selector (or ``None`` if feature selection was off).
    continuous_columns, categorical_columns : list of str
        Original feature column lists from the configuration.

    Returns
    -------
    dict
        Keys: ``labels``, ``proba``, ``log_likelihood``, ``processed_df``,
        ``X``, ``classification_quality``, ``n_samples`` (rows in
        ``raw_df``), and ``n_processed`` (rows surviving preprocessing).
    """
    imputed = preprocessor.transform_impute(raw_df)
    outlier_handled = preprocessor.transform_outliers(imputed)
    processed_df, X_full = preprocessor.transform_preprocess(outlier_handled)
    X = _select_feature_matrix(
        preprocessor,
        processed_df,
        feature_selector,
        continuous_columns,
        categorical_columns,
        fallback_X=X_full,
    )

    from ...pipeline.quality import compute_classification_quality

    labels = np.asarray(model.predict(X))
    proba = np.asarray(model.predict_proba(X))
    log_likelihood = float(model.score(X))
    classification_quality = compute_classification_quality(proba, labels)

    return {
        "labels": labels,
        "proba": proba,
        "log_likelihood": log_likelihood,
        "processed_df": processed_df,
        "X": X,
        "classification_quality": classification_quality,
        "n_samples": int(len(raw_df)),
        "n_processed": int(len(processed_df)),
    }


def safe_apply_to_cohort(
    raw_df: pd.DataFrame,
    *,
    logger,
    label: str,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """Wrapper around :func:`apply_to_cohort` that logs and swallows failures.

    Returns ``None`` when preprocessing fails for the given cohort. Used by
    the iterated evaluator to keep one failed cohort from breaking a run.
    """
    try:
        return apply_to_cohort(raw_df, **kwargs)
    except (ValueError, KeyError, AttributeError, RuntimeError) as exc:
        logger.warning(
            f"Generalizability: apply failed for cohort '{label}' ({type(exc).__name__}): {exc}"
        )
        return None
