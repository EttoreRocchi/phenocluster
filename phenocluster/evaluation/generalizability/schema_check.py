"""
PhenoCluster Validation Cohort Schema Check
===========================================

Compare a validation cohort against the schema the derivation model was fitted
on, before its phenotypes are predicted. Applying a fitted pipeline to a cohort
assembled from a different export silently degrades when a feature is absent,
is entirely missing, or carries category labels the encoder never saw: unknown
categories fall back to the modal one, so the model still returns phenotypes
that look plausible but ignore that variable.

The findings are attached to :class:`~phenocluster.evaluation.generalizability
._types.CohortReport.warnings`, so they reach the HTML report instead of only
the log.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from ...data.encoder import canonical_labels

UNSEEN_CATEGORY_WARN_FRACTION = 0.20


def _missing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


def _known_classes(preprocessor, column: str) -> Optional[set]:
    """Return the labels the fitted encoder holds for ``column``, if any."""
    encoders = getattr(preprocessor, "label_encoders", None) or {}
    if column in encoders:
        return set(encoders[column].classes_)
    frequencies = getattr(preprocessor, "frequency_encodings", None) or {}
    if column in frequencies:
        return set(frequencies[column])
    onehot = getattr(preprocessor, "onehot_encoder", None)
    categories = getattr(onehot, "categories_", None)
    feature_names = getattr(onehot, "feature_names_in_", None)
    if categories is not None and feature_names is not None:
        names = list(feature_names)
        if column in names:
            return set(categories[names.index(column)])
    return None


def check_cohort_schema(
    raw_df: pd.DataFrame,
    *,
    preprocessor,
    continuous_columns: List[str],
    categorical_columns: List[str],
    outcome_columns: Optional[List[str]] = None,
    warn_fraction: float = UNSEEN_CATEGORY_WARN_FRACTION,
) -> Dict[str, Any]:
    """Audit ``raw_df`` against the schema of the fitted derivation pipeline.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Untransformed validation cohort.
    preprocessor
        Fitted :class:`~phenocluster.data.preprocessor.DataPreprocessor` from
        the derivation cohort.
    continuous_columns, categorical_columns : list of str
        Feature columns from the configuration.
    outcome_columns : list of str, optional
        Outcome columns needed for the cross-cohort concordance.
    warn_fraction : float
        Share of values remapped to the modal category above which a column is
        reported as a warning.

    Returns
    -------
    dict
        ``warnings`` (list of str, ready for the cohort report),
        ``missing_features``, ``missing_outcomes``, ``all_missing_features``
        and ``unseen_categories`` (per column: labels and affected row count).
    """
    warnings: List[str] = []
    missing_features = _missing_columns(
        raw_df, list(continuous_columns) + list(categorical_columns)
    )
    missing_outcomes = _missing_columns(raw_df, list(outcome_columns or []))
    all_missing: List[str] = []
    unseen: Dict[str, Dict[str, Any]] = {}

    if missing_features:
        warnings.append(
            f"schema: {len(missing_features)} feature column(s) absent from the cohort "
            f"and treated as missing data: {', '.join(missing_features)}"
        )
    if missing_outcomes:
        warnings.append(
            f"schema: {len(missing_outcomes)} outcome column(s) absent from the cohort, "
            f"so their concordance is not computed: {', '.join(missing_outcomes)}"
        )

    for col in list(continuous_columns) + list(categorical_columns):
        if col in raw_df.columns and raw_df[col].isna().all():
            all_missing.append(col)
    if all_missing:
        warnings.append(
            f"schema: {len(all_missing)} feature column(s) entirely missing in the cohort: "
            f"{', '.join(all_missing)}"
        )

    for col in categorical_columns:
        if col not in raw_df.columns:
            continue
        known = _known_classes(preprocessor, col)
        if not known:
            continue
        observed = raw_df.loc[raw_df[col].notna(), col]
        if observed.empty:
            continue
        labels = canonical_labels(observed)
        unknown_mask = ~labels.isin(known)
        if not unknown_mask.any():
            continue
        n_unknown = int(unknown_mask.sum())
        fraction = n_unknown / len(raw_df)
        unseen[col] = {
            "labels": sorted(set(labels[unknown_mask])),
            "n_rows": n_unknown,
            "fraction": fraction,
            "known_labels": sorted(known),
        }
        message = (
            f"schema: '{col}' has {n_unknown} value(s) ({fraction:.0%} of the cohort) in "
            f"categories the derivation model never saw "
            f"({', '.join(unseen[col]['labels'])}); they fall back to the modal category"
        )
        if fraction >= warn_fraction:
            message += " -- check that both cohorts use the same coding for this variable"
        warnings.append(message)

    return {
        "warnings": warnings,
        "missing_features": missing_features,
        "missing_outcomes": missing_outcomes,
        "all_missing_features": all_missing,
        "unseen_categories": unseen,
    }
