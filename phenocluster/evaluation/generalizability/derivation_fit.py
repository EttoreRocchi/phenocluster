"""
PhenoCluster Per-Split Derivation Fit
=====================================

Fits a fresh preprocessor and StepMix model on a *derivation-only*
dataframe so that the model used for in-CSV split generalizability
metrics has never seen the validation rows during training. Used by
:class:`GeneralizabilityEvaluator` whenever
``generalizability.training_scope`` is ``"per_split"``.

The fresh derivation labels are Hungarian-aligned to the global
full-cohort labels so that phenotype IDs (Phenotype 0, 1, 2, ...) stay
consistent across the descriptive analyses and the per-split
generalizability outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ...data.preprocessor import DataPreprocessor
from .refit_validator import hungarian_alignment


def _resolve_feature_selector_for_split(
    config,
    feature_selector,
    derivation_df: pd.DataFrame,
    feature_selector_scope: str,
    logger=None,
):
    """Decide whether to refit the feature selector on derivation rows.

    Parameters
    ----------
    config : PhenoClusterConfig
        Used to read ``feature_selection`` and the outcome-like column
        list for collision detection.
    feature_selector
        The global pipeline's fitted selector (may be ``None`` if
        feature selection was disabled).
    derivation_df : pd.DataFrame
        Derivation rows for the current split. The selector, if refit,
        is fit on these rows (after the per-split preprocessor has
        already fitted, but using the unencoded original columns).
    feature_selector_scope : {"auto", "global", "per_split"}
        Resolution policy. ``"global"`` always reuses the global
        selector. ``"per_split"`` always refits, raising if not safe.
        ``"auto"`` refits when safe and falls back to the global selector
        with a warning otherwise.

    Returns
    -------
    selector
        The selector to use for this split (may be the same instance as
        ``feature_selector``).
    mode : str
        One of ``"none"`` (feature selection disabled),
        ``"global_reused"``, ``"global_reused_with_warning"``, or
        ``"per_split_refit"``.
    notes : list of str
        Per-split warning messages to attach to the cohort report.
    """
    fs_cfg = config.feature_selection
    if not fs_cfg.enabled or feature_selector is None:
        return feature_selector, "none", []

    if feature_selector_scope == "global":
        return feature_selector, "global_reused", []

    is_supervised = bool(getattr(fs_cfg, "require_target", False))
    target_col = fs_cfg.target_column
    outcomes = config.outcome_like_columns()

    if is_supervised:
        if target_col in outcomes:
            msg = (
                f"feature_selector reused from global pipeline: target "
                f"column '{target_col}' is also an outcome, refitting "
                f"per split would still leak validation outcome into "
                f"selection."
            )
            if feature_selector_scope == "per_split":
                raise ValueError(
                    msg + " Set feature_selector_scope='global' or "
                    "remove the target/outcome collision."
                )
            if logger is not None:
                logger.warning(msg)
            return feature_selector, "global_reused_with_warning", [msg]
        if target_col not in derivation_df.columns:
            msg = (
                f"feature_selector reused from global pipeline: target "
                f"column '{target_col}' missing from derivation rows."
            )
            if feature_selector_scope == "per_split":
                raise ValueError(msg)
            if logger is not None:
                logger.warning(msg)
            return feature_selector, "global_reused_with_warning", [msg]
        y_fs = derivation_df[target_col].values
    else:
        y_fs = None

    try:
        from ...feature_selection import MixedDataFeatureSelector

        fresh = MixedDataFeatureSelector(
            fs_cfg,
            continuous_cols=config.continuous_columns,
            categorical_cols=config.categorical_columns,
        )
        feature_cols = list(config.continuous_columns) + list(config.categorical_columns)
        cols_present = [c for c in feature_cols if c in derivation_df.columns]
        data_for_fs = derivation_df[cols_present].copy()
        fresh.fit(data_for_fs, y=y_fs)
        return fresh, "per_split_refit", []
    except (ValueError, RuntimeError, AttributeError, KeyError) as exc:
        msg = f"feature_selector refit failed ({type(exc).__name__}): {exc}; using global selector."
        if feature_selector_scope == "per_split":
            raise
        if logger is not None:
            logger.warning(msg)
        return feature_selector, "global_reused_with_warning", [msg]


@dataclass
class DerivationFitResult:
    """Container for a per-split derivation-only fit.

    Attributes
    ----------
    preprocessor : DataPreprocessor
        Fitted on derivation rows only.
    model : object
        Fresh StepMix-like model fit on the derivation X.
    processed_df : pd.DataFrame
        Output of ``preprocessor.transform_preprocess`` on derivation rows.
    X : np.ndarray
        Feature matrix passed to ``model.fit``.
    raw_labels : np.ndarray
        Cluster labels assigned to derivation rows by the fresh model
        (pre-alignment).
    aligned_labels : np.ndarray
        Same labels, Hungarian-aligned to the global full-cohort labels
        provided at fit time. Used everywhere downstream so phenotype IDs
        are comparable across splits.
    label_mapping : dict
        Mapping ``fresh_label -> global_label`` produced by Hungarian
        alignment.
    ari_to_global : float
        Adjusted Rand Index between the fresh derivation labels and the
        global full-cohort labels on the same rows. Reported as a
        sanity check (high ARI = the per-split fit recovers the same
        structure as the descriptive global model on this subset).
    n_clusters : int
        Number of components on the fresh model.
    log_likelihood : float
        ``model.score(X)`` of the fresh fit.
    notes : list of str
        Free-form warnings (e.g., refit failed and we fell back to
        global model).
    feature_selector : object, optional
        Selector instance used for this split. When the split refit the
        selector this is a fresh one fit on derivation rows; otherwise
        it is the global pipeline's selector (or ``None`` if feature
        selection is disabled).
    feature_selector_mode : str, default "none"
        Resolved selector mode for this split: ``"none"``,
        ``"global_reused"``, ``"global_reused_with_warning"``, or
        ``"per_split_refit"``.
    """

    preprocessor: Any
    model: Any
    processed_df: pd.DataFrame
    X: np.ndarray
    raw_labels: np.ndarray
    aligned_labels: np.ndarray
    label_mapping: Dict[int, int]
    ari_to_global: float
    n_clusters: int
    log_likelihood: float
    notes: List[str] = field(default_factory=list)
    feature_selector: Any = None
    feature_selector_mode: str = "none"


def _build_measurement_dict(config, has_missing: bool, n_continuous: int, n_categorical: int):
    """Mirror :class:`TrainingStage._build_measurement_dict`."""
    use_nan_models = has_missing and not config.imputation.enabled
    measurement = {}
    if n_continuous > 0:
        model_type = "continuous_nan" if use_nan_models else "continuous"
        measurement["continuous"] = {"model": model_type, "n_columns": n_continuous}
    if n_categorical > 0:
        model_type = "categorical_nan" if use_nan_models else "categorical"
        measurement["categorical"] = {"model": model_type, "n_columns": n_categorical}
    return measurement


def _instantiate_stepmix_like(reference_model, n_components: int, random_state: int, measurement):
    """Build a fresh estimator matching the reference model's class.

    Falls back to a vanilla constructor if the reference does not expose
    ``get_params``.
    """
    cls = type(reference_model)
    if hasattr(reference_model, "get_params"):
        params = dict(reference_model.get_params(deep=False))
    else:
        params = {}
    params["n_components"] = n_components
    params["random_state"] = random_state
    if "measurement" in params or measurement:
        params["measurement"] = measurement
    try:
        return cls(**params)
    except TypeError:
        return cls(
            n_components=n_components,
            measurement=measurement,
            random_state=random_state,
        )


def _select_feature_matrix(
    preprocessor: DataPreprocessor,
    processed_df: pd.DataFrame,
    feature_selector,
    continuous_columns: List[str],
    categorical_columns: List[str],
    fallback_X: np.ndarray,
) -> np.ndarray:
    if feature_selector is None:
        return fallback_X
    selected = feature_selector.get_selected_features()
    sel_cont = [c for c in continuous_columns if c in selected]
    sel_cat = [c for c in categorical_columns if c in selected]
    return preprocessor.get_feature_matrix(processed_df, sel_cont, sel_cat)


def fit_derivation_only(
    derivation_df: pd.DataFrame,
    *,
    config,
    reference_model,
    n_clusters: int,
    feature_selector,
    global_labels_on_derivation: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    feature_selector_scope: str = "auto",
    logger=None,
) -> DerivationFitResult:
    """Fit a fresh preprocessor + StepMix on ``derivation_df`` only.

    Parameters
    ----------
    derivation_df : pd.DataFrame
        Raw rows that constitute the derivation cohort for a single split.
    config : PhenoClusterConfig
        Used to read ``continuous_columns``, ``categorical_columns`` and
        ``imputation.enabled`` for the measurement model.
    reference_model
        The global StepMix model. Used solely for type lookup and to seed
        hyperparameters via ``get_params``; never refit and never used to
        produce labels for the derivation rows.
    n_clusters : int
        Number of components for the fresh fit (mirrors the global model).
    feature_selector
        Fitted feature selector from the global pipeline. Depending on
        ``feature_selector_scope`` it is either reused as-is or refit on
        the derivation rows by
        :func:`_resolve_feature_selector_for_split`.
    global_labels_on_derivation : np.ndarray, optional
        Full-cohort labels restricted to ``derivation_df`` row order. When
        provided, the fresh labels are Hungarian-aligned to these and an
        ARI is reported. When omitted, alignment is the identity mapping.
    random_state : int, optional
        Seed for the fresh fit. Defaults to ``config.random_state``.
    feature_selector_scope : {"auto", "global", "per_split"}, default "auto"
        Resolution policy for the feature selector. ``"global"`` always
        reuses ``feature_selector``; ``"per_split"`` always refits and
        raises if not safe; ``"auto"`` refits when safe and falls back to
        the global selector with a warning otherwise.
    logger
        Optional logger for warning messages.

    Returns
    -------
    DerivationFitResult
        Container with the fitted preprocessor, model, derivation labels
        (raw and Hungarian-aligned to the global model), the resolved
        feature selector, and the ``feature_selector_mode`` tag.
    """
    rs = config.random_state if random_state is None else int(random_state)
    notes: List[str] = []

    preprocessor = DataPreprocessor(config)
    preprocessor.fit_imputer(derivation_df)
    imputed = preprocessor.transform_impute(derivation_df)
    preprocessor.fit_outlier_handler(imputed)
    outlier_handled = preprocessor.transform_outliers(imputed)
    preprocessor.fit_preprocessor(outlier_handled)
    processed_df, X_full = preprocessor.transform_preprocess(outlier_handled)

    selector_to_use, fs_mode, fs_notes = _resolve_feature_selector_for_split(
        config, feature_selector, derivation_df, feature_selector_scope, logger=logger
    )
    notes.extend(fs_notes)

    X = _select_feature_matrix(
        preprocessor,
        processed_df,
        selector_to_use,
        config.continuous_columns,
        config.categorical_columns,
        fallback_X=X_full,
    )

    has_missing = bool(np.isnan(X).any()) if isinstance(X, np.ndarray) else False
    n_total = X.shape[1]
    if selector_to_use is not None:
        selected = selector_to_use.get_selected_features()
        n_cont = sum(1 for c in config.continuous_columns if c in selected)
    else:
        n_cont = len(config.continuous_columns)
    n_cat = max(0, n_total - n_cont)
    measurement = _build_measurement_dict(config, has_missing, n_cont, n_cat)

    model = _instantiate_stepmix_like(reference_model, n_clusters, rs, measurement)
    try:
        model.fit(X)
    except (ValueError, RuntimeError, AttributeError) as exc:
        if logger is not None:
            logger.warning(
                f"Generalizability: derivation-only fit failed "
                f"({type(exc).__name__}): {exc}; "
                "falling back to the global model for this split."
            )
        notes.append(f"derivation_only_fit_failed ({type(exc).__name__}): {exc}")
        return DerivationFitResult(
            preprocessor=preprocessor,
            model=reference_model,
            processed_df=processed_df,
            X=X,
            raw_labels=np.asarray(reference_model.predict(X)),
            aligned_labels=np.asarray(reference_model.predict(X)),
            label_mapping={int(k): int(k) for k in range(n_clusters)},
            ari_to_global=float("nan"),
            n_clusters=int(n_clusters),
            log_likelihood=float(reference_model.score(X)),
            notes=notes,
            feature_selector=selector_to_use,
            feature_selector_mode=fs_mode,
        )

    raw_labels = np.asarray(model.predict(X))
    log_likelihood = float(model.score(X))

    if global_labels_on_derivation is not None and len(global_labels_on_derivation) == len(
        raw_labels
    ):
        global_arr = np.asarray(global_labels_on_derivation)
        n_global = int(global_arr.max()) + 1 if global_arr.size else n_clusters
        n_local = int(raw_labels.max()) + 1 if raw_labels.size else n_clusters
        alignment = hungarian_alignment(
            global_arr, raw_labels, n_deriv_clusters=n_global, n_val_clusters=n_local
        )
        aligned = alignment["aligned_labels"]
        mapping = alignment["mapping"]
        try:
            ari = float(adjusted_rand_score(global_arr, raw_labels))
        except (ValueError, AttributeError):
            ari = float("nan")
    else:
        aligned = raw_labels.copy()
        mapping = {
            int(k): int(k) for k in range(int(raw_labels.max()) + 1 if raw_labels.size else 0)
        }
        ari = float("nan")

    return DerivationFitResult(
        preprocessor=preprocessor,
        model=model,
        processed_df=processed_df,
        X=X,
        raw_labels=raw_labels,
        aligned_labels=np.asarray(aligned),
        label_mapping=mapping,
        ari_to_global=ari,
        n_clusters=int(n_clusters),
        log_likelihood=log_likelihood,
        notes=notes,
        feature_selector=selector_to_use,
        feature_selector_mode=fs_mode,
    )


def apply_to_validation(
    fit: DerivationFitResult,
    validation_df: pd.DataFrame,
    *,
    config,
    feature_selector=None,
) -> Dict[str, Any]:
    """Apply a per-split derivation fit to a validation dataframe.

    Parameters
    ----------
    fit : DerivationFitResult
        Output of :func:`fit_derivation_only` for the corresponding
        derivation cohort.
    validation_df : pd.DataFrame
        Untransformed validation rows (same column schema as the
        derivation CSV).
    config : PhenoClusterConfig
        Used to look up ``continuous_columns`` and
        ``categorical_columns`` for the feature-selector pruning step.
    feature_selector
        Optional fallback selector. By default the per-split selector
        carried on ``fit.feature_selector`` is used so that the same
        feature subset that produced the derivation labels is applied
        to the validation rows.

    Returns
    -------
    dict
        Mirrors :func:`apply_validator.apply_to_cohort`'s output with
        keys ``raw_labels``, ``labels`` (Hungarian-aligned to global
        phenotype IDs), ``proba``, ``log_likelihood``, ``processed_df``,
        ``X``, ``classification_quality``, ``n_samples`` and
        ``n_processed``.
    """
    imputed = fit.preprocessor.transform_impute(validation_df)
    outlier_handled = fit.preprocessor.transform_outliers(imputed)
    processed_df, X_full = fit.preprocessor.transform_preprocess(outlier_handled)
    selector_for_val = feature_selector if feature_selector is not None else fit.feature_selector
    X = _select_feature_matrix(
        fit.preprocessor,
        processed_df,
        selector_for_val,
        config.continuous_columns,
        config.categorical_columns,
        fallback_X=X_full,
    )
    raw_labels = np.asarray(fit.model.predict(X))
    proba = np.asarray(fit.model.predict_proba(X))
    log_likelihood = float(fit.model.score(X))

    aligned = np.array(
        [fit.label_mapping.get(int(lab), int(lab)) for lab in raw_labels],
        dtype=int,
    )

    from ...pipeline.quality import compute_classification_quality

    classification_quality = compute_classification_quality(proba, aligned)

    return {
        "raw_labels": raw_labels,
        "labels": aligned,
        "proba": proba,
        "log_likelihood": log_likelihood,
        "processed_df": processed_df,
        "X": X,
        "classification_quality": classification_quality,
        "n_samples": int(len(validation_df)),
        "n_processed": int(len(processed_df)),
    }
