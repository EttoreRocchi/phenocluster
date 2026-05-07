"""
PhenoCluster Generalizability Result Types
==========================================

Lightweight dataclasses returned by :class:`GeneralizabilityEvaluator`.

The structures are designed to round-trip cleanly to JSON via
:meth:`CohortReport.to_json_safe` so that downstream tools (the static HTML
report and the optional Streamlit dashboard) can consume them without
re-running the evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


def _df_to_records(df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
    if df is None:
        return None
    return df.to_dict(orient="records")


@dataclass
class CohortReport:
    """Generalizability summary for a single validation cohort.

    Parameters
    ----------
    label : str
        Human-readable identifier (e.g., ``"site=ANTWERP"`` or
        ``"window_2018-01-01..2020-12-31"``).
    kind : str
        ``"temporal"`` or ``"site"``.
    n_samples : int
        Validation cohort size after preprocessing.
    cluster_distribution : dict
        Output of :func:`prevalence.cluster_distribution` on the validation
        labels.
    derivation_distribution : dict, optional
        Same statistic on the derivation labels for side-by-side comparison.
    log_likelihood : float
        ``model.score(X_val)`` of the fitted derivation model.
    classification_quality : dict
        Average posterior probability and related metrics on the
        derivation-applied predictions.
    calibration : dict, optional
        Output of :func:`calibration.expected_calibration_error` and friends.
        Populated only when ``refit=True``.
    drift : pd.DataFrame, optional
        Per-feature drift table from :func:`drift.feature_drift`.
    outcome_concordance : dict, optional
        OR/HR concordance tables from :mod:`outcome_concordance`.
    refit : dict, optional
        Output of :func:`refit_validator.refit_and_match` (label-agreement,
        ARI, NMI, mapping, unmatched clusters).
    prevalence_chi2 : dict, optional
        Result of :func:`prevalence.chi2_cohort_comparison` on phenotype
        prevalence between derivation and validation cohorts.
    warnings : list of str
        Per-cohort soft-warnings (e.g., refit skipped due to small cohort).
    fit_mode : str, optional
        Which model produced the metrics for this cohort: ``"per_split"``
        (a fresh derivation-only fit was used; default for in-CSV splits)
        or ``"global"`` (the pipeline's full-cohort model was used;
        default for external CSVs).
    derivation_only_ari : float, optional
        ARI between the per-split derivation-only fit's labels and the
        global full-cohort model's labels on the same derivation rows.
        ``None`` when ``fit_mode="global"`` or when alignment was skipped.
    derivation_only_outcomes : dict, optional
        Per-phenotype outcome regression results (OR + CI) computed on the
        derivation rows of this split using the per-split fit's labels.
        Fed into the cross-cohort outcome concordance comparison.
    source : str, optional
        Free-form provenance string (e.g., the file path for an external
        CSV cohort).
    feature_selector_mode : str, optional
        How the feature selector was applied to this cohort:
        ``"per_split_refit"``, ``"global_reused"``,
        ``"global_reused_with_warning"``, or ``"none"`` (when feature
        selection is disabled).
    """

    label: str
    kind: str
    n_samples: int
    cluster_distribution: Dict[int, Dict] = field(default_factory=dict)
    derivation_distribution: Optional[Dict[int, Dict]] = None
    log_likelihood: Optional[float] = None
    classification_quality: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    drift: Optional[pd.DataFrame] = None
    outcome_concordance: Optional[Dict[str, Any]] = None
    refit: Optional[Dict[str, Any]] = None
    prevalence_chi2: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    fit_mode: Optional[str] = None
    derivation_only_ari: Optional[float] = None
    derivation_only_outcomes: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    feature_selector_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict view that still includes the drift DataFrame."""
        return {
            "label": self.label,
            "kind": self.kind,
            "n_samples": self.n_samples,
            "cluster_distribution": self.cluster_distribution,
            "derivation_distribution": self.derivation_distribution,
            "log_likelihood": self.log_likelihood,
            "classification_quality": self.classification_quality,
            "calibration": self.calibration,
            "drift": self.drift,
            "outcome_concordance": self.outcome_concordance,
            "refit": self.refit,
            "prevalence_chi2": self.prevalence_chi2,
            "warnings": list(self.warnings),
            "fit_mode": self.fit_mode,
            "derivation_only_ari": self.derivation_only_ari,
            "derivation_only_outcomes": self.derivation_only_outcomes,
            "source": self.source,
            "feature_selector_mode": self.feature_selector_mode,
        }

    def to_json_safe(self) -> Dict[str, Any]:
        """Return a JSON-serializable view (DataFrames -> list of records)."""
        d = self.to_dict()
        d["drift"] = _df_to_records(self.drift)
        return d


@dataclass
class GeneralizabilityReport:
    """Top-level report grouping temporal, multi-site, and external cohort reports."""

    temporal: List[CohortReport] = field(default_factory=list)
    multisite: List[CohortReport] = field(default_factory=list)
    external: List[CohortReport] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict view (cohort reports keep their drift DataFrames)."""
        return {
            "temporal": [c.to_dict() for c in self.temporal],
            "multisite": [c.to_dict() for c in self.multisite],
            "external": [c.to_dict() for c in self.external],
            "summary": self.summary,
        }

    def to_json_safe(self) -> Dict[str, Any]:
        """Return a JSON-serializable view (drift DataFrames -> list of records)."""
        return {
            "temporal": [c.to_json_safe() for c in self.temporal],
            "multisite": [c.to_json_safe() for c in self.multisite],
            "external": [c.to_json_safe() for c in self.external],
            "summary": self.summary,
        }
