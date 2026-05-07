"""
PhenoCluster Generalizability Evaluator
=======================================

Facade orchestrating apply -> optional refit-and-match -> calibration ->
drift -> outcome concordance for each validation cohort, returning a
:class:`GeneralizabilityReport`.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.logging import get_logger
from . import refit_validator
from ._types import CohortReport
from .apply_validator import apply_to_cohort
from .calibration import compute_calibration_block
from .drift import feature_drift
from .outcome_concordance import compare_outcomes, compare_survival
from .prevalence import chi2_homogeneity, cluster_distribution


class GeneralizabilityEvaluator:
    """Single-cohort generalizability metric assembler.

    The public entry point is :meth:`evaluate_cohort`, which returns one
    :class:`CohortReport`. Cohort iteration and bucket assembly
    (``temporal`` / ``multisite`` / ``external``) are owned by
    :class:`phenocluster.pipeline.stages.generalization.GeneralizationStage`;
    this class is intentionally cohort-scoped.

    Parameters
    ----------
    config : PhenoClusterConfig
        Full configuration. The ``generalizability`` sub-config drives the
        per-cohort evaluation (refit toggle, calibration bins, drift
        settings, outcome concordance options).
    derivation_labels : np.ndarray
        Phenotype labels of the derivation cohort (full cohort, post-fit).
    derivation_outcomes : dict, optional
        Outcome results from the derivation cohort
        (``OutcomeAnalyzer.analyze`` output). Used for cross-cohort
        concordance.
    derivation_survival : dict, optional
        Survival results from the derivation cohort. Used for cross-cohort
        HR concordance.
    derivation_df : pd.DataFrame, optional
        The processed derivation dataframe (post-preprocessing). Used as the
        reference cohort for drift computation.
    model
        Fitted derivation model (StepMix-like).
    preprocessor
        Fitted derivation preprocessor.
    feature_selector
        Fitted derivation feature selector (or ``None``).
    n_clusters : int
        Number of phenotypes in the derivation cohort.
    """

    def __init__(
        self,
        config,
        *,
        derivation_labels: np.ndarray,
        derivation_outcomes: Optional[Dict[str, Any]] = None,
        derivation_survival: Optional[Dict[str, Any]] = None,
        derivation_df: Optional[pd.DataFrame] = None,
        model,
        preprocessor,
        feature_selector,
        n_clusters: int,
    ):
        self.config = config
        self.gen_cfg = getattr(config, "generalizability", None)
        self.derivation_labels = np.asarray(derivation_labels)
        self.derivation_outcomes = derivation_outcomes or {}
        self.derivation_survival = derivation_survival or {}
        self.derivation_df = derivation_df
        self.model = model
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
        self.n_clusters = int(n_clusters)
        self.logger = get_logger("generalizability", config)

    def evaluate_cohort(
        self,
        raw_df: pd.DataFrame,
        *,
        label: str,
        kind: str,
    ) -> Optional[CohortReport]:
        """Apply the derivation pipeline to ``raw_df`` and assemble metrics.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Untransformed validation cohort with the same column schema as
            the derivation data.
        label : str
            Human-readable identifier for the cohort (e.g.,
            ``"site=ANTWERP"`` or ``"window_2021_2023"``).
        kind : str
            ``"temporal"``, ``"site"``, or ``"external"``. Controls which
            bucket the resulting report is routed to in the
            :class:`GeneralizabilityReport`.

        Returns
        -------
        CohortReport or None
            Populated cohort report. Returns ``None`` if the validation
            cohort could not be preprocessed by the derivation
            preprocessor.
        """
        warnings: List[str] = []

        applied = apply_to_cohort(
            raw_df,
            model=self.model,
            preprocessor=self.preprocessor,
            feature_selector=self.feature_selector,
            continuous_columns=self.config.continuous_columns,
            categorical_columns=self.config.categorical_columns,
        )
        labels = applied["labels"]
        proba = applied["proba"]
        processed_df = applied["processed_df"]
        X_val = applied["X"]
        n_processed = applied["n_processed"]

        report = CohortReport(
            label=label,
            kind=kind,
            n_samples=n_processed,
            cluster_distribution=cluster_distribution(labels),
            log_likelihood=applied["log_likelihood"],
            classification_quality=applied["classification_quality"],
            warnings=warnings,
        )
        report.derivation_distribution = cluster_distribution(self.derivation_labels)

        report.prevalence_chi2 = self._prevalence_chi2(
            report.derivation_distribution, report.cluster_distribution
        )

        if self._is_drift_enabled():
            report.drift = self._compute_drift(processed_df)

        refit_result = None
        if self._refit_enabled():
            refit_result = refit_validator.refit_and_match(
                X_val=X_val,
                derivation_labels_on_val=labels,
                reference_model=self.model,
                n_clusters=self.n_clusters,
                random_state=getattr(self.config, "random_state", 42),
                min_validation_size_for_refit=self._min_validation_size_for_refit(),
                logger=self.logger,
            )
        if refit_result is None and self._refit_enabled():
            warnings.append("refit skipped (cohort below min_validation_size_for_refit)")

        if refit_result is not None and self._calibration_enabled():
            report.calibration = compute_calibration_block(
                proba,
                refit_result["aligned_labels"],
                n_bins=self._calibration_n_bins(),
                strategy=self._calibration_strategy(),
            )
            if report.calibration is not None and report.calibration.get(
                "degenerate_quantile_bins"
            ):
                warnings.append("calibration: degenerate quantile bins, used uniform fallback")
        elif self._calibration_enabled():
            warnings.append("calibration skipped (refit disabled or unavailable)")

        if refit_result is not None:
            report.refit = refit_validator.to_json_safe(refit_result)

        if self._outcome_concordance_enabled():
            report.outcome_concordance = self._compute_outcome_concordance(processed_df, labels)

        return report

    def _prevalence_chi2(
        self,
        derivation_distribution: Dict[int, Dict],
        validation_distribution: Dict[int, Dict],
    ) -> Optional[Dict[str, Any]]:
        if not derivation_distribution or not validation_distribution:
            return None
        return chi2_homogeneity(derivation_distribution, validation_distribution)

    def _compute_drift(self, processed_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.derivation_df is None:
            return None
        n_bins = self._drift_n_bins()
        try:
            return feature_drift(
                self.derivation_df,
                processed_df,
                continuous_cols=self.config.continuous_columns,
                categorical_cols=self.config.categorical_columns,
                n_bins=n_bins,
            )
        except (ValueError, KeyError, AttributeError) as exc:
            self.logger.warning(f"Drift computation failed ({type(exc).__name__}): {exc}")
            return None

    def _compute_outcome_concordance(
        self,
        processed_df: pd.DataFrame,
        validation_labels: np.ndarray,
    ) -> Dict[str, Any]:
        from ..outcome_analysis import OutcomeAnalyzer
        from ..survival import SurvivalAnalyzer

        results: Dict[str, Any] = {}

        deriv_full = (
            self.derivation_outcomes.get("full_cohort", {}) if self.derivation_outcomes else {}
        )
        if deriv_full:
            try:
                analyzer = OutcomeAnalyzer(self.config, self.n_clusters)
                val_outcomes = analyzer.analyze_outcomes(processed_df, validation_labels)
                results["outcomes"] = compare_outcomes(deriv_full, val_outcomes)
            except (ValueError, KeyError, AttributeError) as exc:
                self.logger.warning(f"Outcome concordance failed ({type(exc).__name__}): {exc}")

        if self.derivation_survival and self.config.survival.enabled:
            try:
                ref = getattr(self.config, "reference_phenotype", None)
                ref_id = ref.id if ref is not None and hasattr(ref, "id") else 0
                surv_analyzer = SurvivalAnalyzer(
                    self.config, self.n_clusters, reference_phenotype=ref_id
                )
                val_survival: Dict[str, Any] = {}
                for target in self.config.survival.targets:
                    if (
                        target.time_column not in processed_df.columns
                        or target.event_column not in processed_df.columns
                    ):
                        continue
                    val_survival[target.name] = surv_analyzer.analyze_survival(
                        data=processed_df,
                        labels=validation_labels,
                        time_column=target.time_column,
                        event_column=target.event_column,
                    )
                if val_survival:
                    results["survival"] = compare_survival(self.derivation_survival, val_survival)
            except (ValueError, KeyError, AttributeError) as exc:
                self.logger.warning(f"Survival concordance failed ({type(exc).__name__}): {exc}")

        return results

    def _refit_enabled(self) -> bool:
        return bool(getattr(self.gen_cfg, "refit", True))

    def _calibration_enabled(self) -> bool:
        if self.gen_cfg is None:
            return True
        sub = getattr(self.gen_cfg, "calibration", None)
        return bool(getattr(sub, "enabled", True))

    def _calibration_n_bins(self) -> int:
        sub = getattr(self.gen_cfg, "calibration", None)
        return int(getattr(sub, "n_bins", 10))

    def _calibration_strategy(self) -> str:
        sub = getattr(self.gen_cfg, "calibration", None)
        return str(getattr(sub, "strategy", "quantile"))

    def _is_drift_enabled(self) -> bool:
        if self.gen_cfg is None:
            return True
        sub = getattr(self.gen_cfg, "drift", None)
        return bool(getattr(sub, "enabled", True))

    def _drift_n_bins(self) -> int:
        sub = getattr(self.gen_cfg, "drift", None)
        return int(getattr(sub, "n_bins", 10))

    def _outcome_concordance_enabled(self) -> bool:
        if self.gen_cfg is None:
            return True
        sub = getattr(self.gen_cfg, "outcome_concordance", None)
        return bool(getattr(sub, "enabled", True))

    def _min_validation_size_for_refit(self) -> int:
        if self.gen_cfg is None:
            return 100
        return int(getattr(self.gen_cfg, "min_validation_size_for_refit", 100))
