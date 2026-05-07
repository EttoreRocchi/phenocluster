"""
PhenoCluster Generalization Stage
=================================

Pipeline stage (v0.3.0) that runs the temporal, multi-site and
external-cohort generalizability evaluation after the main pipeline
finishes. With ``training_scope="per_split"`` (default), for each
in-CSV (derivation, validation) split a fresh preprocessor and StepMix
model are fit on the derivation rows only and then applied to the
validation rows; the pipeline's full-cohort model is left untouched for
the descriptive analyses elsewhere in the report. With
``training_scope="global"``, the full-cohort model is used directly.

External CSV cohorts (``generalizability.external_cohorts``) are always
scored by the global model since it never saw those rows.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...config import DataSplitConfig
from ...core.exceptions import DataSplitError, PhenoClusterError
from ...data.splitting import (
    HoldoutGroupSplitter,
    LeaveOneGroupOutSplitter,
    TemporalSplitter,
)
from ...data.splitting._base import BaseSplitter
from ...evaluation.generalizability import GeneralizabilityEvaluator
from ...evaluation.generalizability._types import CohortReport, GeneralizabilityReport
from ...evaluation.generalizability.derivation_fit import fit_derivation_only
from ..context import PipelineContext


class GeneralizationStage:
    """Run temporal, multi-site and external-cohort generalizability assessment.

    The stage is a no-op when ``config.generalizability.enabled`` is False.
    """

    def __init__(self, config, logger):
        self.config = config
        self.gen_cfg = config.generalizability
        self.logger = logger

    def run(
        self,
        ctx: PipelineContext,
        preprocessor,
        feature_selector,
    ) -> None:
        """Populate ``ctx.generalizability_results`` (when enabled)."""
        if not self.gen_cfg.enabled:
            return
        if ctx.model is None or ctx.labels is None:
            self.logger.warning("Generalization stage skipped: derivation model or labels missing.")
            return

        source_df = self._prepare_source_df(ctx)
        self._log_run_header()

        bucket: Dict[str, List[CohortReport]] = {
            "temporal": [],
            "multisite": [],
            "external": [],
        }
        for kind, report in self._run_in_csv(ctx, preprocessor, feature_selector, source_df):
            bucket[kind].append(report)
        for kind, report in self._run_external(ctx, preprocessor, feature_selector):
            bucket[kind].append(report)

        if not (bucket["temporal"] or bucket["multisite"] or bucket["external"]):
            self.logger.warning("Generalizability evaluation produced no reports.")
            return

        final_report = GeneralizabilityReport(
            temporal=bucket["temporal"],
            multisite=bucket["multisite"],
            external=bucket["external"],
            summary=self._build_summary(bucket),
        )
        ctx.generalizability_results = final_report.to_dict()
        self._log_summary(final_report)

    def _prepare_source_df(self, ctx: PipelineContext) -> Optional[pd.DataFrame]:
        source_df = ctx.data_filtered if ctx.data_filtered is not None else ctx.data_raw
        if source_df is None and (
            self.gen_cfg.temporal is not None or self.gen_cfg.multisite is not None
        ):
            self.logger.warning("Generalization stage skipped: no source dataframe.")
            return None
        if source_df is not None:
            return source_df.reset_index(drop=True)
        return None

    def _log_run_header(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("GENERALIZABILITY EVALUATION")
        self.logger.info(
            f"  Mode: training_scope={self.gen_cfg.training_scope}, refit={self.gen_cfg.refit}"
        )
        self.logger.info("=" * 70)

    def _run_in_csv(
        self,
        ctx: PipelineContext,
        preprocessor,
        feature_selector,
        source_df: Optional[pd.DataFrame],
    ) -> Iterable[Tuple[str, CohortReport]]:
        """Yield ``(bucket_key, report)`` for every temporal/multisite split."""
        if source_df is None:
            return
        in_csv_splits: List[Tuple[pd.DataFrame, pd.DataFrame, str, str]] = []
        if self.gen_cfg.temporal is not None:
            in_csv_splits.extend(self._build_temporal_splits(source_df))
        if self.gen_cfg.multisite is not None:
            in_csv_splits.extend(self._build_multisite_splits(source_df))
        for deriv_df, val_df, label, kind in in_csv_splits:
            try:
                report = self._evaluate_in_csv_split(
                    ctx, preprocessor, feature_selector, deriv_df, val_df, label, kind
                )
            except Exception as exc:
                # Cohort-level isolation: one bad split must not abort the run.
                self.logger.warning(
                    f"Generalizability split '{label}' aborted ({type(exc).__name__}): {exc}"
                )
                continue
            if report is not None:
                yield self._route_kind(kind), report

    def _run_external(
        self,
        ctx: PipelineContext,
        preprocessor,
        feature_selector,
    ) -> Iterable[Tuple[str, CohortReport]]:
        """Yield ``(bucket_key, report)`` for every external CSV cohort."""
        for val_df, label, kind, source in self._collect_external_cohorts():
            try:
                report = self._evaluate_external_cohort(
                    ctx, preprocessor, feature_selector, val_df, label, kind, source
                )
            except Exception as exc:
                self.logger.warning(
                    f"External cohort '{label}' aborted ({type(exc).__name__}): {exc}"
                )
                continue
            if report is not None:
                yield self._route_kind(kind), report

    def _route_kind(self, kind: str) -> str:
        """Map a cohort 'kind' string to the result bucket key."""
        if kind == "temporal":
            return "temporal"
        if kind == "site":
            return "multisite"
        return "external"

    def _build_temporal_splits(
        self, source_df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, str, str]]:
        spec = self.gen_cfg.temporal
        splitter: BaseSplitter
        try:
            splitter = TemporalSplitter(
                DataSplitConfig(
                    strategy="temporal",
                    time_column=spec.time_column,
                    time_cutoff=spec.time_cutoff,
                    time_test_fraction=spec.time_test_fraction,
                    time_scheme=spec.scheme,
                    n_windows=spec.n_windows,
                    random_state=self.config.random_state,
                )
            )
        except (ValueError, PhenoClusterError) as exc:
            self.logger.warning(f"Temporal generalizability disabled: {exc}")
            return []

        out: List[Tuple[pd.DataFrame, pd.DataFrame, str, str]] = []
        try:
            for split in splitter.iter_splits(source_df):
                label = split.partition_label or "temporal"
                out.append((split.train, split.test, label, "temporal"))
                self.logger.info(
                    f"  Temporal cohort '{label}': "
                    f"derivation n={split.n_train}, validation n={split.n_test}"
                )
        except PhenoClusterError as exc:
            self.logger.warning(f"Temporal split failed: {exc}")
        return out

    def _build_multisite_splits(
        self, source_df: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, str, str]]:
        spec = self.gen_cfg.multisite
        splitter: BaseSplitter
        if spec.scheme == "holdout":
            try:
                splitter = HoldoutGroupSplitter(
                    DataSplitConfig(
                        strategy="holdout_group",
                        group_column=spec.site_column,
                        holdout_values=spec.holdout_sites or [],
                        min_validation_size=spec.min_site_size,
                        random_state=self.config.random_state,
                    )
                )
            except (ValueError, PhenoClusterError) as exc:
                self.logger.warning(f"Multi-site holdout disabled: {exc}")
                return []
            try:
                split = splitter.split(source_df)
            except PhenoClusterError as exc:
                self.logger.warning(f"Multi-site holdout split failed: {exc}")
                return []
            label = split.partition_label or f"holdout={spec.holdout_sites}"
            self.logger.info(
                f"  Multi-site cohort '{label}': "
                f"derivation n={split.n_train}, validation n={split.n_test}"
            )
            return [(split.train, split.test, label, "site")]

        try:
            splitter = LeaveOneGroupOutSplitter(
                DataSplitConfig(
                    strategy="leave_one_group_out",
                    group_column=spec.site_column,
                    min_validation_size=spec.min_site_size,
                    random_state=self.config.random_state,
                )
            )
        except (ValueError, PhenoClusterError) as exc:
            self.logger.warning(f"Multi-site LOGO disabled: {exc}")
            return []

        out: List[Tuple[pd.DataFrame, pd.DataFrame, str, str]] = []
        try:
            for split in splitter.iter_splits(source_df):
                label = split.partition_label or "site"
                out.append((split.train, split.test, label, "site"))
                self.logger.info(
                    f"  Multi-site cohort '{label}': "
                    f"derivation n={split.n_train}, validation n={split.n_test}"
                )
        except PhenoClusterError as exc:
            self.logger.warning(f"Multi-site LOGO split failed: {exc}")
        return out

    def _collect_external_cohorts(
        self,
    ) -> List[Tuple[pd.DataFrame, str, str, str]]:
        from ...utils.io import safe_read_csv

        specs = self.gen_cfg.external_cohorts or []
        out: List[Tuple[pd.DataFrame, str, str, str]] = []
        for spec in specs:
            path = Path(spec.path).expanduser()
            try:
                df = safe_read_csv(path)
            except DataSplitError as exc:
                self.logger.warning(f"External cohort '{spec.label}' skipped: {exc}")
                continue
            except (
                pd.errors.ParserError,
                UnicodeDecodeError,
                OSError,
            ) as exc:
                self.logger.warning(
                    f"External cohort '{spec.label}' could not be loaded "
                    f"({type(exc).__name__}): {exc}"
                )
                continue
            self.logger.info(
                f"  External cohort '{spec.label}' (kind={spec.kind}): "
                f"loaded {len(df)} rows from {path}"
            )
            out.append((df, spec.label, spec.kind, str(path)))
        return out

    def _evaluate_in_csv_split(
        self,
        ctx,
        preprocessor,
        feature_selector,
        deriv_df: pd.DataFrame,
        val_df: pd.DataFrame,
        label: str,
        kind: str,
    ) -> Optional[CohortReport]:
        if self.gen_cfg.training_scope == "per_split":
            return self._evaluate_per_split(
                ctx, preprocessor, feature_selector, deriv_df, val_df, label, kind
            )
        return self._evaluate_with_global_model(
            ctx,
            preprocessor,
            feature_selector,
            val_df,
            label,
            kind,
            warn=(
                "training_scope=global: derivation model was fit on all rows; "
                "in-CSV split metrics share that model with the validation cohort."
            ),
        )

    def _evaluate_per_split(
        self,
        ctx,
        preprocessor,
        feature_selector,
        deriv_df: pd.DataFrame,
        val_df: pd.DataFrame,
        label: str,
        kind: str,
    ) -> Optional[CohortReport]:
        global_labels_on_deriv = self._global_labels_on(
            deriv_df, ctx, preprocessor, feature_selector
        )
        try:
            fit = fit_derivation_only(
                deriv_df,
                config=self.config,
                reference_model=ctx.model,
                n_clusters=ctx.n_clusters,
                feature_selector=feature_selector,
                global_labels_on_derivation=global_labels_on_deriv,
                random_state=self.config.random_state,
                feature_selector_scope=self.gen_cfg.feature_selector_scope,
                logger=self.logger,
            )
        except (AttributeError, ValueError, RuntimeError, PhenoClusterError) as exc:
            self.logger.warning(
                f"Per-split derivation fit failed for cohort '{label}' "
                f"({type(exc).__name__}): {exc}; "
                "falling back to the global model for this split."
            )
            return self._evaluate_with_global_model(
                ctx,
                preprocessor,
                feature_selector,
                val_df,
                label,
                kind,
                warn=f"per_split_fit_failed ({type(exc).__name__}): {exc}",
            )

        deriv_outcomes = self._derivation_only_outcomes(fit.processed_df, fit.aligned_labels)
        deriv_survival = self._derivation_only_survival(fit.processed_df, fit.aligned_labels)

        evaluator = GeneralizabilityEvaluator(
            self.config,
            derivation_labels=fit.aligned_labels,
            derivation_outcomes=({"full_cohort": deriv_outcomes} if deriv_outcomes else {}),
            derivation_survival=deriv_survival,
            derivation_df=fit.processed_df,
            model=fit.model,
            preprocessor=fit.preprocessor,
            feature_selector=fit.feature_selector,
            n_clusters=ctx.n_clusters,
        )
        try:
            report = evaluator.evaluate_cohort(val_df, label=label, kind=kind)
        except (AttributeError, ValueError, KeyError, PhenoClusterError) as exc:
            self.logger.warning(
                f"Generalizability evaluation failed for cohort '{label}' "
                f"({type(exc).__name__}): {exc}"
            )
            return None
        if report is None:
            return None

        report.fit_mode = "per_split"
        report.derivation_only_ari = fit.ari_to_global
        report.feature_selector_mode = fit.feature_selector_mode
        if deriv_outcomes:
            report.derivation_only_outcomes = deriv_outcomes
        for note in fit.notes:
            report.warnings.append(note)
        return report

    def _evaluate_with_global_model(
        self,
        ctx,
        preprocessor,
        feature_selector,
        val_df: pd.DataFrame,
        label: str,
        kind: str,
        *,
        warn: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[CohortReport]:
        evaluator = GeneralizabilityEvaluator(
            self.config,
            derivation_labels=ctx.labels,
            derivation_outcomes=ctx.outcome_results,
            derivation_survival=ctx.survival_results,
            derivation_df=ctx.data_processed,
            model=ctx.model,
            preprocessor=preprocessor,
            feature_selector=feature_selector,
            n_clusters=ctx.n_clusters,
        )
        try:
            report = evaluator.evaluate_cohort(val_df, label=label, kind=kind)
        except (AttributeError, ValueError, KeyError, PhenoClusterError) as exc:
            self.logger.warning(
                f"Generalizability evaluation failed for cohort '{label}' "
                f"({type(exc).__name__}): {exc}"
            )
            return None
        if report is None:
            return None
        report.fit_mode = "global"
        report.feature_selector_mode = "global_reused" if feature_selector is not None else "none"
        if warn:
            report.warnings.append(warn)
            self.logger.warning(f"  cohort '{label}': {warn}")
        if source:
            report.source = source
        return report

    def _evaluate_external_cohort(
        self,
        ctx,
        preprocessor,
        feature_selector,
        val_df: pd.DataFrame,
        label: str,
        kind: str,
        source: str,
    ) -> Optional[CohortReport]:
        return self._evaluate_with_global_model(
            ctx, preprocessor, feature_selector, val_df, label, kind, source=source
        )

    def _global_labels_on(
        self, deriv_df: pd.DataFrame, ctx, preprocessor, feature_selector
    ) -> Optional[np.ndarray]:
        try:
            from ...evaluation.generalizability.apply_validator import apply_to_cohort

            applied = apply_to_cohort(
                deriv_df,
                model=ctx.model,
                preprocessor=preprocessor,
                feature_selector=feature_selector,
                continuous_columns=self.config.continuous_columns,
                categorical_columns=self.config.categorical_columns,
            )
            return np.asarray(applied["labels"])
        except (ValueError, KeyError, AttributeError, PhenoClusterError) as exc:
            self.logger.warning(
                f"Could not compute global labels on derivation rows "
                f"({type(exc).__name__}): {exc}; "
                "Hungarian alignment will be skipped for this split."
            )
            return None

    def _derivation_only_outcomes(
        self, processed_df: pd.DataFrame, labels: np.ndarray
    ) -> Dict[str, Any]:
        if not getattr(self.config, "outcome_columns", None):
            return {}
        if not self.config.outcome.enabled:
            return {}
        try:
            from ...evaluation.outcome_analysis import OutcomeAnalyzer

            analyzer = OutcomeAnalyzer(self.config, n_clusters=int(np.max(labels)) + 1)
            return analyzer.analyze_outcomes(processed_df, labels)
        except (AttributeError, ValueError, KeyError) as exc:
            self.logger.warning(f"Derivation-only outcome ORs failed ({type(exc).__name__}): {exc}")
            return {}

    def _derivation_only_survival(
        self, processed_df: pd.DataFrame, labels: np.ndarray
    ) -> Dict[str, Any]:
        if not self.config.survival.enabled or not self.config.survival.targets:
            return {}
        try:
            from ...evaluation.survival import SurvivalAnalyzer

            ref = getattr(self.config, "reference_phenotype", None)
            ref_id = ref.id if ref is not None and hasattr(ref, "id") else 0
            analyzer = SurvivalAnalyzer(
                self.config, int(np.max(labels)) + 1, reference_phenotype=ref_id
            )
            out: Dict[str, Any] = {}
            for target in self.config.survival.targets:
                try:
                    out[target.name] = analyzer.analyze_survival(
                        data=processed_df,
                        labels=labels,
                        time_column=target.time_column,
                        event_column=target.event_column,
                    )
                except (AttributeError, ValueError, KeyError) as exc:
                    self.logger.warning(
                        f"Derivation-only survival for target '{target.name}' failed "
                        f"({type(exc).__name__}): {exc}"
                    )
            return out
        except (AttributeError, ValueError, KeyError) as exc:
            self.logger.warning(
                f"Derivation-only survival analysis failed ({type(exc).__name__}): {exc}"
            )
            return {}

    def _build_summary(self, bucket: Dict[str, List[CohortReport]]) -> Dict[str, Any]:
        def _finite(values: Iterable[Optional[float]]) -> List[float]:
            return [float(v) for v in values if v is not None and np.isfinite(v)]

        def _agg(reports: List[CohortReport]) -> Dict[str, Any]:
            if not reports:
                return {}
            ari = _finite(r.refit.get("ari") for r in reports if r.refit and "ari" in r.refit)
            psi_means = _finite(
                float(r.drift["psi"].mean())
                for r in reports
                if r.drift is not None and not r.drift.empty
            )
            deriv_aris = _finite(r.derivation_only_ari for r in reports)
            return {
                "n_cohorts": len(reports),
                "mean_ari": float(np.mean(ari)) if ari else None,
                "mean_psi": float(np.mean(psi_means)) if psi_means else None,
                "mean_derivation_only_ari_to_global": (
                    float(np.mean(deriv_aris)) if deriv_aris else None
                ),
            }

        return {
            "temporal": _agg(bucket["temporal"]),
            "multisite": _agg(bucket["multisite"]),
            "external": _agg(bucket["external"]),
            "training_scope": str(self.gen_cfg.training_scope),
        }

    def _log_summary(self, report: GeneralizabilityReport) -> None:
        for kind, items in (
            ("Temporal", report.temporal),
            ("Multi-site", report.multisite),
            ("External", report.external),
        ):
            for cohort in items:
                ari = cohort.refit.get("ari") if cohort.refit else None
                psi_mean = (
                    float(cohort.drift["psi"].mean())
                    if cohort.drift is not None and not cohort.drift.empty
                    else None
                )
                ari_str = f"ARI={ari:.3f}" if ari is not None else "ARI=NA"
                psi_str = f"mean_PSI={psi_mean:.3f}" if psi_mean is not None else "PSI=NA"
                mode_str = f"mode={cohort.fit_mode}" if cohort.fit_mode else ""
                self.logger.info(
                    f"  [{kind}] {cohort.label}: n={cohort.n_samples}, {ari_str}, "
                    f"{psi_str} {mode_str}"
                )
