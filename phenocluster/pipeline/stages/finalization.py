"""Finalization stage: feature importance, external validation, visualization, results."""

import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from ...visualization import Visualizer
from ..context import PipelineContext
from ..quality import compute_classification_quality


class FinalizationStage:
    """Steps 11-13: Feature importance, external validation, visualization, result compilation."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def run(
        self,
        ctx: PipelineContext,
        evaluator,
        model_selector,
        preprocessor,
        feature_selector,
        split_result,
        reference_phenotype,
    ) -> dict:
        """Execute finalization. Returns compiled results dict."""
        data_processed = ctx.data_processed
        labels = ctx.labels
        n_clusters = ctx.n_clusters
        proba = ctx.proba

        # Step 11: Feature importance
        feature_importance = evaluator.compute_feature_importance(data_processed, labels)
        if feature_importance:
            feature_importance["top_features_per_cluster"] = evaluator.get_top_features_per_cluster(
                feature_importance,
                feature_char_config=self.config.feature_characterization,
            )

        # Step 11b: External validation (if configured)
        self._run_external_validation(ctx, preprocessor, feature_selector, reference_phenotype)

        # Step 12: Create visualizations
        visualizer = Visualizer(self.config, n_clusters)

        selection_results = (
            model_selector.get_selection_results() if model_selector is not None else None
        )
        plots = visualizer.create_all_plots(
            data_processed,
            labels,
            selection_results,
            ctx.stability_results,
            outcome_results=ctx.outcome_results,
            survival_results=ctx.survival_results,
            multistate_results=ctx.multistate_results,
            posterior_probs=proba,
            posterior_probs_test=ctx.proba_test,
            labels_test=ctx.labels_test,
            reference_phenotype=reference_phenotype,
        )

        # Step 13: Add cluster labels to data
        data_result = data_processed.copy()
        data_result["phenotype"] = labels
        for i in range(n_clusters):
            data_result[f"phenotype_prob_{i}"] = proba[:, i]

        # Compile results
        results = {
            "data": data_result,
            "posterior_proba": proba,
            "model": ctx.model,
            "model_selection": ctx.selection_results,
            "model_fit_metrics": ctx.model_fit_metrics,
            "test_metrics": ctx.test_metrics,
            "cluster_stats": ctx.cluster_stats,
            "stability_results": ctx.stability_results,
            "outcome_results": ctx.outcome_results,
            "survival_results": ctx.survival_results,
            "multistate_results": ctx.multistate_results,
            "validation_metrics": ctx.validation_metrics,
            "classification_quality": ctx.classification_quality,
            "classification_quality_test": ctx.classification_quality_test,
            "feature_importance": feature_importance,
            "external_validation_results": ctx.external_validation_results,
            "quality_report": ctx.quality_report,
            "split_info": ctx.split_info,
            "feature_selection": ctx.feature_selection_report,
            "plots": plots,
            "missing_info": ctx.missing_info,
            "row_filter_info": {
                "enabled": self.config.row_filter.enabled,
                "max_missing_pct": self.config.row_filter.max_missing_pct,
                "n_rows_original": ctx.n_rows_original,
                "n_rows_after_filter": len(ctx.data_filtered),
            },
            "config": self.config.to_dict(),
            "n_clusters": n_clusters,
            "n_samples": len(ctx.X),
            "reference_phenotype": reference_phenotype,
        }

        self._log_results_summary(ctx, n_clusters, plots)

        return results

    def _log_results_summary(self, ctx, n_clusters, plots):
        """Log a summary of pipeline results."""
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info(f"End time: {pd.Timestamp.now()}")
        self.logger.info("Results include:")
        self.logger.info(f"  - Phenotype labels ({n_clusters} clusters)")
        self.logger.info("  - Posterior probability matrix")
        self.logger.info("  - Fitted model")
        if ctx.quality_report:
            self.logger.info("  - Data quality assessment")
        self.logger.info(
            f"  - Data split: train={ctx.split_info['train_size']}, "
            f"test={ctx.split_info['test_size']}"
        )
        if ctx.feature_selection_report:
            self.logger.info(
                f"  - Feature selection: "
                f"{ctx.feature_selection_report['n_selected']}/"
                f"{ctx.feature_selection_report['n_original']} features"
            )
        if ctx.selection_results:
            self.logger.info("  - Model selection results")
        self.logger.info("  - Model fit metrics")
        self.logger.info("  - Cluster statistics")
        if ctx.stability_results:
            self.logger.info("  - Stability analysis results")
        if ctx.outcome_results:
            self.logger.info("  - Outcome association results")
        if ctx.survival_results:
            self.logger.info(
                f"  - Survival analysis results ({len(ctx.survival_results)} endpoint(s))"
            )
        if ctx.multistate_results:
            n_trans = len(ctx.multistate_results.get("transition_results", {}))
            n_paths = len(ctx.multistate_results.get("pathway_results", []))
            self.logger.info(f"  - Multistate analysis ({n_trans} transitions, {n_paths} pathways)")
        self.logger.info("  - Feature importance analysis")
        if ctx.external_validation_results:
            self.logger.info("  - External validation results")
        self.logger.info(f"  - Visualizations ({len(plots)} plots)")

    def _run_external_validation(self, ctx, preprocessor, feature_selector, reference_phenotype):
        """Run external validation on independent cohort if configured."""
        ev_config = self.config.external_validation
        if not ev_config.enabled or not ev_config.external_data_path:
            return

        path = Path(ev_config.external_data_path)
        if not path.exists():
            self.logger.warning(f"External data file not found: {path}")
            return

        self.logger.info(f"EXTERNAL VALIDATION: Loading {path}...")
        external_raw = pd.read_csv(path)

        try:
            ext_imp = preprocessor.transform_impute(external_raw)
            ext_out = preprocessor.transform_outliers(ext_imp)
            ext_processed, X_ext = preprocessor.transform_preprocess(ext_out)

            if feature_selector is not None:
                selected = feature_selector.get_selected_features()
                sel_cont = [c for c in self.config.continuous_columns if c in selected]
                sel_cat = [c for c in self.config.categorical_columns if c in selected]
                X_ext = preprocessor.get_feature_matrix(ext_processed, sel_cont, sel_cat)
        except Exception as e:
            self.logger.warning(
                f"External validation preprocessing failed: {e}\n{traceback.format_exc()}"
            )
            return

        from ...evaluation.external_validation import ExternalValidator

        validator = ExternalValidator(self.config, ctx.n_clusters)
        results = validator.validate_with_model(
            X_external=X_ext,
            model=ctx.model,
            derivation_labels=ctx.labels,
            derivation_outcomes=ctx.outcome_results,
            n_external=len(external_raw),
            external_df=ext_processed,
        )

        external_labels = np.array(results["external_labels"])
        ref = reference_phenotype

        # Classification quality for external cohort
        external_proba = ctx.model.predict_proba(X_ext)
        ext_cq = compute_classification_quality(external_proba, external_labels)
        results["classification_quality"] = ext_cq
        self.logger.info(f"  External classification quality: AvePP={ext_cq['overall_avepp']:.3f}")

        # External survival, multistate, and plots
        ext_survival_results = self._external_survival(ctx, ext_processed, external_labels, ref)
        if ext_survival_results:
            results["survival_results"] = ext_survival_results

        self._external_multistate(ctx, results, ext_processed, external_labels, ref)

        ext_plots = self._external_plots(ctx, results, ext_survival_results, ref)
        if ext_plots:
            results["plots"] = ext_plots

        ctx.external_validation_results = results

    def _external_survival(self, ctx, ext_processed, external_labels, ref):
        """Run survival analysis on external cohort."""
        if not self.config.survival.enabled or not self.config.survival.targets:
            return {}

        self.logger.info("Running survival analysis on external cohort...")
        from ...evaluation.survival import SurvivalAnalyzer

        ext_surv_analyzer = SurvivalAnalyzer(self.config, ctx.n_clusters, reference_phenotype=ref)
        ext_survival_results = {}
        for target in self.config.survival.targets:
            if (
                target.time_column in ext_processed.columns
                and target.event_column in ext_processed.columns
            ):
                try:
                    ext_surv = ext_surv_analyzer.analyze_survival(
                        data=ext_processed,
                        labels=external_labels,
                        time_column=target.time_column,
                        event_column=target.event_column,
                    )
                    ext_survival_results[target.name] = ext_surv
                    self.logger.info(f"  External survival: {target.name} OK")
                except Exception as e:
                    self.logger.warning(f"External survival failed for {target.name}: {e}")
            else:
                self.logger.warning(
                    f"External data missing columns for survival target '{target.name}'"
                )
        return ext_survival_results

    def _external_multistate(self, ctx, results, ext_processed, external_labels, ref):
        """Run multistate analysis on external cohort."""
        if not self.config.multistate.enabled:
            return

        self.logger.info("Running multistate analysis on external cohort...")
        try:
            from ...evaluation.multistate import MultistateAnalyzer

            ext_ms_analyzer = MultistateAnalyzer(
                self.config, ctx.n_clusters, reference_phenotype=ref
            )
            ext_ms_results = ext_ms_analyzer.run_full_analysis(ext_processed, external_labels)
            results["multistate_results"] = ext_ms_analyzer.results_to_dict(ext_ms_results)
            self.logger.info("  External multistate analysis OK")
        except Exception as e:
            self.logger.warning(
                f"External multistate analysis failed: {e}\n{traceback.format_exc()}"
            )

    def _external_plots(self, ctx, results, ext_survival_results, ref):
        """Create plots for external validation results."""
        ext_plots = {}
        ms_res = results.get("multistate_results")
        if not ext_survival_results and not ms_res:
            return ext_plots

        ext_visualizer = Visualizer(self.config, ctx.n_clusters)

        for target_name, surv_result in ext_survival_results.items():
            try:
                km = ext_visualizer.create_kaplan_meier_plot(surv_result, target_name=target_name)
                if km:
                    ext_plots[f"ext_kaplan_meier_{target_name}"] = km
                na = ext_visualizer.create_nelson_aalen_plot(surv_result, target_name=target_name)
                if na:
                    ext_plots[f"ext_nelson_aalen_{target_name}"] = na
            except Exception as e:
                self.logger.warning(f"External survival plot failed for {target_name}: {e}")

        if ms_res:
            trans_res = ms_res.get("transition_results", {})
            if trans_res:
                try:
                    fig = ext_visualizer.create_transition_hazard_forest_plot(
                        trans_res, reference_phenotype=ref
                    )
                    if fig:
                        ext_plots["ext_multistate_transition_hazards"] = fig
                except Exception as e:
                    self.logger.warning(f"External multistate HR plot failed: {e}")
            state_occ = ms_res.get("state_occupation_probabilities")
            if state_occ:
                try:
                    fig = ext_visualizer.create_state_occupation_uncertainty_plot(state_occ)
                    if fig:
                        ext_plots["ext_multistate_state_occupation_uncertainty"] = fig
                except Exception as e:
                    self.logger.warning(f"External state occupation plot failed: {e}")

        return ext_plots
