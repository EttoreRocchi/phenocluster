"""Analysis stage: outcome, survival, and multistate analyses."""

import traceback

from ..context import PipelineContext


class AnalysisStage:
    """Steps 10-10c: Outcome, survival, and multistate analyses."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def run(self, ctx: PipelineContext, evaluator, split_result, reference_phenotype) -> None:
        """Execute analyses. Mutates ctx in place."""
        data_processed = ctx.data_processed
        labels = ctx.labels
        proba = ctx.proba
        n_clusters = ctx.n_clusters
        ref = reference_phenotype

        # Step 10: Outcome association analysis
        ctx.outcome_results = self._run_outcome_analysis(
            evaluator, data_processed, labels, split_result, ref
        )

        # Step 10b: Survival analysis
        ctx.survival_results = self._run_survival_analysis(
            data_processed, labels, proba, n_clusters, ref
        )

        # Step 10c: Multistate analysis
        ctx.multistate_results = self._run_multistate_analysis(
            data_processed, labels, n_clusters, ref
        )

    def _run_outcome_analysis(self, evaluator, data_processed, labels, split_result, ref):
        """Run outcome association analysis on train, test, and full cohort."""
        if not self.config.outcome.enabled:
            self.logger.info("Outcome analysis disabled - skipping")
            return {}

        data_train = data_processed.iloc[split_result.train_indices].reset_index(drop=True)
        labels_train = labels[split_result.train_indices]
        outcome_results = evaluator.analyze_outcomes(
            data_train, labels_train, reference_phenotype=ref
        )

        data_test = data_processed.iloc[split_result.test_indices].reset_index(drop=True)
        labels_test = labels[split_result.test_indices]
        outcome_results_test = evaluator.analyze_outcomes(
            data_test, labels_test, reference_phenotype=ref
        )

        return {
            "train": outcome_results,
            "test": outcome_results_test,
            "full_cohort": evaluator.analyze_outcomes(
                data_processed, labels, reference_phenotype=ref
            ),
        }

    def _run_survival_analysis(self, data_processed, labels, proba, n_clusters, ref):
        """Run survival and weighted survival analyses with FDR correction."""
        if not self.config.survival.enabled or not self.config.survival.targets:
            return {}

        self.logger.info("Running survival analysis (full cohort)...")
        from ...evaluation.survival import SurvivalAnalyzer

        survival_analyzer = SurvivalAnalyzer(self.config, n_clusters, reference_phenotype=ref)
        survival_results = {}

        for target in self.config.survival.targets:
            try:
                result = survival_analyzer.analyze_survival(
                    data=data_processed,
                    labels=labels,
                    time_column=target.time_column,
                    event_column=target.event_column,
                )
                survival_results[target.name] = result
                self.logger.info(f"  {target.name} survival analysis completed")
            except Exception as e:
                self.logger.warning(
                    f"  Survival analysis failed for {target.name} ({type(e).__name__}): {e}"
                )
                self.logger.warning(traceback.format_exc())

        # FDR correction across ALL survival pairwise p-values globally
        self._apply_survival_fdr(survival_results)

        # Weighted survival
        if self.config.survival.use_weighted:
            self._run_weighted_survival(survival_analyzer, data_processed, proba, survival_results)

        if survival_results:
            self.logger.info(
                f"  Completed survival analysis for {len(survival_results)} endpoint(s)"
            )
        return survival_results

    def _apply_survival_fdr(self, survival_results):
        """Apply FDR correction across all survival pairwise p-values."""
        if not self.config.inference.enabled or not self.config.inference.fdr_correction:
            return

        from ...evaluation.stats_utils import apply_fdr_correction

        all_p_entries = []
        for target_name, target_data in survival_results.items():
            comparison = target_data.get("comparison", {})
            for key, val in comparison.items():
                if isinstance(val, dict) and "p_value" in val:
                    all_p_entries.append((target_name, key))

        if all_p_entries:
            raw_p = [survival_results[t]["comparison"][k]["p_value"] for t, k in all_p_entries]
            adjusted = apply_fdr_correction(raw_p)
            for (t, k), q in zip(all_p_entries, adjusted):
                survival_results[t]["comparison"][k]["p_value_fdr"] = q

    def _run_weighted_survival(self, survival_analyzer, data_processed, proba, survival_results):
        """Run weighted survival analysis."""
        self.logger.info("  Running weighted survival analysis (posterior probability weights)...")
        for target in self.config.survival.targets:
            try:
                weighted_result = survival_analyzer.analyze_weighted_survival(
                    data=data_processed,
                    posterior_probs=proba,
                    time_column=target.time_column,
                    event_column=target.event_column,
                )
                survival_results[f"{target.name}_weighted"] = weighted_result
            except Exception as e:
                self.logger.warning(
                    f"  Weighted survival analysis failed for {target.name} "
                    f"({type(e).__name__}): {e}"
                )
                self.logger.warning(traceback.format_exc())

    def _run_multistate_analysis(self, data_processed, labels, n_clusters, ref):
        """Run multistate illness-death analysis."""
        if not self.config.multistate.enabled:
            return {}

        self.logger.info("Running multistate analysis (full cohort)...")
        try:
            from ...evaluation.multistate import MultistateAnalyzer

            multistate_analyzer = MultistateAnalyzer(
                self.config, n_clusters, reference_phenotype=ref
            )
            ms_results = multistate_analyzer.run_full_analysis(data_processed, labels)
            result = multistate_analyzer.results_to_dict(ms_results)

            n_trans = len(ms_results.transition_results)
            n_paths = len(ms_results.pathway_results)
            self.logger.info(f"  Fitted {n_trans} transition model(s)")
            self.logger.info(f"  Observed {n_paths} unique pathway(s)")
            if ms_results.state_occupation_probabilities:
                mc_n = ms_results.state_occupation_probabilities.n_simulations
                self.logger.info(f"  Monte Carlo: {mc_n:,} simulations per phenotype")
            self.logger.info("  Multistate analysis completed")
            return result
        except Exception as e:
            self.logger.warning(f"  Multistate analysis failed ({type(e).__name__}): {e}")
            self.logger.warning(traceback.format_exc())
            return {}
