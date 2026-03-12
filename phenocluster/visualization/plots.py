"""
PhenoCluster Visualization - Facade
====================================

Composes domain-specific visualizer classes into a single ``Visualizer`` facade
that preserves the original public API.

Individual plotting logic lives in:
  - ``_cluster_distribution.py`` - cluster distribution, model selection
  - ``_cluster_heatmap.py``      - continuous/categorical heatmaps, Sankey
  - ``_cluster_quality.py``      - consensus matrix, classification quality
  - ``_survival.py``             - Kaplan-Meier, Nelson-Aalen curves
  - ``_outcome.py``              - odds-ratio forest plots
  - ``_multistate.py``           - pathway, transition HR, state occupation, state diagram
  - ``_base.py``                 - shared style, layout helper, base utilities
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger
from ._cluster_distribution import ClusterDistributionVisualizer
from ._cluster_heatmap import ClusterHeatmapVisualizer
from ._cluster_quality import ClusterQualityVisualizer
from ._multistate import MultistateVisualizer
from ._outcome import OutcomeVisualizer
from ._survival import SurvivalVisualizer


class Visualizer:
    """
    Handles all visualization tasks for phenotype discovery.

    Creates high-quality visualizations suitable for scientific analysis,
    with support for both interactive HTML and static PNG outputs.

    Uses composition to delegate to domain-specific visualizer classes.
    """

    def __init__(self, config: PhenoClusterConfig, n_clusters: int):
        self.config = config
        self.n_clusters = n_clusters
        self.logger = get_logger("visualization", config)

        self._distribution = ClusterDistributionVisualizer(config, n_clusters)
        self._heatmap = ClusterHeatmapVisualizer(config, n_clusters)
        self._quality = ClusterQualityVisualizer(config, n_clusters)
        self._survival = SurvivalVisualizer(config, n_clusters)
        self._outcome = OutcomeVisualizer(config, n_clusters)
        self._multistate = MultistateVisualizer(config, n_clusters)

    def create_cluster_distribution(self, labels, title=None):
        return self._distribution.create_cluster_distribution(labels, title)

    def create_model_selection_plot(self, selection_results, title=None):
        return self._distribution.create_model_selection_plot(selection_results, title)

    def create_heatmap(
        self,
        df,
        labels,
        consensus_matrix=None,
        show_significance=True,
        title=None,
    ):
        return self._heatmap.create_heatmap(df, labels, consensus_matrix, show_significance, title)

    def create_categorical_flow_plots(self, df, labels):
        return self._heatmap.create_categorical_flow_plots(df, labels)

    def create_categorical_heatmap(self, df, labels, title=None):
        return self._heatmap.create_categorical_heatmap(df, labels, title)

    def create_classification_quality_plot(
        self,
        posterior_probs,
        labels,
        title=None,
    ):
        return self._quality.create_classification_quality_plot(posterior_probs, labels, title)

    def create_consensus_matrix_plot(
        self,
        consensus_matrix,
        labels,
        title=None,
    ):
        return self._quality.create_consensus_matrix_plot(consensus_matrix, labels, title)

    def create_kaplan_meier_plot(
        self,
        survival_result,
        target_name,
        title=None,
    ):
        return self._survival.create_kaplan_meier_plot(survival_result, target_name, title)

    def create_nelson_aalen_plot(
        self,
        survival_result,
        target_name,
        title=None,
    ):
        return self._survival.create_nelson_aalen_plot(survival_result, target_name, title)

    def create_odds_ratio_forest_plot(
        self,
        outcome_results,
        title=None,
        reference_label=None,
        reference_phenotype=0,
    ):
        return self._outcome.create_odds_ratio_forest_plot(
            outcome_results, title, reference_label, reference_phenotype
        )

    def create_pathway_frequency_plot(
        self,
        pathway_results,
        top_n=15,
        title=None,
    ):
        return self._multistate.create_pathway_frequency_plot(pathway_results, top_n, title)

    def create_transition_hazard_forest_plot(
        self,
        transition_results,
        title=None,
        reference_phenotype=0,
    ):
        return self._multistate.create_transition_hazard_forest_plot(
            transition_results, title, reference_phenotype
        )

    def create_state_occupation_uncertainty_plot(
        self,
        mc_results,
        title=None,
    ):
        return self._multistate.create_state_occupation_uncertainty_plot(mc_results, title)

    def create_state_diagram(self, transition_results, title=None):
        return self._multistate.create_state_diagram(transition_results, title)

    def create_all_plots(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        selection_results: Optional[Dict] = None,
        stability_results: Optional[Dict] = None,
        outcome_results: Optional[Dict] = None,
        survival_results: Optional[Dict] = None,
        multistate_results: Optional[Dict] = None,
        posterior_probs: Optional[np.ndarray] = None,
        posterior_probs_test: Optional[np.ndarray] = None,
        labels_test: Optional[np.ndarray] = None,
        reference_phenotype: int = 0,
    ) -> Dict[str, go.Figure]:
        """
        Create all visualizations.

        Parameters
        ----------
        df : pd.DataFrame
            Data used for clustering
        labels : np.ndarray
            Cluster assignments (full dataset)
        selection_results : Dict, optional
            Model selection results
        stability_results : Dict, optional
            Stability analysis results (contains consensus_matrix)
        outcome_results : Dict, optional
            Outcome association results
        survival_results : Dict, optional
            Survival analysis results
        multistate_results : Dict, optional
            Multistate analysis results (pathway_results, transition_results)
        posterior_probs : np.ndarray, optional
            N x K matrix of posterior probabilities from the model (full dataset)
        posterior_probs_test : np.ndarray, optional
            N_test x K matrix of posterior probabilities for test set only
        labels_test : np.ndarray, optional
            Cluster assignments for test set only
        reference_phenotype : int, optional
            Index of the reference phenotype (default 0)

        Returns
        -------
        Dict[str, go.Figure]
            Dictionary of plot name to Plotly figure
        """
        self.logger.info("GENERATING VISUALIZATIONS")

        plots: Dict[str, go.Figure] = {}

        # 1. Cluster distribution (always created)
        dist_plot = self.create_cluster_distribution(labels)
        if dist_plot:
            plots["cluster_distribution"] = dist_plot

        # 2. Classification quality (entropy-based, appropriate for LCA/LPA)
        if posterior_probs_test is not None and labels_test is not None:
            quality_plot = self.create_classification_quality_plot(
                posterior_probs_test,
                labels_test,
                title="Classification Quality (Test Set)",
            )
            if quality_plot:
                plots["classification_quality"] = quality_plot
        elif posterior_probs is not None:
            quality_plot = self.create_classification_quality_plot(posterior_probs, labels)
            if quality_plot:
                plots["classification_quality"] = quality_plot

        # 3. Consensus matrix (if stability analysis was run)
        consensus_matrix = None
        if stability_results and "consensus_matrix" in stability_results:
            consensus_matrix = stability_results["consensus_matrix"]
            consensus_plot = self.create_consensus_matrix_plot(consensus_matrix, labels)
            if consensus_plot:
                plots["consensus_matrix"] = consensus_plot

        # 4. Enhanced heatmap for continuous variables
        if self.config.continuous_columns:
            heatmap = self.create_heatmap(df, labels, consensus_matrix=consensus_matrix)
            if heatmap:
                plots["heatmap_continuous"] = heatmap

        # 5. Categorical flow plots (heatmap and/or Sankey diagrams)
        if self.config.categorical_columns:
            categorical_flow_plots = self.create_categorical_flow_plots(df, labels)
            plots.update(categorical_flow_plots)

        # 6. Model selection plot
        if selection_results and "all_results" in selection_results:
            model_plot = self.create_model_selection_plot(selection_results)
            if model_plot:
                plots["model_selection"] = model_plot

        # 7. Odds ratio forest plot
        if outcome_results:
            forest_plot = self.create_odds_ratio_forest_plot(
                outcome_results,
                reference_label=(f"Phenotype {reference_phenotype} (Reference)"),
                reference_phenotype=reference_phenotype,
            )
            if forest_plot:
                plots["forest_plot_outcomes"] = forest_plot

        # 8. Kaplan-Meier and Nelson-Aalen survival curves
        if survival_results:
            for target_name, target_results in survival_results.items():
                if target_name.endswith("_weighted"):
                    continue
                km_plot = self.create_kaplan_meier_plot(
                    survival_result=target_results, target_name=target_name
                )
                if km_plot:
                    plots[f"kaplan_meier_{target_name}"] = km_plot

                na_plot = self.create_nelson_aalen_plot(
                    survival_result=target_results, target_name=target_name
                )
                if na_plot:
                    plots[f"nelson_aalen_{target_name}"] = na_plot

        # 9. Multistate analysis plots
        if multistate_results:
            pathway_results = multistate_results.get("pathway_results", [])
            if pathway_results:
                pathway_plot = self.create_pathway_frequency_plot(pathway_results)
                if pathway_plot:
                    plots["multistate_pathways"] = pathway_plot

            transition_results = multistate_results.get("transition_results", {})
            if transition_results:
                trans_forest = self.create_transition_hazard_forest_plot(
                    transition_results,
                    reference_phenotype=reference_phenotype,
                )
                if trans_forest:
                    plots["multistate_transition_hazards"] = trans_forest

            state_occ = multistate_results.get("state_occupation_probabilities")
            if state_occ:
                occ_plot = self.create_state_occupation_uncertainty_plot(state_occ)
                if occ_plot:
                    plots["multistate_state_occupation_uncertainty"] = occ_plot

            if transition_results:
                state_diagram = self.create_state_diagram(transition_results)
                if state_diagram:
                    plots["multistate_state_diagram"] = state_diagram

        self.logger.info(f"Created {len(plots)} visualization(s)")

        return plots
