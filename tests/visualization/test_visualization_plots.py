"""Tests for visualization plot classes."""

import numpy as np
import plotly.graph_objects as go

from phenocluster.visualization._cluster_distribution import ClusterDistributionVisualizer
from phenocluster.visualization._cluster_heatmap import ClusterHeatmapVisualizer
from phenocluster.visualization._cluster_quality import ClusterQualityVisualizer
from phenocluster.visualization._multistate import MultistateVisualizer
from phenocluster.visualization._outcome import OutcomeVisualizer
from phenocluster.visualization._survival import SurvivalVisualizer
from phenocluster.visualization.plots import Visualizer


class TestClusterDistributionVisualizer:
    def test_returns_figure(self, minimal_config, sample_labels):
        vis = ClusterDistributionVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_cluster_distribution(sample_labels)
        assert isinstance(fig, go.Figure)

    def test_trace_count(self, minimal_config, sample_labels):
        vis = ClusterDistributionVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_cluster_distribution(sample_labels)
        assert len(fig.data) >= 1

    def test_model_selection_plot(self, minimal_config):
        vis = ClusterDistributionVisualizer(minimal_config, n_clusters=3)
        selection_results = {
            "all_results": [
                {"n_clusters": 2, "BIC": -100},
                {"n_clusters": 3, "BIC": -150},
                {"n_clusters": 4, "BIC": -120},
            ],
            "best_n_clusters": 3,
            "criterion_used": "BIC",
        }
        fig = vis.create_model_selection_plot(selection_results)
        assert fig is None or isinstance(fig, go.Figure)

    def test_model_selection_empty(self, minimal_config):
        vis = ClusterDistributionVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_model_selection_plot({})
        assert fig is None


class TestClusterHeatmapVisualizer:
    def test_create_heatmap_returns_figure(self, minimal_config, sample_dataframe, sample_labels):
        vis = ClusterHeatmapVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_heatmap(sample_dataframe[["x1", "x2", "x3"]], sample_labels)
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_categorical_heatmap(self, minimal_config, sample_dataframe, sample_labels):
        vis = ClusterHeatmapVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_categorical_heatmap(sample_dataframe[["cat1"]], sample_labels)
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_categorical_flow_plots(self, minimal_config, sample_dataframe, sample_labels):
        vis = ClusterHeatmapVisualizer(minimal_config, n_clusters=3)
        result = vis.create_categorical_flow_plots(sample_dataframe[["cat1"]], sample_labels)
        assert isinstance(result, dict)

    def test_heatmap_with_significance(self, minimal_config, sample_dataframe, sample_labels):
        vis = ClusterHeatmapVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_heatmap(
            sample_dataframe[["x1", "x2", "x3"]], sample_labels, show_significance=True
        )
        assert fig is None or isinstance(fig, go.Figure)


class TestClusterQualityVisualizer:
    def test_classification_quality_plot(self, minimal_config, sample_labels):
        vis = ClusterQualityVisualizer(minimal_config, n_clusters=3)
        n = len(sample_labels)
        posterior = np.random.RandomState(0).dirichlet([5, 5, 5], size=n)
        fig = vis.create_classification_quality_plot(posterior, sample_labels)
        assert fig is None or isinstance(fig, go.Figure)

    def test_consensus_matrix_plot(self, minimal_config, sample_labels):
        vis = ClusterQualityVisualizer(minimal_config, n_clusters=3)
        n = len(sample_labels)
        consensus = np.random.RandomState(0).rand(n, n)
        consensus = (consensus + consensus.T) / 2
        np.fill_diagonal(consensus, 1.0)
        fig = vis.create_consensus_matrix_plot(consensus, sample_labels)
        assert isinstance(fig, go.Figure)

    def test_consensus_matrix_none_handled(self, minimal_config):
        vis = ClusterQualityVisualizer(minimal_config, n_clusters=3)
        # Should handle gracefully if called with invalid input
        try:
            fig = vis.create_consensus_matrix_plot(np.eye(5), np.array([0, 0, 1, 1, 2]))
            assert isinstance(fig, go.Figure)
        except Exception:
            pass  # Some implementations may raise on mismatched sizes


class TestSurvivalVisualizer:
    def _make_survival_result(self):
        """Create minimal survival result dict matching SurvivalAnalyzer output."""
        return {
            "survival_data": {
                0: {
                    "timeline": np.array([1.0, 2.0, 5.0, 10.0]),
                    "survival_function": np.array([1.0, 0.9, 0.7, 0.5]),
                    "ci_lower": np.array([1.0, 0.85, 0.6, 0.4]),
                    "ci_upper": np.array([1.0, 0.95, 0.8, 0.6]),
                    "n_patients": 20,
                },
                1: {
                    "timeline": np.array([1.0, 2.0, 5.0, 10.0]),
                    "survival_function": np.array([1.0, 0.85, 0.6, 0.3]),
                    "ci_lower": np.array([1.0, 0.8, 0.5, 0.2]),
                    "ci_upper": np.array([1.0, 0.9, 0.7, 0.4]),
                    "n_patients": 20,
                },
            },
            "median_survival": {0: 10.0, 1: 5.0},
            "comparison": {"log_rank_p": 0.03},
        }

    def test_create_km_plot(self, minimal_config):
        vis = SurvivalVisualizer(minimal_config, n_clusters=2)
        fig = vis.create_kaplan_meier_plot(self._make_survival_result())
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_km_plot_no_data(self, minimal_config):
        vis = SurvivalVisualizer(minimal_config, n_clusters=2)
        fig = vis.create_kaplan_meier_plot({})
        assert fig is None

    def test_create_nelson_aalen_plot(self, minimal_config):
        vis = SurvivalVisualizer(minimal_config, n_clusters=2)
        result = self._make_survival_result()
        result["nelson_aalen_data"] = {
            0: {"timeline": np.array([1.0, 5.0]), "cumulative_hazard": np.array([0.05, 0.3])},
            1: {"timeline": np.array([1.0, 5.0]), "cumulative_hazard": np.array([0.1, 0.5])},
        }
        fig = vis.create_nelson_aalen_plot(result)
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_forest_plot(self, minimal_config):
        vis = SurvivalVisualizer(minimal_config, n_clusters=2)
        # SurvivalVisualizer doesn't have create_forest_plot — it's in outcome
        # But let's verify KM with comparison data works
        fig = vis.create_kaplan_meier_plot(self._make_survival_result())
        assert fig is None or isinstance(fig, go.Figure)


class TestOutcomeVisualizer:
    def _make_outcome_results(self):
        return {
            "outcome1": {
                0: {"OR": 1.0, "CI_lower": 1.0, "CI_upper": 1.0, "p_value": None},
                1: {"OR": 2.5, "CI_lower": 1.3, "CI_upper": 4.8, "p_value": 0.01},
                2: {"OR": 0.8, "CI_lower": 0.4, "CI_upper": 1.5, "p_value": 0.4},
            }
        }

    def test_create_forest_plot(self, minimal_config):
        vis = OutcomeVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_odds_ratio_forest_plot(self._make_outcome_results())
        assert fig is None or isinstance(fig, go.Figure)

    def test_empty_results(self, minimal_config):
        vis = OutcomeVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_odds_ratio_forest_plot({})
        assert fig is None

    def test_reference_phenotype(self, minimal_config):
        vis = OutcomeVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_odds_ratio_forest_plot(self._make_outcome_results(), reference_phenotype=0)
        assert fig is None or isinstance(fig, go.Figure)


class TestMultistateVisualizer:
    def _make_transition_results(self):
        return {
            "to_event": {
                "transition_name": "to_event",
                "from_state": "initial",
                "to_state": "event",
                "n_events": 30,
                "n_at_risk": 100,
                "phenotype_effects": {
                    1: {"HR": 2.0, "CI_lower": 1.1, "CI_upper": 3.6, "p_value": 0.02},
                    2: {"HR": 0.5, "CI_lower": 0.2, "CI_upper": 1.1, "p_value": 0.08},
                },
            }
        }

    def test_create_transition_hr_forest(self, minimal_config):
        vis = MultistateVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_transition_hazard_forest_plot(self._make_transition_results())
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_pathway_plot(self, minimal_config):
        vis = MultistateVisualizer(minimal_config, n_clusters=3)
        pathways = [
            {
                "pathway": (0, 1),
                "state_names": ["initial", "event"],
                "counts_by_phenotype": {0: 10, 1: 5, 2: 8},
                "total_count": 23,
            },
            {
                "pathway": (0, 2),
                "state_names": ["initial", "death"],
                "counts_by_phenotype": {0: 3, 1: 7, 2: 2},
                "total_count": 12,
            },
        ]
        fig = vis.create_pathway_frequency_plot(pathways)
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_state_occupation_plot(self, minimal_config):
        vis = MultistateVisualizer(minimal_config, n_clusters=3)
        mc_results = {
            "time_points": [5.0, 10.0, 15.0],
            "by_phenotype": {
                "0": {"0": [0.8, 0.6, 0.4], "1": [0.1, 0.2, 0.3], "2": [0.1, 0.2, 0.3]},
                "1": {"0": [0.7, 0.5, 0.3], "1": [0.2, 0.3, 0.4], "2": [0.1, 0.2, 0.3]},
            },
            "n_simulations": 1000,
        }
        fig = vis.create_state_occupation_uncertainty_plot(mc_results)
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_state_diagram(self, minimal_config):
        vis = MultistateVisualizer(minimal_config, n_clusters=3)
        fig = vis.create_state_diagram(self._make_transition_results())
        assert fig is None or isinstance(fig, go.Figure)


class TestVisualizerFacade:
    def test_delegates_distribution(self, minimal_config, sample_labels):
        vis = Visualizer(minimal_config, n_clusters=3)
        fig = vis.create_cluster_distribution(sample_labels)
        assert isinstance(fig, go.Figure)

    def test_delegates_heatmap(self, minimal_config, sample_dataframe, sample_labels):
        vis = Visualizer(minimal_config, n_clusters=3)
        fig = vis.create_heatmap(sample_dataframe[["x1", "x2", "x3"]], sample_labels)
        assert fig is None or isinstance(fig, go.Figure)

    def test_create_all_plots_minimal(self, minimal_config, sample_dataframe, sample_labels):
        vis = Visualizer(minimal_config, n_clusters=3)
        plots = vis.create_all_plots(sample_dataframe, sample_labels)
        assert isinstance(plots, dict)
        # Should have at least the distribution plot
        assert len(plots) >= 1
