"""Smoke tests covering uncovered branches in the visualization package."""

import numpy as np
import plotly.graph_objects as go

from phenocluster.config import (
    MultistateConfig,
    MultistateState,
    MultistateTransition,
    PhenoClusterConfig,
)
from phenocluster.visualization._cluster_distribution import ClusterDistributionVisualizer
from phenocluster.visualization._cluster_heatmap import ClusterHeatmapVisualizer
from phenocluster.visualization._cluster_quality import ClusterQualityVisualizer
from phenocluster.visualization._multistate import MultistateVisualizer
from phenocluster.visualization._outcome import OutcomeVisualizer
from phenocluster.visualization._survival import SurvivalVisualizer
from phenocluster.visualization.plots import Visualizer


def _config_with_multistate():
    """PhenoClusterConfig with a small multistate model."""
    return PhenoClusterConfig(
        continuous_columns=["x1", "x2"],
        categorical_columns=["cat1"],
        multistate=MultistateConfig(
            enabled=True,
            states=[
                MultistateState(id=0, name="initial", state_type="initial"),
                MultistateState(
                    id=1,
                    name="event",
                    state_type="transient",
                    event_column="event",
                    time_column="time",
                ),
                MultistateState(id=2, name="death", state_type="absorbing"),
            ],
            transitions=[
                MultistateTransition(name="t01", from_state=0, to_state=1),
                MultistateTransition(name="t12", from_state=1, to_state=2),
            ],
        ),
    )


def _survival_payload():
    return {
        "survival_data": {
            0: {
                "timeline": np.array([1.0, 2.0, 5.0, 10.0]),
                "survival_function": np.array([1.0, 0.9, 0.7, 0.5]),
                "confidence_interval_lower": np.array([1.0, 0.85, 0.6, 0.4]),
                "confidence_interval_upper": np.array([1.0, 0.95, 0.8, 0.6]),
                "n_patients": 20,
                "n_events": 8,
            },
            1: {
                "timeline": np.array([1.0, 2.0, 5.0, 10.0]),
                "survival_function": np.array([1.0, 0.85, 0.6, 0.3]),
                "confidence_interval_lower": np.array([1.0, 0.8, 0.5, 0.2]),
                "confidence_interval_upper": np.array([1.0, 0.9, 0.7, 0.4]),
                "n_patients": 20,
                "n_events": 12,
            },
        },
        "median_survival": {0: 10.0, 1: 5.0},
        "comparison": {"log_rank_p": 0.03},
        "nelson_aalen_data": {
            0: {
                "timeline": np.array([1.0, 5.0]),
                "cumulative_hazard": np.array([0.05, 0.3]),
                "confidence_interval_lower": np.array([0.0, 0.1]),
                "confidence_interval_upper": np.array([0.1, 0.5]),
                "n_patients": 20,
                "n_events": 8,
            },
            1: {
                "timeline": np.array([1.0, 5.0]),
                "cumulative_hazard": np.array([0.1, 0.5]),
                "confidence_interval_lower": np.array([0.05, 0.3]),
                "confidence_interval_upper": np.array([0.2, 0.7]),
                "n_patients": 20,
                "n_events": 12,
            },
        },
    }


def _transition_results():
    return {
        "t01": {
            "transition_name": "t01",
            "from_state": 0,
            "to_state": 1,
            "n_events": 30,
            "n_at_risk": 100,
            "phenotype_effects": {
                "0": {"HR": 1.0},
                "1": {
                    "HR": 2.0,
                    "CI_lower": 1.1,
                    "CI_upper": 3.6,
                    "p_value": 0.02,
                    "unreliable": False,
                },
                "2": {
                    "HR": 0.5,
                    "CI_lower": 0.2,
                    "CI_upper": 1.1,
                    "p_value": 0.08,
                    "unreliable": False,
                },
            },
        },
        "t12": {
            "transition_name": "t12",
            "from_state": 1,
            "to_state": 2,
            "n_events": 20,
            "n_at_risk": 60,
            "phenotype_effects": {
                "0": {"HR": 1.0},
                "1": {
                    "HR": 1.4,
                    "CI_lower": 0.9,
                    "CI_upper": 2.2,
                    "p_value": 0.10,
                    "unreliable": False,
                },
                "2": {
                    "HR": 50.0,
                    "CI_lower": 0.0,
                    "CI_upper": 1000.0,
                    "p_value": 0.99,
                    "unreliable": True,
                },
            },
        },
    }


def _pathway_results():
    return [
        {
            "pathway": (0, 1, 2),
            "state_names": ["initial", "event", "death"],
            "counts_by_phenotype": {"0": 5, "1": 8, "2": 12},
            "total_count": 25,
        },
        {
            "pathway": (0, 2),
            "state_names": ["initial", "death"],
            "counts_by_phenotype": {"0": 3, "1": 4, "2": 1},
            "total_count": 8,
        },
    ]


def _outcome_results():
    return {
        "full_cohort": {
            "outcome1": {
                0: {"OR": 1.0, "CI_lower": 1.0, "CI_upper": 1.0, "p_value": None},
                1: {"OR": 2.4, "CI_lower": 1.2, "CI_upper": 4.5, "p_value": 0.01},
                2: {"OR": 0.6, "CI_lower": 0.3, "CI_upper": 1.2, "p_value": 0.20},
            },
            "outcome2": {
                0: {"OR": 1.0, "CI_lower": 1.0, "CI_upper": 1.0, "p_value": None},
                1: {"OR": 1.8, "CI_lower": 0.9, "CI_upper": 3.5, "p_value": 0.07},
                2: {"OR": 1.1, "CI_lower": 0.5, "CI_upper": 2.5, "p_value": 0.80},
            },
        }
    }


def _state_occupation_payload():
    return {
        "time_points": [5.0, 10.0, 15.0],
        "by_phenotype": {
            "0": {
                "0": [0.8, 0.6, 0.4],
                "1": [0.1, 0.2, 0.3],
                "2": [0.1, 0.2, 0.3],
            },
            "1": {
                "0": [0.7, 0.5, 0.3],
                "1": [0.2, 0.3, 0.4],
                "2": [0.1, 0.2, 0.3],
            },
        },
        "by_phenotype_lower": {
            "0": {
                "0": [0.7, 0.5, 0.3],
                "1": [0.05, 0.1, 0.2],
                "2": [0.05, 0.1, 0.2],
            },
            "1": {
                "0": [0.6, 0.4, 0.2],
                "1": [0.1, 0.2, 0.3],
                "2": [0.05, 0.1, 0.2],
            },
        },
        "by_phenotype_upper": {
            "0": {
                "0": [0.9, 0.7, 0.5],
                "1": [0.2, 0.3, 0.4],
                "2": [0.2, 0.3, 0.4],
            },
            "1": {
                "0": [0.8, 0.6, 0.4],
                "1": [0.3, 0.4, 0.5],
                "2": [0.2, 0.3, 0.4],
            },
        },
        "n_simulations": 500,
    }


class TestVisualizerCreateAllPlots:
    def test_full_pipeline_plots_returned(self, sample_dataframe, sample_labels):
        cfg = _config_with_multistate()
        cfg.continuous_columns = ["x1", "x2", "x3"]
        cfg.categorical_columns = ["cat1"]
        vis = Visualizer(cfg, n_clusters=3)
        rng = np.random.RandomState(0)
        posterior = rng.dirichlet([3, 3, 3], size=len(sample_labels))
        plots = vis.create_all_plots(
            sample_dataframe,
            sample_labels,
            selection_results={
                "all_results": [
                    {"n_clusters": 2, "BIC": -100.0},
                    {"n_clusters": 3, "BIC": -200.0},
                ],
                "best_n_clusters": 3,
                "criterion_used": "BIC",
            },
            stability_results={"consensus_matrix": np.eye(len(sample_labels))},
            outcome_results=_outcome_results(),
            survival_results={"mortality": _survival_payload()},
            multistate_results={
                "transition_results": _transition_results(),
                "pathway_results": _pathway_results(),
                "state_occupation_probabilities": _state_occupation_payload(),
            },
            posterior_probs=posterior,
            posterior_probs_test=posterior,
            labels_test=sample_labels,
            reference_phenotype=0,
        )
        assert isinstance(plots, dict)
        assert "cluster_distribution" in plots
        assert "classification_quality" in plots
        assert "consensus_matrix" in plots

    def test_facade_create_kaplan_meier(self):
        cfg = _config_with_multistate()
        vis = Visualizer(cfg, n_clusters=2)
        fig = vis.create_kaplan_meier_plot(_survival_payload(), "mortality")
        assert fig is None or isinstance(fig, go.Figure)

    def test_facade_create_nelson_aalen(self):
        cfg = _config_with_multistate()
        vis = Visualizer(cfg, n_clusters=2)
        fig = vis.create_nelson_aalen_plot(_survival_payload(), "mortality")
        assert fig is None or isinstance(fig, go.Figure)

    def test_facade_create_categorical_heatmap(self, sample_dataframe, sample_labels):
        cfg = _config_with_multistate()
        cfg.continuous_columns = ["x1", "x2", "x3"]
        cfg.categorical_columns = ["cat1"]
        vis = Visualizer(cfg, n_clusters=3)
        fig = vis.create_categorical_heatmap(sample_dataframe[["cat1"]], sample_labels)
        assert fig is None or isinstance(fig, go.Figure)


class TestMultistateVisualizerExtras:
    def test_state_diagram_with_config(self):
        cfg = _config_with_multistate()
        vis = MultistateVisualizer(cfg, n_clusters=3)
        fig = vis.create_state_diagram(_transition_results())
        assert isinstance(fig, go.Figure)

    def test_state_diagram_no_config_returns_none(self):
        cfg = _config_with_multistate()
        cfg.multistate.states = []
        vis = MultistateVisualizer(cfg, n_clusters=3)
        fig = vis.create_state_diagram(_transition_results())
        assert fig is None

    def test_state_occupation_with_uncertainty(self):
        cfg = _config_with_multistate()
        vis = MultistateVisualizer(cfg, n_clusters=2)
        fig = vis.create_state_occupation_uncertainty_plot(_state_occupation_payload())
        assert fig is None or isinstance(fig, go.Figure)

    def test_pathway_frequency_top_n(self):
        cfg = _config_with_multistate()
        vis = MultistateVisualizer(cfg, n_clusters=3)
        fig = vis.create_pathway_frequency_plot(_pathway_results(), top_n=1)
        assert fig is None or isinstance(fig, go.Figure)

    def test_transition_forest_skips_reference(self):
        cfg = _config_with_multistate()
        vis = MultistateVisualizer(cfg, n_clusters=3)
        fig = vis.create_transition_hazard_forest_plot(_transition_results(), reference_phenotype=0)
        assert fig is None or isinstance(fig, go.Figure)

    def test_empty_transition_results(self):
        cfg = _config_with_multistate()
        vis = MultistateVisualizer(cfg, n_clusters=3)
        assert vis.create_transition_hazard_forest_plot({}) is None
        assert vis.create_pathway_frequency_plot([]) is None
        assert vis.create_state_diagram({}) is None


class TestSurvivalVisualizerEdge:
    def test_km_with_full_payload(self):
        cfg = _config_with_multistate()
        vis = SurvivalVisualizer(cfg, n_clusters=2)
        fig = vis.create_kaplan_meier_plot(_survival_payload(), "mortality")
        assert isinstance(fig, go.Figure)

    def test_nelson_aalen_with_full_payload(self):
        cfg = _config_with_multistate()
        vis = SurvivalVisualizer(cfg, n_clusters=2)
        fig = vis.create_nelson_aalen_plot(_survival_payload(), "mortality")
        assert isinstance(fig, go.Figure)

    def test_km_empty_returns_none(self):
        cfg = _config_with_multistate()
        vis = SurvivalVisualizer(cfg, n_clusters=2)
        assert vis.create_kaplan_meier_plot({}) is None


class TestOutcomeVisualizerExtras:
    def test_with_full_cohort_wrapper(self):
        cfg = _config_with_multistate()
        vis = OutcomeVisualizer(cfg, n_clusters=3)
        fig = vis.create_odds_ratio_forest_plot(_outcome_results())
        assert fig is None or isinstance(fig, go.Figure)

    def test_with_train_wrapper(self):
        cfg = _config_with_multistate()
        vis = OutcomeVisualizer(cfg, n_clusters=3)
        fig = vis.create_odds_ratio_forest_plot({"train": _outcome_results()["full_cohort"]})
        assert fig is None or isinstance(fig, go.Figure)

    def test_filters_nan_entries(self):
        cfg = _config_with_multistate()
        vis = OutcomeVisualizer(cfg, n_clusters=2)
        nan_only = {
            "full_cohort": {
                "x": {
                    0: {"OR": 1.0, "CI_lower": 1.0, "CI_upper": 1.0, "p_value": None},
                    1: {
                        "OR": float("nan"),
                        "CI_lower": float("nan"),
                        "CI_upper": float("nan"),
                        "p_value": 0.5,
                    },
                }
            }
        }
        assert vis.create_odds_ratio_forest_plot(nan_only) is None


class TestClusterQualityNearPerfect:
    def test_near_perfect_branch(self, minimal_config):
        vis = ClusterQualityVisualizer(minimal_config, n_clusters=3)
        n = 60
        posterior = np.zeros((n, 3))
        for i in range(n):
            posterior[i, i % 3] = 0.999
            posterior[i, (i + 1) % 3] = 0.0005
            posterior[i, (i + 2) % 3] = 0.0005
        labels = np.array([i % 3 for i in range(n)])
        fig = vis.create_classification_quality_plot(posterior, labels)
        assert isinstance(fig, go.Figure)


class TestClusterDistributionExtras:
    def test_model_selection_with_metadata(self, minimal_config):
        vis = ClusterDistributionVisualizer(minimal_config, n_clusters=3)
        results = {
            "all_results": [
                {"n_clusters": 2, "BIC": -100.0, "AIC": -90.0},
                {"n_clusters": 3, "BIC": -150.0, "AIC": -130.0},
                {"n_clusters": 4, "BIC": -130.0, "AIC": -120.0},
            ],
            "best_n_clusters": 3,
            "criterion_used": "BIC",
        }
        fig = vis.create_model_selection_plot(results)
        assert fig is None or isinstance(fig, go.Figure)


class TestClusterHeatmapExtras:
    def test_heatmap_with_consensus(self, minimal_config, sample_dataframe, sample_labels):
        vis = ClusterHeatmapVisualizer(minimal_config, n_clusters=3)
        n = len(sample_labels)
        consensus = np.eye(n)
        fig = vis.create_heatmap(
            sample_dataframe[["x1", "x2", "x3"]],
            sample_labels,
            consensus_matrix=consensus,
        )
        assert fig is None or isinstance(fig, go.Figure)

    def test_categorical_flow_with_sankey_enabled(
        self, minimal_config, sample_dataframe, sample_labels
    ):
        cfg = minimal_config
        cfg.categorical_flow.show_sankey = True
        vis = ClusterHeatmapVisualizer(cfg, n_clusters=3)
        result = vis.create_categorical_flow_plots(sample_dataframe[["cat1"]], sample_labels)
        assert isinstance(result, dict)
