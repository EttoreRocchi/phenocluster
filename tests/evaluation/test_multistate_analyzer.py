"""Tests for multistate analysis orchestrator and helpers."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from phenocluster.config import PhenoClusterConfig
from phenocluster.evaluation.multistate.analyzer import MultistateAnalyzer
from phenocluster.evaluation.multistate.pathways import analyze_pathway_frequencies
from phenocluster.evaluation.multistate.types import MultistateResults


def _make_config(n_clusters=3):
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {"continuous_columns": ["x1"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": n_clusters},
            "outcome": {"enabled": False},
            "multistate": {
                "enabled": True,
                "states": [
                    {"id": 0, "name": "initial", "state_type": "initial"},
                    {
                        "id": 1,
                        "name": "event",
                        "state_type": "transient",
                        "event_column": "event",
                        "time_column": "time_event",
                    },
                    {
                        "id": 2,
                        "name": "death",
                        "state_type": "absorbing",
                        "event_column": "death",
                        "time_column": "time_death",
                    },
                ],
                "transitions": [
                    {"name": "to_event", "from_state": 0, "to_state": 1},
                    {"name": "to_death", "from_state": 0, "to_state": 2},
                ],
            },
            "inference": {"enabled": True},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


class TestMultistateAnalyzerInit:
    def test_init(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        assert analyzer.n_clusters == 3
        assert analyzer.reference_phenotype == 0

    def test_init_custom_reference(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3, reference_phenotype=1)
        assert analyzer.reference_phenotype == 1


class TestMultistateAnalyzerCheckColumns:
    def test_missing_columns_detected(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        df = pd.DataFrame({"x1": [1, 2, 3]})  # Missing event/time columns
        missing = analyzer._check_columns(df)
        assert len(missing) > 0

    def test_all_columns_present(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        df = pd.DataFrame(
            {
                "event": [0, 1, 0],
                "time_event": [5.0, 3.0, 10.0],
                "death": [0, 0, 1],
                "time_death": [10.0, 10.0, 8.0],
            }
        )
        missing = analyzer._check_columns(df)
        assert len(missing) == 0


class TestMultistateAnalyzerResultsToDict:
    def test_converts_to_dict(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        results = MultistateResults(
            model_summary={"n_patients": 100},
            transition_results={},
            pathway_results=[],
        )
        d = analyzer.results_to_dict(results)
        assert isinstance(d, dict)
        assert "model_summary" in d
        assert d["model_summary"]["n_patients"] == 100


class TestMultistateAnalyzerWarnExtrapolation:
    def test_no_warning_within_range(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        # Create mock trajectories with time_at_each_state
        traj = MagicMock()
        traj.time_at_each_state = [0, 5.0, 10.0]
        # Times within range — should not warn
        analyzer._warn_extrapolation([traj], [5.0, 10.0])

    def test_warning_beyond_range(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        traj = MagicMock()
        traj.time_at_each_state = [0, 5.0, 10.0]
        # Times beyond range — should warn
        analyzer._warn_extrapolation([traj], [5.0, 20.0])

    def test_empty_trajectories(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        # Should not raise
        analyzer._warn_extrapolation([], [5.0, 10.0])


class TestRunFullAnalysisMissingColumns:
    def test_missing_columns_returns_error(self):
        cfg = _make_config()
        analyzer = MultistateAnalyzer(cfg, n_clusters=3)
        df = pd.DataFrame({"x1": range(20)})
        labels = np.array([i % 3 for i in range(20)])
        result = analyzer.run_full_analysis(df, labels)
        assert "error" in result.model_summary


class TestAnalyzePathwayFrequencies:
    def test_basic_with_trajectories(self):
        """Test pathway frequency analysis with pre-built trajectories."""
        from phenocluster.config import MultistateState
        from phenocluster.evaluation.multistate.types import PatientTrajectory

        state_by_id = {
            0: MultistateState(id=0, name="initial", state_type="initial"),
            1: MultistateState(
                id=1, name="event", state_type="transient", event_column="evt", time_column="t_evt"
            ),
            2: MultistateState(
                id=2, name="death", state_type="absorbing", event_column="dth", time_column="t_dth"
            ),
        }

        n = 12
        labels = np.array([i % 3 for i in range(n)])
        data = pd.DataFrame({"x": range(n)})

        # Build mock trajectories
        trajs = []
        for i in range(n):
            t = MagicMock(spec=PatientTrajectory)
            t.sample_id = i
            t.states = [0, 1] if i % 2 == 0 else [0, 2]
            trajs.append(t)

        import logging

        result = analyze_pathway_frequencies(
            data,
            labels,
            3,
            state_by_id,
            trajectories=trajs,
            logger=logging.getLogger("test"),
        )
        assert isinstance(result, list)
        assert len(result) >= 1
        # Should have pathway attribute
        assert hasattr(result[0], "pathway")


class TestMCResultsToDict:
    def test_none_input(self):
        result = MultistateAnalyzer._mc_results_to_dict(None)
        assert result is None

    def test_with_mc_results(self):
        from phenocluster.evaluation.multistate.types import MonteCarloResults

        mc = MonteCarloResults(
            time_points=[5.0, 10.0],
            state_probabilities={0: {0: [0.8, 0.6], 1: [0.2, 0.4]}},
            n_simulations=100,
            simulation_summary={"method": "test"},
        )
        result = MultistateAnalyzer._mc_results_to_dict(mc)
        assert isinstance(result, dict)
        assert "time_points" in result
        assert "by_phenotype" in result
