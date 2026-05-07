"""Tests for the multistate TrajectoryBuilder."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import (
    MultistateConfig,
    MultistateState,
    MultistateTransition,
    PhenoClusterConfig,
)
from phenocluster.evaluation.multistate.trajectory import TrajectoryBuilder
from phenocluster.evaluation.multistate.types import MIN_TRANSITION_TIME


def _full_config(baseline_confounders=None):
    """Build a config with initial -> event -> death/censor states."""
    states = [
        MultistateState(id=0, name="init", state_type="initial"),
        MultistateState(
            id=1,
            name="event",
            state_type="transient",
            event_column="event",
            time_column="event_time",
        ),
        MultistateState(
            id=2,
            name="death",
            state_type="absorbing",
            event_column="death",
            time_column="death_time",
        ),
        MultistateState(id=3, name="censor", state_type="absorbing"),
    ]
    transitions = [
        MultistateTransition(name="init_to_event", from_state=0, to_state=1),
        MultistateTransition(name="init_to_death", from_state=0, to_state=2),
        MultistateTransition(name="event_to_death", from_state=1, to_state=2),
    ]
    cfg = PhenoClusterConfig(
        continuous_columns=["age"],
        multistate=MultistateConfig(
            enabled=True,
            states=states,
            transitions=transitions,
            baseline_confounders=baseline_confounders or [],
            default_followup_time=30.0,
        ),
    )
    return cfg


def _patient_data():
    """Mixed cohort: one patient transitions, one is censored."""
    return pd.DataFrame(
        {
            "age": [50.0, 60.0, np.nan],
            "event": [1, 0, 0],
            "event_time": [5.0, np.nan, np.nan],
            "death": [1, 0, 0],
            "death_time": [10.0, np.nan, np.nan],
        }
    )


class TestTrajectoryBuilderHelpers:
    def test_get_initial_state(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        assert builder.get_initial_state().id == 0

    def test_initial_state_missing_raises(self):
        cfg = _full_config()
        cfg.multistate.states = [
            MultistateState(id=2, name="death", state_type="absorbing"),
        ]
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        with pytest.raises(ValueError):
            builder.get_initial_state()

    def test_get_absorbing_states(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        absorbing = builder.get_absorbing_states()
        assert {s.id for s in absorbing} == {2, 3}

    def test_get_terminal_state_ids(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        assert set(builder.get_terminal_state_ids()) == {2, 3}

    def test_get_censoring_state(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        cens = builder.get_censoring_state()
        assert cens is not None
        assert cens.id == 3

    def test_no_censoring_state_returns_none(self):
        cfg = _full_config()
        cfg.multistate.states = [s for s in cfg.multistate.states if s.id != 3]
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        assert builder.get_censoring_state() is None


class TestPrepareTrajectories:
    def test_returns_trajectories_and_phenotype_indices(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        df = _patient_data()
        labels = np.array([0, 1, 1])
        trajs, idx_map = builder.prepare_trajectories(df, labels)
        assert isinstance(trajs, list)
        assert set(idx_map.keys()) == {0, 1}
        assert idx_map[1] == [1, 2]

    def test_skips_patients_with_no_transitions(self):
        cfg = _full_config()
        cfg.multistate.states = [s for s in cfg.multistate.states if s.id != 3]
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        df = pd.DataFrame(
            {
                "age": [50.0],
                "event": [0],
                "event_time": [np.nan],
                "death": [0],
                "death_time": [np.nan],
            }
        )
        trajs, _ = builder.prepare_trajectories(df, np.array([0]))
        assert trajs == []

    def test_baseline_confounder_argument_overrides_config(self):
        cfg = _full_config(baseline_confounders=["age"])
        builder = TrajectoryBuilder(cfg, n_clusters=2, reference_phenotype=0)
        df = _patient_data()
        labels = np.array([0, 1, 1])
        trajs, _ = builder.prepare_trajectories(df, labels, baseline_confounders=["age"])
        for traj in trajs:
            assert "age" in traj.covariates.index

    def test_logger_messages_when_provided(self):
        cfg = _full_config()
        log = MagicMock()
        builder = TrajectoryBuilder(cfg, n_clusters=2, logger=log)
        df = _patient_data()
        builder.prepare_trajectories(df, np.array([0, 0, 0]))
        log.info.assert_called()


class TestDetermineStateSequence:
    def test_visited_states_ordered(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        row = pd.Series({"event": 1, "event_time": 5.0, "death": 1, "death_time": 10.0})
        states, times = builder._determine_state_sequence(row, 0)
        assert states == [0, 1, 2]
        assert times[0] == pytest.approx(5.0)
        assert times[1] == pytest.approx(5.0)

    def test_disallowed_transition_skipped(self):
        cfg = _full_config()
        cfg.multistate.transitions = [
            MultistateTransition(name="init_to_event", from_state=0, to_state=1),
        ]
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        row = pd.Series({"event": 0, "event_time": np.nan, "death": 1, "death_time": 10.0})
        states, _ = builder._determine_state_sequence(row, 0)
        assert 2 not in states

    def test_min_transition_time_floor(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        row = pd.Series({"event": 1, "event_time": 0.0, "death": 0, "death_time": np.nan})
        _, times = builder._determine_state_sequence(row, 0)
        assert all(t >= MIN_TRANSITION_TIME for t in times)


class TestCensoringHandling:
    def test_censoring_appended(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        row = pd.Series({"event": 0, "event_time": np.nan, "death": 0, "death_time": np.nan})
        states, _ = builder._determine_state_sequence(row, 0)
        assert states[-1] == 3

    def test_default_followup_used(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        row = pd.Series({"event": 0, "event_time": np.nan, "death": 0, "death_time": np.nan})
        _, times = builder._determine_state_sequence(row, 0)
        assert times[-1] == pytest.approx(cfg.multistate.default_followup_time)

    def test_no_censoring_when_absorbed(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        row = pd.Series({"event": 0, "event_time": np.nan, "death": 1, "death_time": 8.0})
        states, _ = builder._determine_state_sequence(row, 0)
        assert states[-1] == 2


class TestBuildCovariateDict:
    def test_phenotype_dummies_set(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=3, reference_phenotype=0)
        cov = builder._build_covariate_dict(
            phenotype=2, row=pd.Series(dtype=float), confounders=[], confounder_means={}
        )
        assert cov["phenotype_2"] == 1.0
        assert cov["phenotype_1"] == 0.0
        assert "phenotype_0" not in cov

    def test_nan_confounder_filled_with_mean(self):
        cfg = _full_config(baseline_confounders=["age"])
        builder = TrajectoryBuilder(cfg, n_clusters=2)
        cov = builder._build_covariate_dict(
            phenotype=0,
            row=pd.Series({"age": np.nan}),
            confounders=["age"],
            confounder_means={"age": 55.0},
        )
        assert cov["age"] == 55.0


class TestAlignTimes:
    def test_terminal_state_truncates_extra_times(self):
        cfg = _full_config()
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        states = [0, 1, 2]
        times = [5.0, 5.0, 99.0]
        builder._align_times(states, times, current_time=10.0)
        assert len(times) <= len(states)

    def test_non_terminal_pads_times(self):
        cfg = _full_config()
        cfg.multistate.states = [
            MultistateState(id=0, name="init", state_type="initial"),
            MultistateState(
                id=1,
                name="event",
                state_type="transient",
                event_column="event",
                time_column="event_time",
            ),
            MultistateState(id=2, name="death", state_type="absorbing"),
        ]
        builder = TrajectoryBuilder(cfg, n_clusters=1)
        states = [0, 1]
        times = []
        builder._align_times(states, times, current_time=0.0)
        assert len(times) == len(states)
