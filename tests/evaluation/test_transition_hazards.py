"""Tests for the transition_hazards helpers and TransitionHazardFitter."""

import logging

import numpy as np
import pandas as pd
import pytest

from phenocluster.evaluation.multistate.transition_hazards import (
    TransitionHazardFitter,
    _prepare_transition_data,
    _stepfunc,
)
from phenocluster.evaluation.multistate.types import PatientTrajectory


def _logger():
    return logging.getLogger("test_transition_hazards")


def _make_trajectory(states, times, sample_id, age=50.0):
    return PatientTrajectory(
        states=states,
        time_at_each_state=times,
        covariates=pd.Series({"age": age}),
        sample_id=sample_id,
    )


class TestStepFunc:
    def test_right_continuous(self):
        x = np.array([0.0, 5.0, 10.0])
        y = np.array([1.0, 0.5, 0.2])
        f = _stepfunc(x, y)
        assert f(7.0) == 0.5
        assert f(0.0) == 1.0
        assert f(15.0) == 0.2

    def test_empty_y(self):
        f = _stepfunc(np.array([0.0]), np.array([0.0]))
        assert f(1.0) == 0.0


class TestPrepareTransitionData:
    def test_returns_rows_per_transition(self):
        trajs = [
            _make_trajectory([0, 1, 2], [3.0, 5.0], sample_id=0),
            _make_trajectory([0, 2], [4.0], sample_id=1),
        ]
        df = _prepare_transition_data(trajs, ["age"], terminal_states=[2])
        assert "origin_state" in df.columns
        assert "target_state" in df.columns
        assert (df["origin_state"] == 0).any()

    def test_handles_empty_trajectories(self):
        df = _prepare_transition_data([], ["age"], terminal_states=[2])
        assert df.empty


class TestTransitionHazardFitter:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::lifelines.exceptions.ConvergenceWarning")
    def test_fit_populates_models(self):
        rng = np.random.RandomState(0)
        trajs = []
        for i in range(60):
            target = 1 if rng.rand() < 0.6 else 2
            trajs.append(
                _make_trajectory(
                    [0, target], [rng.uniform(1, 5)], sample_id=i, age=rng.normal(60, 10)
                )
            )
        fitter = TransitionHazardFitter(
            terminal_states=[2],
            covariate_names=["age"],
            logger=_logger(),
        )
        fitter.fit(trajs)
        assert 0 in fitter.state_models
        assert isinstance(fitter.failure_types[0], list)

    def test_fit_no_data_warns(self):
        fitter = TransitionHazardFitter(
            terminal_states=[2],
            covariate_names=["age"],
            logger=_logger(),
        )
        fitter.fit([])
        assert fitter.state_models == {}
