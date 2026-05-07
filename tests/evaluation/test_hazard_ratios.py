"""Tests for transition-specific Cox PH hazard ratio extraction."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import InferenceConfig, MultistateTransition
from phenocluster.evaluation.multistate.hazard_ratios import (
    _apply_transition_fdr,
    _collect_confounder_data,
    _collect_transition_data,
    _fit_cox_for_transition,
    _nan_effect,
    extract_hazard_ratios,
)
from phenocluster.evaluation.multistate.types import (
    MIN_TRANSITION_TIME,
    PatientTrajectory,
    TransitionResult,
)


def _logger():
    log = MagicMock()
    log.info = MagicMock()
    log.warning = MagicMock()
    return log


def _trajectory(states, times, phenotype, n_clusters=2, reference=0, age=50.0):
    """Build a single PatientTrajectory with phenotype-encoded covariates."""
    cov = {}
    for p in range(n_clusters):
        if p == reference:
            continue
        cov[f"phenotype_{p}"] = 1.0 if phenotype == p else 0.0
    cov["age"] = age
    return PatientTrajectory(
        states=states,
        time_at_each_state=times,
        covariates=pd.Series(cov),
    )


def _synthetic_cohort(n_per_pheno=30):
    """Build a synthetic cohort where phenotype 1 has higher hazard."""
    rng = np.random.RandomState(0)
    trajs = []
    for _ in range(n_per_pheno):
        if rng.rand() < 0.7:
            trajs.append(_trajectory([0, 1], [rng.uniform(1, 5)], phenotype=0))
        else:
            trajs.append(_trajectory([0, 1], [rng.uniform(1, 5)], phenotype=0))
    for _ in range(n_per_pheno):
        if rng.rand() < 0.5:
            trajs.append(_trajectory([0, 1], [rng.uniform(0.5, 3)], phenotype=1))
        else:
            trajs.append(_trajectory([0, 2], [rng.uniform(0.5, 3)], phenotype=1))
    return trajs


class TestExtractHazardRatios:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::lifelines.exceptions.ConvergenceWarning")
    def test_returns_dict_per_transition(self):
        trajs = _synthetic_cohort()
        transitions = [
            MultistateTransition(name="t01", from_state=0, to_state=1),
            MultistateTransition(name="t02", from_state=0, to_state=2),
        ]
        out = extract_hazard_ratios(
            transitions,
            trajs,
            n_clusters=2,
            reference_phenotype=0,
            inference_cfg=InferenceConfig(fdr_correction=False),
            max_hr=20.0,
            logger=_logger(),
            min_events_per_transition=3,
        )
        assert "t01" in out
        assert all(isinstance(v, TransitionResult) for v in out.values())

    def test_fdr_adds_q_value(self):
        trajs = _synthetic_cohort()
        transitions = [MultistateTransition(name="t01", from_state=0, to_state=1)]
        out = extract_hazard_ratios(
            transitions,
            trajs,
            n_clusters=2,
            reference_phenotype=0,
            inference_cfg=InferenceConfig(fdr_correction=True),
            max_hr=20.0,
            logger=_logger(),
            min_events_per_transition=3,
        )
        result = out["t01"]
        non_ref = [k for k in result.phenotype_effects if k != 0]
        assert any("q_value" in result.phenotype_effects[k] for k in non_ref)


class TestCollectTransitionData:
    def test_only_matching_from_state_collected(self):
        trajs = [
            _trajectory([0, 1, 2], [2.0, 4.0], phenotype=0),
            _trajectory([0, 2], [3.0], phenotype=1),
        ]
        trans = MultistateTransition(name="evt", from_state=1, to_state=2)
        times, events, phens = _collect_transition_data(
            trans, trajs, n_clusters=2, reference_phenotype=0
        )
        assert len(times) == 1
        assert events[0] == 1

    def test_event_zero_when_to_state_mismatch(self):
        trajs = [
            _trajectory([0, 1], [3.0], phenotype=0),
        ]
        trans = MultistateTransition(name="evt", from_state=0, to_state=2)
        _, events, _ = _collect_transition_data(trans, trajs, n_clusters=2, reference_phenotype=0)
        assert events[0] == 0

    def test_phenotype_recovered_from_covariates(self):
        trajs = [
            _trajectory([0, 1], [3.0], phenotype=1, n_clusters=3, reference=0),
        ]
        trans = MultistateTransition(name="evt", from_state=0, to_state=1)
        _, _, phens = _collect_transition_data(trans, trajs, n_clusters=3, reference_phenotype=0)
        assert phens[0] == 1

    def test_min_transition_time_floor_enforced(self):
        trajs = [_trajectory([0, 1], [0.0], phenotype=0)]
        trans = MultistateTransition(name="evt", from_state=0, to_state=1)
        times, _, _ = _collect_transition_data(trans, trajs, n_clusters=2, reference_phenotype=0)
        assert times[0] >= MIN_TRANSITION_TIME


class TestCollectConfounderData:
    def test_aligned_lengths(self):
        trajs = [
            _trajectory([0, 1], [2.0], phenotype=0, age=40.0),
            _trajectory([0, 1, 2], [2.0, 3.0], phenotype=1, age=70.0),
        ]
        trans = MultistateTransition(name="evt", from_state=0, to_state=1)
        out = _collect_confounder_data(trans, trajs, ["age"])
        assert "age" in out
        assert len(out["age"]) == 2


class TestFitCoxForTransition:
    def test_failure_populates_nan_effects(self):
        log = _logger()
        phenotype_effects = {0: {"HR": 1.0}}
        _fit_cox_for_transition(
            times_arr=np.array([1.0, 2.0]),
            events_arr=np.array([1, 1]),
            phen_arr=np.array([0, 0]),
            non_ref_ids=[1],
            reference_phenotype=0,
            inference_cfg=InferenceConfig(),
            max_hr=10.0,
            phenotype_effects=phenotype_effects,
            logger=log,
            trans_name="degenerate",
        )
        assert np.isnan(phenotype_effects[1]["HR"])

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::lifelines.exceptions.ConvergenceWarning")
    def test_high_hr_marked_unreliable(self):
        rng = np.random.RandomState(1)
        n = 80
        log = _logger()
        phen = np.array([0] * n + [1] * n)
        times = np.concatenate([rng.uniform(5, 10, n), rng.uniform(0.1, 0.5, n)])
        events = np.ones(2 * n, dtype=int)
        phenotype_effects = {0: {"HR": 1.0}}
        _fit_cox_for_transition(
            times_arr=times,
            events_arr=events,
            phen_arr=phen,
            non_ref_ids=[1],
            reference_phenotype=0,
            inference_cfg=InferenceConfig(fdr_correction=False),
            max_hr=2.0,
            phenotype_effects=phenotype_effects,
            logger=log,
            trans_name="strong",
        )
        assert phenotype_effects[1].get("unreliable") is True


class TestNanEffect:
    def test_keys(self):
        eff = _nan_effect()
        for key in ("HR", "CI_lower", "CI_upper", "p_value", "p_value_method", "unreliable"):
            assert key in eff
        assert np.isnan(eff["HR"])
        assert eff["unreliable"] is True


class TestApplyTransitionFdr:
    def test_q_values_added(self):
        results = {
            "t1": TransitionResult(
                transition_name="t1",
                from_state=0,
                to_state=1,
                n_events=5,
                n_at_risk=20,
                phenotype_effects={
                    0: {"p_value": None},
                    1: {"p_value": 0.01},
                    2: {"p_value": 0.04},
                },
                covariate_effects={},
            ),
        }
        _apply_transition_fdr(results)
        assert "q_value" in results["t1"].phenotype_effects[1]
        assert "q_value" in results["t1"].phenotype_effects[2]
        assert "q_value" not in results["t1"].phenotype_effects[0]

    def test_skips_nan_p_values(self):
        results = {
            "t1": TransitionResult(
                transition_name="t1",
                from_state=0,
                to_state=1,
                n_events=5,
                n_at_risk=20,
                phenotype_effects={1: {"p_value": float("nan")}},
                covariate_effects={},
            ),
        }
        _apply_transition_fdr(results)
        assert "q_value" not in results["t1"].phenotype_effects[1]


class TestSkipsLowEventTransition:
    def test_returns_no_entry(self):
        trajs = [_trajectory([0, 1], [3.0], phenotype=0)]
        transitions = [MultistateTransition(name="t01", from_state=0, to_state=1)]
        out = extract_hazard_ratios(
            transitions,
            trajs,
            n_clusters=2,
            reference_phenotype=0,
            inference_cfg=InferenceConfig(fdr_correction=False),
            max_hr=10.0,
            logger=_logger(),
            min_events_per_transition=10,
        )
        assert "t01" not in out

    def test_returns_no_entry_when_no_data(self):
        transitions = [MultistateTransition(name="ghost", from_state=9, to_state=10)]
        out = extract_hazard_ratios(
            transitions,
            [],
            n_clusters=2,
            reference_phenotype=0,
            inference_cfg=InferenceConfig(fdr_correction=False),
            max_hr=10.0,
            logger=_logger(),
            min_events_per_transition=1,
        )
        assert out == {}
