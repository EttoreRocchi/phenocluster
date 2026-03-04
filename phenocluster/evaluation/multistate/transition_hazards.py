"""Transition-specific hazard models using lifelines CoxPHFitter."""

import contextlib
import io
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from scipy.interpolate import interp1d

from .types import MIN_TRANSITION_TIME, RIGHT_CENSORING, PatientTrajectory


def _stepfunc(x: np.ndarray, y: np.ndarray) -> interp1d:
    """Create a right-continuous step function interpolator."""
    return interp1d(
        x,
        y,
        kind="previous",
        bounds_error=False,
        fill_value=(0.0, y[-1] if len(y) > 0 else 0.0),
    )


def _break_ties(
    t: np.ndarray,
    eps_max: float = 0.0001,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Break tied event times by adding small random epsilon."""
    if rng is None:
        rng = np.random.RandomState(42)
    _, inverse, count = np.unique(t, return_inverse=True, return_counts=True)
    tied_idx = np.where(count[inverse] > 1)[0]
    t_new = t.copy().astype(float)
    eps = rng.uniform(0.0, eps_max, size=len(tied_idx))
    np.add.at(t_new, tied_idx, eps)
    return t_new


def _check_ph_assumption(cox, event_df, origin, target, logger):
    """Check proportional hazards assumption (diagnostic only)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ph_test = cox.check_assumptions(event_df, p_value_threshold=0.05)
        if ph_test is not None and len(ph_test) > 0:
            if logger:
                violated_covs = [c for c, _ in ph_test]
                logger.warning(
                    f"PH assumption may be violated for "
                    f"origin={origin}->target={target}: {violated_covs}"
                )
    except Exception:
        pass


def _prepare_transition_data(
    trajectories: List[PatientTrajectory],
    covariate_names: List[str],
    terminal_states: List[int],
) -> pd.DataFrame:
    """Convert patient trajectories to a transition-analysis DataFrame."""
    rows = []
    for traj in trajectories:
        origin_state = traj.states[0]
        covs = dict(zip(covariate_names, traj.covariates.values))
        time_entry_to_origin = 0.0

        for i in range(len(traj.states)):
            if i < len(traj.time_at_each_state):
                time_in_origin = traj.time_at_each_state[i]
            else:
                break

            time_transition = time_entry_to_origin + time_in_origin
            target_state = traj.states[i + 1] if i + 1 < len(traj.states) else RIGHT_CENSORING

            row = {
                "sample_id": traj.sample_id,
                "origin_state": origin_state,
                "target_state": target_state,
                "time_entry_to_origin": time_entry_to_origin,
                "time_transition_to_target": time_transition,
            }
            row.update(covs)
            rows.append(row)

            if target_state == RIGHT_CENSORING or target_state in terminal_states:
                break
            else:
                origin_state = target_state
                time_entry_to_origin = time_transition

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["sample_id"] = df["sample_id"].astype(int)
        df["origin_state"] = df["origin_state"].astype(int)
        df["target_state"] = df["target_state"].astype(int)
    return df


class TransitionHazardFitter:
    """Multistate transition-specific hazard model using lifelines CoxPHFitter.

    For each origin state, fits one CoxPHFitter per possible target state
    (treating transitions to other targets as censoring).
    """

    def __init__(
        self,
        terminal_states: List[int],
        covariate_names: List[str],
        logger=None,
        random_state: int = 42,
    ):
        self.terminal_states = terminal_states
        self.covariate_names = covariate_names
        self.logger = logger
        self.rng = np.random.RandomState(random_state)
        self.state_models: Dict[int, Dict[int, CoxPHFitter]] = {}
        self.failure_types: Dict[int, List[int]] = {}
        self.transition_dataset: Optional[pd.DataFrame] = None

    def fit(self, trajectories: List[PatientTrajectory]) -> None:
        """Fit transition-specific Cox models for each origin state."""
        self.transition_dataset = _prepare_transition_data(
            trajectories,
            self.covariate_names,
            self.terminal_states,
        )
        df = self.transition_dataset

        if len(df) == 0:
            if self.logger:
                self.logger.warning("No transition data to fit")
            return

        for origin in df["origin_state"].unique():
            state_df = df[df["origin_state"] == origin].copy()
            state_df = state_df.drop(columns=["origin_state"]).reset_index(drop=True)

            ftypes = sorted([int(ft) for ft in state_df["target_state"].unique() if ft > 0])
            if not ftypes:
                continue

            self.failure_types[int(origin)] = ftypes
            self.state_models[int(origin)] = {}

            for target in ftypes:
                self._fit_single_transition(state_df, int(origin), target)

    def _fit_single_transition(self, state_df, origin, target):
        """Fit Cox model for a single origin -> target transition."""
        event_df = state_df.copy()
        is_event = event_df["target_state"] == target
        event_df.loc[~is_event, "target_state"] = 0
        event_df.loc[is_event, "target_state"] = 1

        event_df["time_transition_to_target"] = _break_ties(
            event_df["time_transition_to_target"].values
        )
        event_df["time_transition_to_target"] = event_df["time_transition_to_target"].clip(
            lower=1e-6
        )
        event_df["time_entry_to_origin"] = event_df["time_entry_to_origin"].clip(lower=0)

        mask = event_df["time_entry_to_origin"] >= event_df["time_transition_to_target"]
        event_df.loc[mask, "time_entry_to_origin"] = (
            event_df.loc[mask, "time_transition_to_target"] - 1e-6
        ).clip(lower=0)

        n_events = int(is_event.sum())
        if n_events < 2:
            if self.logger:
                self.logger.debug(
                    f"Skipping origin={origin}->target={target}: only {n_events} events"
                )
            return

        try:
            cox = CoxPHFitter()
            cox.fit(
                event_df,
                duration_col="time_transition_to_target",
                event_col="target_state",
                entry_col="time_entry_to_origin",
                cluster_col="sample_id",
                show_progress=False,
            )
            self.state_models[origin][target] = cox

            if self.logger:
                self.logger.debug(
                    f"Fitted Cox model: origin={origin}->target={target}, {n_events} events"
                )

            _check_ph_assumption(cox, event_df, origin, target, self.logger)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Cox model failed for origin={origin}->target={target}: {e}")

    def _get_hazard(self, model: CoxPHFitter, covariates: np.ndarray) -> np.ndarray:
        """Compute hazard at unique event times."""
        coefs = model.params_.values
        partial_hazard = np.exp(np.dot(covariates, coefs))
        baseline_h = model.baseline_hazard_["baseline hazard"].values
        return baseline_h * partial_hazard

    def _get_unique_event_times(self, model: CoxPHFitter) -> np.ndarray:
        return model.baseline_hazard_.index.values

    def _get_cumulative_hazard(
        self, model: CoxPHFitter, t: np.ndarray, covariates: np.ndarray
    ) -> np.ndarray:
        """Compute cumulative hazard at arbitrary times t via step function."""
        coefs = model.params_.values
        partial_hazard = np.exp(np.dot(covariates, coefs))
        baseline_ch = model.baseline_cumulative_hazard_["baseline cumulative hazard"].values
        unique_times = self._get_unique_event_times(model)
        ch_func = _stepfunc(unique_times, baseline_ch)
        return ch_func(t) * partial_hazard

    def _get_survival(self, origin_state: int, t: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """Compute overall survival: exp(-sum of cumulative hazards)."""
        models = self.state_models.get(origin_state, {})
        exponent = np.zeros_like(t, dtype=float)
        for target, model in models.items():
            exponent -= self._get_cumulative_hazard(model, t, covariates)
        return np.exp(exponent)

    def _probability_for_next_state(
        self,
        origin_state: int,
        next_state: int,
        covariates: np.ndarray,
        t_entry: float,
    ) -> float:
        """Compute probability of transitioning to next_state."""
        model = self.state_models[origin_state][next_state]
        unique_times = self._get_unique_event_times(model)
        mask = unique_times > t_entry
        hazard = self._get_hazard(model, covariates)[mask]
        survival = self._get_survival(origin_state, unique_times[mask], covariates)
        return float(np.nansum(hazard * survival))

    def _sample_next_state(
        self,
        current_state: int,
        covariates: np.ndarray,
        t_entry: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> Optional[int]:
        """Sample next state using multinomial distribution."""
        rng = rng or self.rng
        models = self.state_models.get(current_state, {})
        if not models:
            return None

        targets = list(models.keys())
        probs = np.array(
            [
                self._probability_for_next_state(current_state, t, covariates, t_entry)
                for t in targets
            ]
        )
        prob_sum = probs.sum()
        if prob_sum < 1e-10:
            return None
        probs = probs / prob_sum
        idx = rng.multinomial(1, probs).argmax()
        return targets[idx]

    def _sample_time_to_next_state(
        self,
        current_state: int,
        next_state: int,
        covariates: np.ndarray,
        t_entry: float,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Sample transition time using inverse CDF."""
        rng = rng or self.rng
        model = self.state_models[current_state][next_state]
        unique_times = self._get_unique_event_times(model)
        mask = unique_times > t_entry
        unique_times_masked = unique_times[mask]

        if len(unique_times_masked) == 0:
            return MIN_TRANSITION_TIME

        hazard = self._get_hazard(model, covariates)[mask]
        survival = self._get_survival(current_state, unique_times_masked, covariates)

        cdf = np.nancumsum(hazard * survival)
        if cdf[-1] > 0:
            cdf_normalized = cdf / cdf[-1]
        else:
            return float(unique_times_masked[0] - t_entry)

        eps = rng.uniform()
        idx = np.searchsorted(cdf_normalized, eps, side="left")
        if idx < len(unique_times_masked):
            time_abs = unique_times_masked[idx]
        else:
            time_abs = unique_times_masked[-1]
        return max(MIN_TRANSITION_TIME, float(time_abs - t_entry))

    def _one_monte_carlo_run(
        self,
        covariates: np.ndarray,
        origin_state: int,
        max_transitions: int,
        current_time: float,
        seed: Optional[int] = None,
    ) -> PatientTrajectory:
        """Generate one random trajectory via Monte Carlo simulation."""
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        traj = PatientTrajectory(
            states=[], time_at_each_state=[], covariates=pd.Series(dtype=float)
        )
        current_state = origin_state

        for _ in range(max_transitions):
            next_state = self._sample_next_state(current_state, covariates, current_time, rng=rng)
            if next_state is None:
                break

            time_to_next = self._sample_time_to_next_state(
                current_state, next_state, covariates, current_time, rng=rng
            )
            traj.states.append(current_state)
            traj.time_at_each_state.append(time_to_next)

            if next_state in self.terminal_states:
                traj.states.append(next_state)
                break

            current_state = next_state
            current_time = current_time + time_to_next

        return traj

    def run_monte_carlo_simulation(
        self,
        sample_covariates: np.ndarray,
        origin_state: int,
        current_time: float = 0,
        n_random_samples: int = 100,
        max_transitions: int = 10,
        n_jobs: int = 1,
    ) -> List[PatientTrajectory]:
        """Sample random trajectories using Monte Carlo simulation."""
        seeds = [int(self.rng.randint(0, 2**31)) for _ in range(n_random_samples)]
        if n_jobs == 1:
            return [
                self._one_monte_carlo_run(
                    sample_covariates,
                    origin_state,
                    max_transitions,
                    current_time,
                    seed=s,
                )
                for s in seeds
            ]
        return Parallel(n_jobs=n_jobs)(
            delayed(self._one_monte_carlo_run)(
                sample_covariates,
                origin_state,
                max_transitions,
                current_time,
                seed=s,
            )
            for s in seeds
        )
