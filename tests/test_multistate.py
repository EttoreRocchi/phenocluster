"""Tests for the multistate analysis module."""

import numpy as np
import pandas as pd

from phenocluster.evaluation.multistate import (
    PatientTrajectory,
    TransitionHazardFitter,
)


def _make_trajectories(n=50, seed=42):
    """Create deterministic synthetic patient trajectories for testing.

    Guarantees >=5 events for each transition type (0->1, 0->2, 1->2)
    by using a fixed allocation pattern rather than random draws.
    """
    rng = np.random.RandomState(seed)
    trajectories = []

    # Fixed allocation pattern to guarantee events per transition:
    # Cycle through: 0->1->2, 0->1->censor, 0->2, 0->censor
    patterns = [
        "0_1_2",  # 0->1 event + 1->2 event
        "0_1_censor",  # 0->1 event + 1->censor
        "0_1_censor",  # 0->1 event + 1->censor
        "0_2",  # 0->2 event
        "0_censor",  # 0->censor
    ]

    for i in range(n):
        phenotype = i % 3
        pattern = patterns[i % len(patterns)]

        if pattern == "0_1_2":
            t1 = rng.exponential(10) + 0.1
            t2 = rng.exponential(5) + 0.1
            states = [0, 1, 2]
            times = [t1, t2]
        elif pattern == "0_1_censor":
            t1 = rng.exponential(10) + 0.1
            t_censor = rng.exponential(10) + 0.1
            states = [0, 1, 99]
            times = [t1, t_censor]
        elif pattern == "0_2":
            t_death = rng.exponential(15) + 0.1
            states = [0, 2]
            times = [t_death]
        else:  # 0_censor
            states = [0, 99]
            times = [30.0]

        covs = pd.Series(
            {
                "phenotype_1": 1.0 if phenotype == 1 else 0.0,
                "phenotype_2": 1.0 if phenotype == 2 else 0.0,
            }
        )
        trajectories.append(
            PatientTrajectory(
                states=states,
                time_at_each_state=times,
                covariates=covs,
                sample_id=i,
            )
        )
    return trajectories


class TestPatientTrajectory:
    def test_basic_creation(self):
        traj = PatientTrajectory(
            states=[0, 1, 2],
            time_at_each_state=[5.0, 3.0],
            covariates=pd.Series({"phenotype_1": 1.0}),
            sample_id=0,
        )
        assert len(traj.states) == 3
        assert len(traj.time_at_each_state) == 2
        assert traj.sample_id == 0


class TestTransitionHazardFitter:
    def test_fit_produces_models(self):
        """Test that fitting produces at least one Cox model."""
        trajectories = _make_trajectories(n=80)
        fitter = TransitionHazardFitter(
            terminal_states=[2, 99],
            covariate_names=["phenotype_1", "phenotype_2"],
            random_state=42,
        )
        fitter.fit(trajectories)

        # Should have at least one origin state with fitted models
        assert len(fitter.state_models) > 0

    def test_mc_simulation_produces_valid_trajectories(self):
        """Test that MC simulation produces trajectories with valid states."""
        trajectories = _make_trajectories(n=80)
        fitter = TransitionHazardFitter(
            terminal_states=[2, 99],
            covariate_names=["phenotype_1", "phenotype_2"],
            random_state=42,
        )
        fitter.fit(trajectories)

        assert len(fitter.state_models) > 0, "No models fitted - data generation is deterministic"

        covs = np.array([1.0, 0.0])
        simulated = fitter.run_monte_carlo_simulation(
            sample_covariates=covs,
            origin_state=0,
            n_random_samples=20,
            max_transitions=5,
        )

        assert len(simulated) == 20
        for traj in simulated:
            assert isinstance(traj, PatientTrajectory)
            # Every trajectory should have at least one state
            assert len(traj.states) >= 1
            # Times should be non-negative
            for t in traj.time_at_each_state:
                assert t >= 0

    def test_mc_state_probabilities_sum_to_one(self):
        """Test that state occupation probabilities approximately sum to 1."""
        trajectories = _make_trajectories(n=100)
        fitter = TransitionHazardFitter(
            terminal_states=[2, 99],
            covariate_names=["phenotype_1", "phenotype_2"],
            random_state=42,
        )
        fitter.fit(trajectories)

        assert len(fitter.state_models) > 0, "No models fitted - data generation is deterministic"

        covs = np.array([0.0, 0.0])
        simulated = fitter.run_monte_carlo_simulation(
            sample_covariates=covs,
            origin_state=0,
            n_random_samples=200,
            max_transitions=5,
        )

        # Count state occupation at t=15
        all_states = {0, 1, 2, 99}
        state_counts = {s: 0 for s in all_states}
        for traj in simulated:
            cumtime = 0.0
            current = traj.states[0] if traj.states else 0
            for i, t in enumerate(traj.time_at_each_state):
                if cumtime + t >= 15:
                    break
                cumtime += t
                if i + 1 < len(traj.states):
                    current = traj.states[i + 1]
            if sum(traj.time_at_each_state) < 15 and len(traj.states) > len(
                traj.time_at_each_state
            ):
                current = traj.states[-1]
            state_counts[current] = state_counts.get(current, 0) + 1

        total = sum(state_counts.values())
        assert total == 200
        # Probabilities should sum to 1
        prob_sum = sum(c / total for c in state_counts.values())
        assert abs(prob_sum - 1.0) < 1e-10

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces same MC results."""
        trajectories = _make_trajectories(n=80)

        results = []
        for _ in range(2):
            fitter = TransitionHazardFitter(
                terminal_states=[2, 99],
                covariate_names=["phenotype_1", "phenotype_2"],
                random_state=42,
            )
            fitter.fit(trajectories)
            assert len(fitter.state_models) > 0, "No models fitted"

            covs = np.array([0.0, 0.0])
            simulated = fitter.run_monte_carlo_simulation(
                sample_covariates=covs,
                origin_state=0,
                n_random_samples=10,
                max_transitions=5,
            )
            results.append([t.states for t in simulated])

        # Both runs should produce identical state sequences
        for i in range(len(results[0])):
            assert results[0][i] == results[1][i]


class TestPrepareTransitionData:
    """Test trajectory-to-DataFrame conversion for Cox models."""

    def _make_fitter(self, terminal_states=None, cov_names=None):
        return TransitionHazardFitter(
            terminal_states=terminal_states or [2],
            covariate_names=cov_names or ["phenotype_1"],
            random_state=42,
        )

    def _traj(self, states, times, sample_id=0, covs=None):
        covs = covs or pd.Series({"phenotype_1": 0.0})
        return PatientTrajectory(
            states=states,
            time_at_each_state=times,
            covariates=covs,
            sample_id=sample_id,
        )

    def _prepare(self, trajectories, terminal_states=None, cov_names=None):
        """Call the standalone _prepare_transition_data function."""
        from phenocluster.evaluation.multistate.transition_hazards import _prepare_transition_data

        ts = terminal_states or [2]
        cn = cov_names or ["phenotype_1"]
        return _prepare_transition_data(trajectories, cn, ts)

    def test_simple_trajectory(self):
        """0->1->2 with times [5,10] produces correct origin/target/time rows."""
        traj = self._traj([0, 1, 2], [5.0, 10.0])
        df = self._prepare([traj])

        # Two rows: 0->1 and 1->2
        assert len(df) == 2

        row0 = df.iloc[0]
        assert row0["origin_state"] == 0
        assert row0["target_state"] == 1
        assert row0["time_entry_to_origin"] == 0.0
        assert row0["time_transition_to_target"] == 5.0

        row1 = df.iloc[1]
        assert row1["origin_state"] == 1
        assert row1["target_state"] == 2
        assert row1["time_entry_to_origin"] == 5.0
        assert row1["time_transition_to_target"] == 15.0

    def test_right_censored_trajectory(self):
        """Patient censored at origin -> target_state = 0 (RIGHT_CENSORING)."""
        from phenocluster.evaluation.multistate import RIGHT_CENSORING

        traj = self._traj([0], [20.0])
        df = self._prepare([traj])

        assert len(df) == 1
        assert df.iloc[0]["target_state"] == RIGHT_CENSORING
        assert df.iloc[0]["time_transition_to_target"] == 20.0

    def test_terminal_state_stops_generation(self):
        """After reaching terminal state 2, no more rows should be generated."""
        traj = self._traj([0, 2], [10.0])
        df = self._prepare([traj], terminal_states=[2])

        # Only one transition row: 0->2
        assert len(df) == 1
        assert df.iloc[0]["origin_state"] == 0
        assert df.iloc[0]["target_state"] == 2

    def test_empty_trajectories(self):
        """Empty trajectory list produces empty DataFrame."""
        df = self._prepare([])
        assert len(df) == 0
