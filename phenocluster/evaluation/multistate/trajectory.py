"""Trajectory building from wide-format clinical data."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...config import MultistateState, PhenoClusterConfig
from .types import MIN_TRANSITION_TIME, PatientTrajectory


class TrajectoryBuilder:
    """Build patient trajectories from wide-format event/time data."""

    def __init__(
        self, config: PhenoClusterConfig, n_clusters: int, reference_phenotype: int = 0, logger=None
    ):
        self.config = config
        self.n_clusters = n_clusters
        self.reference_phenotype = reference_phenotype
        self.logger = logger
        ms = config.multistate
        self.states = ms.states
        self.transitions = ms.transitions
        self.default_followup = ms.default_followup_time
        self.baseline_confounders = ms.baseline_confounders

        self._state_by_id = {s.id: s for s in self.states}
        self._transitions_from: Dict[int, list] = defaultdict(list)
        for t in self.transitions:
            self._transitions_from[t.from_state].append(t)

    def get_initial_state(self) -> MultistateState:
        for s in self.states:
            if s.state_type == "initial":
                return s
        raise ValueError("No initial state defined in multistate config")

    def get_absorbing_states(self) -> List[MultistateState]:
        return [s for s in self.states if s.state_type == "absorbing"]

    def get_terminal_state_ids(self) -> List[int]:
        return [s.id for s in self.get_absorbing_states()]

    def get_censoring_state(self) -> Optional[MultistateState]:
        for s in self.states:
            if s.state_type == "absorbing" and s.event_column is None:
                return s
        return None

    def get_state_labels(self) -> Dict[int, str]:
        return {s.id: s.name for s in self.states}

    def prepare_trajectories(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        baseline_confounders: Optional[List[str]] = None,
    ) -> Tuple[List[PatientTrajectory], Dict[int, List[int]]]:
        """Convert wide-format data to PatientTrajectory objects."""
        if self.logger:
            self.logger.info("Preparing patient trajectories...")

        trajectories = []
        phenotype_indices: Dict[int, List[int]] = {p: [] for p in range(self.n_clusters)}
        confounders = baseline_confounders or self.baseline_confounders

        confounder_means = {}
        for cov in confounders:
            if cov in data.columns:
                confounder_means[cov] = float(data[cov].mean())

        for idx, (_, row) in enumerate(data.iterrows()):
            phenotype = labels[idx]
            phenotype_indices[phenotype].append(idx)

            states, times = self._determine_state_sequence(row, idx)
            if len(states) < 2:
                if self.logger:
                    self.logger.debug(f"Patient {idx}: No transitions, skipping")
                continue

            cov_dict = self._build_covariate_dict(phenotype, row, confounders, confounder_means)
            trajectories.append(
                PatientTrajectory(
                    states=states,
                    time_at_each_state=times,
                    covariates=pd.Series(cov_dict),
                    sample_id=idx,
                )
            )

        if self.logger:
            self.logger.info(f"Created {len(trajectories)} trajectories from {len(data)} patients")
        return trajectories, phenotype_indices

    def _build_covariate_dict(self, phenotype, row, confounders, confounder_means):
        """Build covariate dictionary for a single patient."""
        cov_dict = {}
        non_ref = [p for p in range(self.n_clusters) if p != self.reference_phenotype]
        for p in non_ref:
            cov_dict[f"phenotype_{p}"] = 1.0 if phenotype == p else 0.0
        for cov in confounders:
            if cov in row.index:
                val = row[cov]
                cov_dict[cov] = float(val) if pd.notna(val) else confounder_means.get(cov, 0.0)
        return cov_dict

    def _determine_state_sequence(
        self,
        patient_data: pd.Series,
        patient_id: int,
    ) -> Tuple[List[int], List[float]]:
        """Determine the ordered sequence of states visited by a patient."""
        initial_state = self.get_initial_state()
        censoring_state = self.get_censoring_state()

        visited = self._collect_visited_states(patient_data)
        visited.sort(key=lambda x: x[1])

        states = [initial_state.id]
        times: List[float] = []
        current_time = 0.0

        for next_state, entry_time in visited:
            allowed = {t.to_state for t in self._transitions_from.get(states[-1], [])}
            if next_state in allowed:
                times.append(max(MIN_TRANSITION_TIME, entry_time - current_time))
                states.append(next_state)
                current_time = entry_time
            elif self.logger:
                self.logger.debug(
                    f"Patient {patient_id}: Skipped disallowed {states[-1]} -> {next_state}"
                )

        self._handle_censoring(states, times, current_time, patient_data, censoring_state)
        self._align_times(states, times, current_time)
        return states, times

    def _collect_visited_states(self, patient_data):
        """Collect event states and their times from patient data."""
        visited = []
        for state in self.states:
            if state.state_type == "initial":
                continue
            if state.event_column and state.time_column:
                event_val = patient_data.get(state.event_column, 0)
                time_val = patient_data.get(state.time_column, np.nan)
                if pd.notna(event_val) and event_val == 1 and pd.notna(time_val):
                    visited.append((state.id, float(time_val)))
        return visited

    def _handle_censoring(self, states, times, current_time, patient_data, censoring_state):
        """Append censoring state if patient didn't reach an absorbing state."""
        absorbing_ids = {s.id for s in self.get_absorbing_states()}
        if states[-1] not in absorbing_ids:
            max_time = self.default_followup
            for state in self.states:
                if state.time_column:
                    time_val = patient_data.get(state.time_column, np.nan)
                    if pd.notna(time_val):
                        max_time = max(max_time, float(time_val))

            if censoring_state is not None:
                times.append(max(MIN_TRANSITION_TIME, max_time - current_time))
                states.append(censoring_state.id)

    def _align_times(self, states, times, current_time):
        """Ensure times list has correct length relative to states."""
        terminal_ids = self.get_terminal_state_ids()
        if states[-1] in terminal_ids:
            if len(times) >= len(states):
                del times[len(states) - 1 :]
        else:
            while len(times) < len(states):
                times.append(self.default_followup - current_time)
