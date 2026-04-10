"""Multistate analysis data types and constants."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

RIGHT_CENSORING = 0
MIN_TRANSITION_TIME = 0.001
TRAJECTORY_FLOOR_WARN_FRACTION = 0.05


@dataclass
class PatientTrajectory:
    """Data container for a single patient's state trajectory."""

    states: List[int]
    time_at_each_state: List[float]
    covariates: pd.Series
    sample_id: Optional[int] = None

    def state_at_time(self, t: float, initial_state: int) -> int:
        """Return the state occupied at time t."""
        cumulative_time = 0.0
        current_state = self.states[0] if self.states else initial_state
        for i, time_in_state in enumerate(self.time_at_each_state):
            if cumulative_time + time_in_state >= t:
                return current_state
            cumulative_time += time_in_state
            if i + 1 < len(self.states):
                current_state = self.states[i + 1]
        if self.states and len(self.states) > len(self.time_at_each_state):
            return self.states[-1]
        return current_state


@dataclass
class TransitionResult:
    """Results from fitting a transition-specific Cox PH model."""

    transition_name: str
    from_state: int
    to_state: int
    n_events: int
    n_at_risk: int
    phenotype_effects: Dict[int, Dict[str, float]]
    covariate_effects: Dict[str, Dict[str, float]]


@dataclass
class PathwayResult:
    """Results from pathway frequency analysis."""

    pathway: str
    state_names: List[str]
    counts_by_phenotype: Dict[int, int]
    total_count: int
    fraction_by_phenotype: Dict[int, float]


@dataclass
class MonteCarloResults:
    """Monte Carlo simulation results for state occupation probabilities."""

    time_points: List[float]
    state_probabilities: Dict[int, Dict[int, List[float]]]
    n_simulations: int
    simulation_summary: Dict[str, Any] = field(default_factory=dict)
    state_probabilities_lower: Optional[Dict[int, Dict[int, List[float]]]] = None
    state_probabilities_upper: Optional[Dict[int, Dict[int, List[float]]]] = None


@dataclass
class MultistateResults:
    """Complete multistate analysis results."""

    model_summary: Dict[str, Any]
    transition_results: Dict[str, TransitionResult]
    pathway_results: List[PathwayResult]
    state_occupation_probabilities: Optional[MonteCarloResults] = None
    fitted_model: Optional[Any] = None
