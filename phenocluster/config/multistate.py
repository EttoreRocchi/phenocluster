"""Multistate model configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MultistateState:
    """Definition of a single state in the multistate model."""

    id: int
    name: str
    state_type: str
    event_column: Optional[str] = None
    time_column: Optional[str] = None

    def __post_init__(self):
        valid_types = ["initial", "transient", "absorbing"]
        if self.state_type not in valid_types:
            raise ValueError(f"state_type must be one of {valid_types}")
        if self.state_type == "initial" and (self.event_column or self.time_column):
            raise ValueError("Initial states should not have event_column or time_column")
        if self.state_type == "transient" and not (self.event_column and self.time_column):
            raise ValueError("Transient states require both event_column and time_column")


@dataclass
class MultistateTransition:
    """Definition of an allowed transition between states."""

    name: str
    from_state: int
    to_state: int


@dataclass
class MultistateConfig:
    """Configuration for multistate survival analysis."""

    enabled: bool = False
    states: List[MultistateState] = field(default_factory=list)
    transitions: List[MultistateTransition] = field(default_factory=list)
    baseline_confounders: List[str] = field(default_factory=list)
    min_events_per_transition: int = 3
    default_followup_time: float = 30.0
    monte_carlo_n_simulations: int = 1000
    monte_carlo_time_points: List[float] = field(
        default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    )
    max_transitions_per_path: int = 10

    def __post_init__(self):
        if self.min_events_per_transition < 1:
            raise ValueError("min_events_per_transition must be at least 1")
        if self.default_followup_time <= 0:
            raise ValueError("default_followup_time must be positive")
        if self.monte_carlo_n_simulations < 1:
            raise ValueError("monte_carlo_n_simulations must be at least 1")
        if self.max_transitions_per_path < 1:
            raise ValueError("max_transitions_per_path must be at least 1")
        if not self.monte_carlo_time_points:
            raise ValueError("monte_carlo_time_points cannot be empty")
        if self.states and isinstance(self.states[0], dict):
            self.states = [MultistateState(**s) for s in self.states]  # type: ignore[arg-type]
        if self.transitions and isinstance(self.transitions[0], dict):
            self.transitions = [MultistateTransition(**t) for t in self.transitions]  # type: ignore[arg-type]
        if self.enabled and self.states:
            self._validate_state_structure()

    def _validate_state_structure(self):
        """Validate that states and transitions form a valid multistate model."""
        state_ids = {s.id for s in self.states}
        initial_states = [s for s in self.states if s.state_type == "initial"]
        absorbing_states = [s for s in self.states if s.state_type == "absorbing"]
        if len(initial_states) != 1:
            raise ValueError("Multistate model must have exactly one initial state")
        if len(absorbing_states) == 0:
            raise ValueError("Multistate model must have at least one absorbing state")
        for t in self.transitions:
            if t.from_state not in state_ids:
                raise ValueError(
                    f"Transition '{t.name}' references unknown from_state: {t.from_state}"
                )
            if t.to_state not in state_ids:
                raise ValueError(f"Transition '{t.name}' references unknown to_state: {t.to_state}")
        absorbing_ids = {s.id for s in absorbing_states}
        for t in self.transitions:
            if t.from_state in absorbing_ids:
                raise ValueError(
                    f"Transition '{t.name}' starts from absorbing state {t.from_state}, "
                    "which is not allowed"
                )
