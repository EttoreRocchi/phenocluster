"""Analysis-related configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StabilityConfig:
    """Configuration for cluster stability analysis using consensus clustering."""

    enabled: bool = True
    n_runs: int = 100
    subsample_fraction: float = 0.8
    n_jobs: int = -1

    def __post_init__(self):
        if not 0 < self.subsample_fraction < 1:
            raise ValueError("subsample_fraction must be between 0 and 1")


@dataclass
class FeatureCharacterizationConfig:
    """Configuration for Descriptive Feature Characterization."""

    group_by_prefix: bool = True
    prefix_separator: str = "_"
    custom_groups: Optional[Dict[str, List[str]]] = None
    n_top_per_group: int = 5
    n_top_overall: int = 20

    def __post_init__(self):
        if self.custom_groups is None:
            self.custom_groups = {}
        if self.n_top_per_group < 1:
            raise ValueError("n_top_per_group must be >= 1")
        if self.n_top_overall < 1:
            raise ValueError("n_top_overall must be >= 1")


@dataclass
class OutcomeConfig:
    """Configuration for binary outcome association analysis."""

    enabled: bool = True
    outcome_columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.enabled and not self.outcome_columns:
            raise ValueError("outcome.enabled=true but no outcome_columns specified")


@dataclass
class SurvivalTarget:
    """A single survival analysis target."""

    name: str
    time_column: str
    event_column: str
    time_unit: str = "days"


@dataclass
class SurvivalConfig:
    """Configuration for survival analysis."""

    enabled: bool = False
    use_weighted: bool = False
    targets: List[SurvivalTarget] = field(default_factory=list)

    def __post_init__(self):
        if self.targets and isinstance(self.targets[0], dict):
            self.targets = [SurvivalTarget(**t) for t in self.targets]  # type: ignore[arg-type]


@dataclass
class InferenceConfig:
    """Configuration for statistical inference throughout the pipeline.

    Controls logistic regression (outcomes), Cox proportional hazards
    (survival/multistate), and multiple-comparison correction.
    """

    enabled: bool = True
    confidence_level: float = 0.95
    fdr_correction: bool = True
    outcome_test: str = "auto"
    cox_penalizer: float = 0.0

    def __post_init__(self):
        valid_outcome = ["auto", "chi-square", "fisher"]
        if self.outcome_test not in valid_outcome:
            raise ValueError(f"outcome_test must be one of {valid_outcome}")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if self.cox_penalizer < 0:
            raise ValueError("cox_penalizer must be non-negative")
