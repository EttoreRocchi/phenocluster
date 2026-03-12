"""Model-related configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelSelectionConfig:
    """Configuration for automatic model selection via information criteria."""

    enabled: bool = True
    min_clusters: int = 2
    max_clusters: int = 6
    criterion: str = "BIC"
    n_init: List[int] = field(default_factory=lambda: [100])
    min_cluster_size: Union[int, float] = 0.05
    n_jobs: int = -1
    refit: bool = True
    # Populated from global.random_state - not user-facing in YAML
    random_state: int = 42

    def __post_init__(self):
        valid_criteria = ["BIC", "AIC", "ICL", "CAIC", "SABIC", "ENTROPY"]
        if self.criterion.upper() not in valid_criteria:
            raise ValueError(f"criterion must be one of {valid_criteria}")
        self.criterion = self.criterion.upper()

        if isinstance(self.min_cluster_size, float) and 0 < self.min_cluster_size < 1:
            pass
        elif isinstance(self.min_cluster_size, (int, float)) and self.min_cluster_size >= 1:
            self.min_cluster_size = int(self.min_cluster_size)
        else:
            raise ValueError(
                "min_cluster_size must be an integer >= 1 (absolute count) "
                "or a float in (0, 1) (percentage of samples)"
            )

        if not self.n_init or any(n < 1 for n in self.n_init):
            raise ValueError("n_init must be a non-empty list of positive integers")

    def get_min_cluster_size(self, n_samples: int) -> int:
        """Get the minimum cluster size as an absolute count."""
        if isinstance(self.min_cluster_size, float) and 0 < self.min_cluster_size < 1:
            return max(1, int(n_samples * self.min_cluster_size))
        return int(self.min_cluster_size)


@dataclass
class StepMixConfig:
    """Configuration for StepMix model parameters.

    Only includes parameters actually consumed by the pipeline.
    """

    abs_tol: float = 1e-10
    rel_tol: float = 1e-7
    n_init: int = 100
    max_iter: int = 1000
    # Populated from global.random_state - not user-facing in YAML
    random_state: int = 42


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""

    enabled: bool = False
    method: str = "variance"
    variance_threshold: float = 0.01
    frequency_threshold: float = 0.99
    correlation_threshold: float = 0.9
    n_features: Optional[int] = None
    percentile: float = 50.0
    lasso_alpha: Optional[float] = None
    target_column: Optional[str] = None
    # Populated from global.random_state - not user-facing in YAML
    random_state: int = 42

    def __post_init__(self):
        valid_methods = ["variance", "correlation", "mutual_info", "lasso", "combined"]
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        # Auto-derive require_target from method
        self._require_target = self.method in ["mutual_info", "lasso"]

        if self.enabled and self._require_target and self.target_column is None:
            raise ValueError(
                f"Feature selection method '{self.method}' requires a target column. "
                f"Set feature_selection.target_column in config."
            )

        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")
        if not 0 < self.frequency_threshold <= 1:
            raise ValueError("frequency_threshold must be between 0 and 1")
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if not 0 < self.percentile <= 100:
            raise ValueError("percentile must be between 0 and 100")

    @property
    def require_target(self) -> bool:
        """Whether the method requires a target variable."""
        return self._require_target
