"""Data-related configuration dataclasses."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class DataSplitConfig:
    """Configuration for train/test data splitting."""

    test_size: float = 0.2
    stratify_by: Optional[str] = None
    shuffle: bool = True
    # Populated from global.random_state - not user-facing in YAML
    random_state: int = 42

    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")


@dataclass
class ImputationConfig:
    """Configuration for missing data imputation."""

    enabled: bool = False
    method: str = "iterative"
    estimator: str = "bayesian_ridge"
    max_iter: int = 10
    n_nearest_features: Optional[int] = None

    def __post_init__(self):
        if self.enabled:
            valid_methods = ["iterative", "knn", "simple"]
            if self.method.lower() not in valid_methods:
                raise ValueError(f"method must be one of {valid_methods}")
            self.method = self.method.lower()
            valid_estimators = ["bayesian_ridge", "random_forest"]
            if self.estimator.lower() not in valid_estimators:
                raise ValueError(f"estimator must be one of {valid_estimators}")
            self.estimator = self.estimator.lower()


@dataclass
class CategoricalEncodingConfig:
    """Configuration for categorical variable encoding."""

    method: str = "label"
    handle_unknown: str = "ignore"

    def __post_init__(self):
        valid_methods = ["label", "onehot", "frequency"]
        if self.method.lower() not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        self.method = self.method.lower()


@dataclass
class OutlierConfig:
    """Configuration for outlier detection and handling."""

    enabled: bool = False
    method: str = "isolation_forest"
    contamination: Union[float, str] = "auto"
    winsorize_limits: tuple = (0.01, 0.01)

    def __post_init__(self):
        valid_methods = ["isolation_forest", "winsorize"]
        if self.method.lower() not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        self.method = self.method.lower()
        if self.contamination != "auto":
            if not isinstance(self.contamination, (int, float)):
                raise ValueError("contamination must be 'auto' or a numeric value in (0, 0.5]")
            if not (0 < self.contamination <= 0.5):
                raise ValueError("contamination must be 'auto' or a numeric value in (0, 0.5]")


@dataclass
class RowFilterConfig:
    """Configuration for row-level missing data filtering."""

    enabled: bool = True
    max_missing_pct: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.max_missing_pct <= 1.0:
            raise ValueError("max_missing_pct must be between 0.0 and 1.0")


@dataclass
class DataQualityConfig:
    """Configuration for data quality reporting."""

    enabled: bool = True
    missing_threshold: float = 0.15
    correlation_threshold: float = 0.9
    variance_threshold: float = 0.01
    generate_report: bool = True

    def __post_init__(self):
        if not 0 < self.missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1")
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        if self.variance_threshold < 0:
            raise ValueError("variance_threshold must be non-negative")
