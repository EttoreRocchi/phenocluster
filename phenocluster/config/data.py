"""Data-related configuration dataclasses."""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

VALID_SPLIT_STRATEGIES = ("random", "temporal", "holdout_group", "leave_one_group_out")
VALID_TIME_SCHEMES = ("cutoff", "fraction", "sliding", "expanding")


@dataclass
class DataSplitConfig:
    """Configuration for train/test data splitting.

    Supports four strategies via the ``strategy`` field:

    - ``"random"`` (default): random or stratified split using
      ``test_size`` and ``stratify_by``. Backwards-compatible.
    - ``"temporal"``: chronological split on ``time_column``. Sub-strategy is
      controlled by ``time_scheme``: ``"cutoff"`` (rows with
      ``time_column <= time_cutoff`` are derivation), ``"fraction"`` (the
      latest ``time_test_fraction`` of rows is the test set), ``"sliding"``
      and ``"expanding"`` produce ``n_windows`` rolling cohorts.
    - ``"holdout_group"``: rows where ``group_column`` is in
      ``holdout_values`` form the validation set.
    - ``"leave_one_group_out"``: yields one (derivation, validation) pair
      per unique value of ``group_column``.
    """

    test_size: float = 0.2
    stratify_by: Optional[str] = None
    shuffle: bool = True
    random_state: int = 42

    strategy: str = "random"
    time_column: Optional[str] = None
    time_cutoff: Optional[Union[str, Any]] = None
    time_test_fraction: Optional[float] = None
    time_scheme: str = "cutoff"
    n_windows: Optional[int] = None
    group_column: Optional[str] = None
    holdout_values: Optional[List[Any]] = field(default=None)
    min_validation_size: int = 25

    def __post_init__(self):
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        if self.strategy not in VALID_SPLIT_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {VALID_SPLIT_STRATEGIES}, got '{self.strategy}'"
            )

        if self.strategy == "temporal":
            if not self.time_column:
                raise ValueError("temporal strategy requires 'time_column'")
            if self.time_scheme not in VALID_TIME_SCHEMES:
                raise ValueError(
                    f"time_scheme must be one of {VALID_TIME_SCHEMES}, got '{self.time_scheme}'"
                )
            if self.time_scheme == "cutoff" and self.time_cutoff is None:
                raise ValueError("time_scheme='cutoff' requires 'time_cutoff'")
            if self.time_scheme == "fraction":
                if self.time_test_fraction is None:
                    raise ValueError("time_scheme='fraction' requires 'time_test_fraction'")
                if not 0 < self.time_test_fraction < 1:
                    raise ValueError("time_test_fraction must be between 0 and 1")
            if self.time_scheme in ("sliding", "expanding"):
                if not self.n_windows or self.n_windows < 2:
                    raise ValueError(f"time_scheme='{self.time_scheme}' requires n_windows >= 2")

        if self.strategy == "holdout_group":
            if not self.group_column:
                raise ValueError("holdout_group strategy requires 'group_column'")
            if not self.holdout_values:
                raise ValueError("holdout_group strategy requires non-empty 'holdout_values'")

        if self.strategy == "leave_one_group_out":
            if not self.group_column:
                raise ValueError("leave_one_group_out strategy requires 'group_column'")

        if self.min_validation_size < 1:
            raise ValueError("min_validation_size must be >= 1")


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
