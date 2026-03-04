"""
PhenoCluster Configuration Module
==================================

Dataclass-based configuration management for the pipeline.

YAML layout (new nested format)::

    global:
      project_name: ...
      output_dir: ...
      random_state: 42

    data:
      continuous_columns: [...]
      categorical_columns: [...]
      outcome_columns: [...]
      split: { test_size: 0.2, ... }

    preprocessing:
      row_filter: { ... }
      imputation: { ... }
      categorical_encoding: { ... }
      outlier: { ... }
      feature_selection: { ... }

    model:
      n_clusters: 3
      selection: { ... }
      stepmix: { ... }

    # Top-level analysis / output sections (unchanged)
    stability: { ... }
    survival: { ... }
    ...

"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


def _config_to_dict(obj, exclude=()):
    """Convert a config dataclass to a plain dict, handling nested dataclasses."""
    result = {}
    for f in dataclasses.fields(obj):
        if f.name in exclude or f.name.startswith("_"):
            continue
        val = getattr(obj, f.name)
        if dataclasses.is_dataclass(val):
            result[f.name] = _config_to_dict(val)
        elif (
            isinstance(val, (list, tuple))
            and val
            and all(dataclasses.is_dataclass(item) for item in val)
        ):
            result[f.name] = [_config_to_dict(item) for item in val]
        elif isinstance(val, tuple):
            result[f.name] = list(val)
        else:
            result[f.name] = val
    return result


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
class StabilityConfig:
    """Configuration for cluster stability analysis using consensus clustering."""

    enabled: bool = True
    n_runs: int = 100
    subsample_fraction: float = 0.8
    n_jobs: int = -1

    def __post_init__(self):
        if not 0 < self.subsample_fraction < 1:
            raise ValueError("sample_fraction must be between 0 and 1")


@dataclass
class CategoricalFlowConfig:
    """Configuration for categorical variable visualization."""

    group_by_prefix: bool = True
    prefix_separator: str = "_"
    custom_groups: Optional[Dict[str, List[str]]] = None
    show_sankey: bool = False
    show_proportion_heatmap: bool = True
    min_category_pct: float = 0.03

    def __post_init__(self):
        if self.custom_groups is None:
            self.custom_groups = {}
        if not 0 <= self.min_category_pct < 1:
            raise ValueError("min_category_pct must be between 0 and 1")


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


@dataclass
class ExternalValidationConfig:
    """Configuration for external validation on independent cohort."""

    enabled: bool = False
    external_data_path: Optional[str] = None


@dataclass
class CacheConfig:
    """Configuration for artifact caching."""

    enabled: bool = True
    compress_level: int = 3

    def __post_init__(self):
        if not 0 <= self.compress_level <= 9:
            raise ValueError("compress_level must be between 0 and 9")


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


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "detailed"
    log_to_file: bool = True
    log_file: str = "phenocluster.log"
    quiet_mode: bool = False

    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        self.level = self.level.upper()
        valid_formats = ["minimal", "standard", "detailed"]
        if self.format.lower() not in valid_formats:
            raise ValueError(f"format must be one of {valid_formats}")
        self.format = self.format.lower()


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    save_plots: bool = True
    dpi: int = 300


@dataclass
class ReferenceConfig:
    """Configuration for reference phenotype selection."""

    strategy: str = "largest"
    specific_id: Optional[int] = None
    health_outcome: Optional[str] = None

    def __post_init__(self):
        valid = ("largest", "healthiest", "specific")
        if self.strategy not in valid:
            raise ValueError(f"reference_phenotype.strategy must be one of {valid}")


# Main config


@dataclass
class PhenoClusterConfig:
    """Main configuration for PhenoCluster pipeline."""

    # Project settings
    project_name: str = "PhenoCluster"
    output_dir: str = "results"

    # Feature columns
    continuous_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)

    # Model parameters
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    stepmix: StepMixConfig = field(default_factory=StepMixConfig)
    n_clusters: int = 3

    # Global random seed
    random_state: int = 42

    # Data splitting
    data_split: DataSplitConfig = field(default_factory=DataSplitConfig)

    # Feature selection
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)

    # Preprocessing
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    categorical_encoding: CategoricalEncodingConfig = field(
        default_factory=CategoricalEncodingConfig
    )
    outlier: OutlierConfig = field(default_factory=OutlierConfig)
    row_filter: RowFilterConfig = field(default_factory=RowFilterConfig)

    # Evaluation
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    categorical_flow: CategoricalFlowConfig = field(default_factory=CategoricalFlowConfig)
    feature_characterization: FeatureCharacterizationConfig = field(
        default_factory=FeatureCharacterizationConfig
    )
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)

    # Analysis
    outcome: OutcomeConfig = field(default_factory=lambda: OutcomeConfig(enabled=False))
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    multistate: MultistateConfig = field(default_factory=MultistateConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    reference_phenotype: ReferenceConfig = field(default_factory=ReferenceConfig)
    external_validation: ExternalValidationConfig = field(default_factory=ExternalValidationConfig)

    # Output
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @property
    def outcome_columns(self) -> List[str]:
        """Convenience accessor for outcome column names."""
        return self.outcome.outcome_columns

    def validate(self):
        """Validate config for pipeline execution. Raises ValueError on issues."""
        if not self.continuous_columns and not self.categorical_columns:
            raise ValueError("Must specify at least one continuous or categorical column")

    # I/O

    @classmethod
    def from_yaml(cls, config_file: Union[str, Path]) -> "PhenoClusterConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, config_file: Union[str, Path]) -> "PhenoClusterConfig":
        """Load configuration from JSON file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PhenoClusterConfig":
        """Create configuration from dictionary (nested format)."""
        d = dict(config_dict)
        return cls._from_nested_dict(d)

    # New nested format

    @classmethod
    def _from_nested_dict(cls, d: dict) -> "PhenoClusterConfig":
        """Parse the new nested YAML format."""
        global_cfg = d.pop("global", {})
        data_cfg = d.pop("data", {})
        preproc_cfg = d.pop("preprocessing", {})
        model_cfg = d.pop("model", {})

        # Extract global values
        project_name = global_cfg.get("project_name", "PhenoCluster")
        output_dir = global_cfg.get("output_dir", "results")
        random_state = global_cfg.get("random_state", 42)

        # Data section
        continuous_columns = data_cfg.get("continuous_columns", [])
        categorical_columns = data_cfg.get("categorical_columns", [])
        data_split_dict = data_cfg.get("split", {})

        # Preprocessing sub-sections
        row_filter_dict = preproc_cfg.get("row_filter", {})
        imputation_dict = preproc_cfg.get("imputation", {})
        categorical_encoding_dict = preproc_cfg.get("categorical_encoding", {})
        outlier_dict = preproc_cfg.get("outlier", {})
        feature_selection_dict = preproc_cfg.get("feature_selection", {})

        # Model sub-sections
        n_clusters = model_cfg.get("n_clusters", 3)
        model_selection_dict = model_cfg.get("selection", {})
        stepmix_dict = model_cfg.get("stepmix", {})

        # Remaining top-level sections
        outcome_dict = d.pop("outcome", {})
        stability_dict = d.pop("stability", {})
        survival_dict = d.pop("survival", {})
        multistate_dict = d.pop("multistate", {})
        inference_dict = d.pop("inference", {})
        cache_dict = d.pop("cache", {})
        reference_phenotype_dict = d.pop("reference_phenotype", {})
        external_validation_dict = d.pop("external_validation", {})
        logging_dict = d.pop("logging", {})
        visualization_dict = d.pop("visualization", {})
        data_quality_dict = d.pop("data_quality", {})
        categorical_flow_dict = d.pop("categorical_flow", {})
        feature_char_dict = d.pop("feature_characterization", {})

        # Handle multistate monte_carlo nested sub-block
        _unpack_multistate_monte_carlo(multistate_dict)

        # Propagate global random_state to sub-configs
        _propagate_random_state(
            random_state,
            [
                data_split_dict,
                model_selection_dict,
                feature_selection_dict,
                stepmix_dict,
            ],
        )

        # Remove deprecated cv_folds if present in user config
        model_selection_dict.pop("cv_folds", None)

        # Default model selection n_init from stepmix if not explicitly set
        if "n_init" not in model_selection_dict and "n_init" in stepmix_dict:
            model_selection_dict["n_init"] = [stepmix_dict["n_init"]]

        # Build sub-config objects
        scalars = {
            "project_name": project_name,
            "output_dir": output_dir,
            "random_state": random_state,
            "continuous_columns": continuous_columns,
            "categorical_columns": categorical_columns,
            "n_clusters": n_clusters,
        }
        sub_dicts = {
            "outcome": outcome_dict,
            "data_split": data_split_dict,
            "model_selection": model_selection_dict,
            "stepmix": stepmix_dict,
            "row_filter": row_filter_dict,
            "imputation": imputation_dict,
            "categorical_encoding": categorical_encoding_dict,
            "outlier": outlier_dict,
            "feature_selection": feature_selection_dict,
            "stability": stability_dict,
            "survival": survival_dict,
            "multistate": multistate_dict,
            "inference": inference_dict,
            "cache": cache_dict,
            "reference_phenotype": reference_phenotype_dict,
            "external_validation": external_validation_dict,
            "logging": logging_dict,
            "visualization": visualization_dict,
            "data_quality": data_quality_dict,
            "categorical_flow": categorical_flow_dict,
            "feature_characterization": feature_char_dict,
        }
        return cls._build(scalars=scalars, sub_dicts=sub_dicts)

    # Shared builder

    # Mapping: PhenoClusterConfig field name -> config dataclass
    _SUBCONFIG_FIELDS = {
        "outcome": OutcomeConfig,
        "model_selection": ModelSelectionConfig,
        "stepmix": StepMixConfig,
        "data_split": DataSplitConfig,
        "feature_selection": FeatureSelectionConfig,
        "imputation": ImputationConfig,
        "categorical_encoding": CategoricalEncodingConfig,
        "outlier": OutlierConfig,
        "row_filter": RowFilterConfig,
        "stability": StabilityConfig,
        "categorical_flow": CategoricalFlowConfig,
        "feature_characterization": FeatureCharacterizationConfig,
        "data_quality": DataQualityConfig,
        "survival": SurvivalConfig,
        "multistate": MultistateConfig,
        "inference": InferenceConfig,
        "cache": CacheConfig,
        "reference_phenotype": ReferenceConfig,
        "external_validation": ExternalValidationConfig,
        "logging": LoggingConfig,
        "visualization": VisualizationConfig,
    }

    @classmethod
    def _build(cls, *, scalars, sub_dicts) -> "PhenoClusterConfig":
        """Build config from scalar values and sub-config dicts."""
        # Convert winsorize_limits list -> tuple
        outlier_d = sub_dicts.get("outlier", {})
        if "winsorize_limits" in outlier_d and isinstance(outlier_d["winsorize_limits"], list):
            outlier_d["winsorize_limits"] = tuple(outlier_d["winsorize_limits"])

        kwargs = dict(scalars)
        for field_name, config_cls in cls._SUBCONFIG_FIELDS.items():
            d = sub_dicts.get(field_name, {})
            if field_name == "outcome":
                kwargs[field_name] = config_cls(**d) if d else OutcomeConfig(enabled=False)
            else:
                kwargs[field_name] = config_cls(**d)
        return cls(**kwargs)

    # Serialization - new nested format

    def to_dict(self) -> dict:
        """Convert configuration to the new nested dict layout."""
        _exc_rs = ("random_state",)
        ms = _config_to_dict(self.multistate)
        # Re-nest monte_carlo_* -> monte_carlo.*
        ms["monte_carlo"] = {
            "n_simulations": ms.pop("monte_carlo_n_simulations"),
            "time_points": ms.pop("monte_carlo_time_points"),
            "max_transitions_per_path": ms.pop("max_transitions_per_path"),
        }
        return {
            "global": {
                "project_name": self.project_name,
                "output_dir": self.output_dir,
                "random_state": self.random_state,
            },
            "data": {
                "continuous_columns": self.continuous_columns,
                "categorical_columns": self.categorical_columns,
                "split": _config_to_dict(self.data_split, exclude=_exc_rs),
            },
            "preprocessing": {
                "row_filter": _config_to_dict(self.row_filter),
                "imputation": _config_to_dict(self.imputation),
                "categorical_encoding": _config_to_dict(self.categorical_encoding),
                "outlier": _config_to_dict(self.outlier),
                "feature_selection": _config_to_dict(self.feature_selection, exclude=_exc_rs),
            },
            "model": {
                "n_clusters": self.n_clusters,
                "selection": _config_to_dict(self.model_selection, exclude=_exc_rs),
                "stepmix": _config_to_dict(self.stepmix, exclude=_exc_rs),
            },
            "outcome": _config_to_dict(self.outcome),
            "stability": _config_to_dict(self.stability),
            "survival": _config_to_dict(self.survival),
            "multistate": ms,
            "inference": _config_to_dict(self.inference),
            "cache": _config_to_dict(self.cache),
            "reference_phenotype": _config_to_dict(self.reference_phenotype),
            "external_validation": _config_to_dict(self.external_validation),
            "logging": _config_to_dict(self.logging),
            "visualization": _config_to_dict(self.visualization),
            "data_quality": _config_to_dict(self.data_quality),
            "categorical_flow": _config_to_dict(self.categorical_flow),
            "feature_characterization": _config_to_dict(self.feature_characterization),
        }

    def to_yaml(self, output_file: Union[str, Path]):
        """Save configuration to YAML file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, output_file: Union[str, Path]):
        """Save configuration to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save(self, output_file: Union[str, Path], format: str = "yaml"):
        """Save configuration to file."""
        if format == "yaml":
            self.to_yaml(output_file)
        elif format == "json":
            self.to_json(output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")


def _propagate_random_state(random_state: int, dicts: list) -> None:
    """Set ``random_state`` in each sub-config dict (internal propagation)."""
    for d in dicts:
        d["random_state"] = random_state


def _unpack_multistate_monte_carlo(ms_dict: dict) -> None:
    """Flatten ``multistate.monte_carlo.*`` -> ``multistate.monte_carlo_*``."""
    mc = ms_dict.pop("monte_carlo", None)
    if mc and isinstance(mc, dict):
        if "n_simulations" in mc:
            ms_dict.setdefault("monte_carlo_n_simulations", mc["n_simulations"])
        if "time_points" in mc:
            ms_dict.setdefault("monte_carlo_time_points", mc["time_points"])
        if "max_transitions_per_path" in mc:
            ms_dict.setdefault("max_transitions_per_path", mc["max_transitions_per_path"])
