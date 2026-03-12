"""
PhenoCluster Configuration — Main Config Class
================================================

Dataclass-based configuration management for the pipeline.

YAML layout (nested format)::

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

    # Top-level analysis / output sections
    stability: { ... }
    survival: { ... }
    ...

"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import yaml

from .analysis import (
    FeatureCharacterizationConfig,
    InferenceConfig,
    OutcomeConfig,
    StabilityConfig,
    SurvivalConfig,
)
from .data import (
    CategoricalEncodingConfig,
    DataQualityConfig,
    DataSplitConfig,
    ImputationConfig,
    OutlierConfig,
    RowFilterConfig,
)
from .model import (
    FeatureSelectionConfig,
    ModelSelectionConfig,
    StepMixConfig,
)
from .multistate import (
    MultistateConfig,
)
from .output import (
    CacheConfig,
    CategoricalFlowConfig,
    ExternalValidationConfig,
    LoggingConfig,
    ReferenceConfig,
    VisualizationConfig,
)


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

        if d:
            import warnings

            warnings.warn(
                f"Unknown config sections ignored: {list(d.keys())}",
                stacklevel=2,
            )

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
