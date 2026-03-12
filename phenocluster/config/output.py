"""Output and infrastructure configuration dataclasses."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CacheConfig:
    """Configuration for artifact caching."""

    enabled: bool = True
    compress_level: int = 3

    def __post_init__(self):
        if not 0 <= self.compress_level <= 9:
            raise ValueError("compress_level must be between 0 and 9")


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
class ReferenceConfig:
    """Configuration for reference phenotype selection."""

    strategy: str = "largest"
    specific_id: Optional[int] = None
    health_outcome: Optional[str] = None

    def __post_init__(self):
        valid = ("largest", "healthiest", "specific")
        if self.strategy not in valid:
            raise ValueError(f"reference_phenotype.strategy must be one of {valid}")


@dataclass
class ExternalValidationConfig:
    """Configuration for external validation on independent cohort."""

    enabled: bool = False
    external_data_path: Optional[str] = None
