"""
PhenoCluster: Clinical Phenotype Discovery using Latent Class / Profile Analysis.

A pipeline for identifying latent clinical phenotypes with automatic
model selection, comprehensive validation, and advanced visualizations.

Author: Ettore Rocchi <ettore.rocchi3@unibo.it>
License: MIT
"""

__version__ = "0.2.0"
__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

# Configuration classes
from .config import (
    CacheConfig,
    CategoricalEncodingConfig,
    CategoricalFlowConfig,
    DataQualityConfig,
    DataSplitConfig,
    ExternalValidationConfig,
    FeatureCharacterizationConfig,
    FeatureSelectionConfig,
    ImputationConfig,
    InferenceConfig,
    LoggingConfig,
    ModelSelectionConfig,
    MultistateConfig,
    MultistateState,
    MultistateTransition,
    OutcomeConfig,
    OutlierConfig,
    PhenoClusterConfig,
    ReferenceConfig,
    RowFilterConfig,
    StabilityConfig,
    StepMixConfig,
    SurvivalConfig,
    SurvivalTarget,
    VisualizationConfig,
)

# Core exceptions
from .core.exceptions import (
    DataSplitError,
    FeatureSelectionError,
    ModelNotFittedError,
    PhenoClusterError,
)

# Core types
from .core.types import (
    DataSplitResult,
    ModelSelectionResult,
)

# Data handling
from .data import DataPreprocessor, DataSplitter

# Evaluation
from .evaluation import (
    ClusterEvaluator,
    ClusterStatistics,
    DataQualityAssessor,
    ExternalValidator,
    FeatureCharacterizer,
    MonteCarloResults,
    MultistateAnalyzer,
    MultistateResults,
    OutcomeAnalyzer,
    StabilityAnalyzer,
    SurvivalAnalyzer,
)

# Feature selection
from .feature_selection import (
    BaseFeatureSelector,
    CorrelationSelector,
    LassoSelector,
    MixedDataFeatureSelector,
    MutualInfoSelector,
    VarianceSelector,
)

# Model selection
from .model_selection import (
    AVAILABLE_CRITERIA,
    StepMixModelSelector,
    aic_score,
    bic_score,
    caic_score,
    create_scorer,
    get_all_criteria,
    icl_score,
    relative_entropy_score,
    sabic_score,
)

# Pipeline
from .pipeline import PhenoClusterPipeline, run_pipeline

# Config profiles
from .profiles import create_config_yaml, get_profile, list_profiles

# Utils
from .utils import PhenoClusterLogger

# Visualization
from .visualization import Visualizer

__all__ = [
    # Main pipeline
    "PhenoClusterPipeline",
    "run_pipeline",
    # Configuration
    "PhenoClusterConfig",
    "CacheConfig",
    "CategoricalEncodingConfig",
    "CategoricalFlowConfig",
    "DataQualityConfig",
    "DataSplitConfig",
    "FeatureCharacterizationConfig",
    "FeatureSelectionConfig",
    "ExternalValidationConfig",
    "InferenceConfig",
    "ImputationConfig",
    "LoggingConfig",
    "ModelSelectionConfig",
    "MultistateConfig",
    "MultistateState",
    "MultistateTransition",
    "OutcomeConfig",
    "OutlierConfig",
    "ReferenceConfig",
    "RowFilterConfig",
    "StabilityConfig",
    "StepMixConfig",
    "SurvivalConfig",
    "SurvivalTarget",
    "VisualizationConfig",
    # Config profiles
    "list_profiles",
    "get_profile",
    "create_config_yaml",
    # Data handling
    "DataPreprocessor",
    "DataSplitter",
    "DataSplitResult",
    # Evaluation
    "ClusterEvaluator",
    "ClusterStatistics",
    "OutcomeAnalyzer",
    "FeatureCharacterizer",
    "StabilityAnalyzer",
    "SurvivalAnalyzer",
    "DataQualityAssessor",
    "ExternalValidator",
    "MultistateAnalyzer",
    "MultistateResults",
    "MonteCarloResults",
    # Visualization
    "Visualizer",
    # Utils
    "PhenoClusterLogger",
    # Feature selection
    "BaseFeatureSelector",
    "VarianceSelector",
    "CorrelationSelector",
    "MutualInfoSelector",
    "LassoSelector",
    "MixedDataFeatureSelector",
    # Model selection
    "StepMixModelSelector",
    "bic_score",
    "aic_score",
    "caic_score",
    "sabic_score",
    "icl_score",
    "relative_entropy_score",
    "create_scorer",
    "get_all_criteria",
    "AVAILABLE_CRITERIA",
    "ModelSelectionResult",
    # Exceptions
    "PhenoClusterError",
    "ModelNotFittedError",
    "FeatureSelectionError",
    "DataSplitError",
]
