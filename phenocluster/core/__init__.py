"""
PhenoCluster Core Module
========================

Types and exceptions for the PhenoCluster framework.

Note: Base classes are defined in their respective modules:
- feature_selection/base.py for BaseFeatureSelector
- model_selection/grid_search.py for StepMixModelSelector
"""

from .exceptions import (
    DataSplitError,
    FeatureSelectionError,
    ModelNotFittedError,
    PhenoClusterError,
)
from .phenotype_order import (
    PhenotypeOrderedModel,
    is_identity,
    size_order,
    unwrap_model,
)
from .types import (
    DataSplitResult,
    ModelSelectionResult,
)

__all__ = [
    # Types
    "ModelSelectionResult",
    "DataSplitResult",
    # Phenotype ordering
    "PhenotypeOrderedModel",
    "size_order",
    "is_identity",
    "unwrap_model",
    # Exceptions
    "PhenoClusterError",
    "ModelNotFittedError",
    "FeatureSelectionError",
    "DataSplitError",
]
