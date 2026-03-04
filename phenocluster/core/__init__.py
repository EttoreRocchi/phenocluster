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
from .types import (
    DataSplitResult,
    ModelSelectionResult,
)

__all__ = [
    # Types
    "ModelSelectionResult",
    "DataSplitResult",
    # Exceptions
    "PhenoClusterError",
    "ModelNotFittedError",
    "FeatureSelectionError",
    "DataSplitError",
]
