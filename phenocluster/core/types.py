"""
PhenoCluster Type Definitions
=============================

Type aliases and protocols for static type checking.
"""

from typing import Any, Dict, List, Optional, TypeAlias

import numpy as np
import pandas as pd

# Type aliases for common types
DataFrameType: TypeAlias = pd.DataFrame
ArrayType: TypeAlias = np.ndarray


# Result type definitions
class ModelSelectionResult:
    """Container for model selection results."""

    def __init__(
        self,
        best_model: Any,
        best_params: Dict[str, Any],
        best_score: float,
        cv_results: Optional[Dict] = None,
        all_models: Optional[List[Any]] = None,
    ):
        self.best_model = best_model
        self.best_params = best_params
        self.best_score = best_score
        self.cv_results = cv_results or {}
        self.all_models = all_models or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_results": self.cv_results,
        }


class DataSplitResult:
    """Container for train/test split results."""

    def __init__(
        self,
        train: DataFrameType,
        test: DataFrameType,
        train_indices: ArrayType,
        test_indices: ArrayType,
    ):
        self.train = train
        self.test = test
        self.train_indices = train_indices
        self.test_indices = test_indices

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    @property
    def train_fraction(self) -> float:
        return self.n_train / (self.n_train + self.n_test)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without data)."""
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_fraction": self.train_fraction,
        }
