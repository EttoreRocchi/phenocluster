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
        stratification_used: bool = False,
        stratification_fallback_reason: Optional[str] = None,
    ):
        self.train = train
        self.test = test
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.stratification_used = stratification_used
        self.stratification_fallback_reason = stratification_fallback_reason

    @property
    def n_train(self) -> int:
        """Number of samples in the training set."""
        return len(self.train)

    @property
    def n_test(self) -> int:
        """Number of samples in the test set."""
        return len(self.test)

    @property
    def train_fraction(self) -> float:
        """Fraction of samples assigned to the training set (``n_train / n_total``)."""
        return self.n_train / (self.n_train + self.n_test)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without data)."""
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_fraction": self.train_fraction,
            "stratification_used": self.stratification_used,
            "stratification_fallback_reason": self.stratification_fallback_reason,
        }
