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
    """Container for train/test split results.

    Parameters
    ----------
    train : pd.DataFrame
        Derivation (training) partition.
    test : pd.DataFrame
        Validation (test) partition.
    train_indices, test_indices : np.ndarray
        Row indices into the original dataframe.
    stratification_used : bool, default False
        Whether the split was stratified on a categorical column.
    stratification_fallback_reason : str, optional
        Set when stratification was requested but skipped (e.g., a stratum had
        fewer than two samples).
    partition_kind : str, optional
        ``"random"``, ``"temporal"``, or ``"site"``. Identifies how the split
        was produced, for downstream reporting.
    partition_label : str, optional
        Free-form label distinguishing one validation cohort from another
        (e.g., ``"site=ANTWERP"`` or ``"window=2022_2024"``). Used by
        leave-one-group-out and sliding/expanding temporal schemes.
    derivation_window, validation_window : str, optional
        Human-readable window descriptors for temporal splits
        (e.g., ``"<=2020-12-31"``, ``">2020-12-31"``).
    """

    def __init__(
        self,
        train: DataFrameType,
        test: DataFrameType,
        train_indices: ArrayType,
        test_indices: ArrayType,
        stratification_used: bool = False,
        stratification_fallback_reason: Optional[str] = None,
        partition_kind: Optional[str] = None,
        partition_label: Optional[str] = None,
        derivation_window: Optional[str] = None,
        validation_window: Optional[str] = None,
    ):
        self.train = train
        self.test = test
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.stratification_used = stratification_used
        self.stratification_fallback_reason = stratification_fallback_reason
        self.partition_kind = partition_kind
        self.partition_label = partition_label
        self.derivation_window = derivation_window
        self.validation_window = validation_window

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
            "partition_kind": self.partition_kind,
            "partition_label": self.partition_label,
            "derivation_window": self.derivation_window,
            "validation_window": self.validation_window,
        }
