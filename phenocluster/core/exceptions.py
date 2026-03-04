"""
PhenoCluster Custom Exceptions
==============================

Exception classes for error handling across the framework.
"""

from typing import Any, Dict, List, Optional


class PhenoClusterError(Exception):
    """Base exception for all PhenoCluster errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ModelNotFittedError(PhenoClusterError):
    """Raised when trying to use an unfitted model."""

    def __init__(self, model_name: str = "Model"):
        message = f"{model_name} has not been fitted. Call fit() first."
        super().__init__(message)
        self.model_name = model_name


class FeatureSelectionError(PhenoClusterError):
    """Raised when feature selection fails."""

    def __init__(
        self,
        message: str,
        method: Optional[str] = None,
        features: Optional[List[str]] = None,
    ):
        details = {}
        if method:
            details["method"] = method
        if features:
            details["features"] = features
        super().__init__(message, details)
        self.method = method
        self.features = features


class DataSplitError(PhenoClusterError):
    """Raised when data splitting fails."""

    def __init__(
        self,
        message: str,
        n_samples: Optional[int] = None,
        split_sizes: Optional[Dict[str, Any]] = None,
    ):
        details = {}
        if n_samples is not None:
            details["n_samples"] = n_samples
        if split_sizes:
            details["split_sizes"] = split_sizes
        super().__init__(message, details)
        self.n_samples = n_samples
        self.split_sizes = split_sizes


class InsufficientDataError(PhenoClusterError):
    """Raised when there isn't enough data for an operation."""

    def __init__(
        self,
        message: str,
        n_samples: Optional[int] = None,
        min_required: Optional[int] = None,
    ):
        details = {}
        if n_samples is not None:
            details["n_samples"] = n_samples
        if min_required is not None:
            details["min_required"] = min_required
        super().__init__(message, details)
