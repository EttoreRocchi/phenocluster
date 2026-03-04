"""
PhenoCluster Feature Selection Base Class
==========================================

Abstract base class for all feature selectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import FeatureSelectionError, ModelNotFittedError


class BaseFeatureSelector(ABC):
    """
    Abstract base class for feature selectors.

    All feature selectors must implement fit, transform, get_selected_features,
    and get_feature_scores methods.
    """

    def __init__(self):
        self.selected_features_: Optional[List[str]] = None
        self.removed_features_: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "BaseFeatureSelector":
        """
        Fit the feature selector to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names
        y : np.ndarray, optional
            Target variable (required for supervised methods)

        Returns
        -------
        self : BaseFeatureSelector
            Fitted selector
        """
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names

        Returns
        -------
        pd.DataFrame
            Transformed data with only selected features

        Raises
        ------
        ModelNotFittedError
            If selector has not been fitted
        FeatureSelectionError
            If selected features not found in X
        """
        if not self._is_fitted:
            raise ModelNotFittedError("FeatureSelector")
        assert self.selected_features_ is not None

        # Verify all selected features are in X
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise FeatureSelectionError(
                f"Selected features not found in input: {missing}",
                features=list(missing),
            )

        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Target variable

        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """
        Get the list of selected feature names.

        Returns
        -------
        List[str]
            Names of selected features

        Raises
        ------
        ModelNotFittedError
            If selector has not been fitted
        """
        if not self._is_fitted:
            raise ModelNotFittedError("FeatureSelector")
        assert self.selected_features_ is not None
        return self.selected_features_.copy()

    def get_removed_features(self) -> List[str]:
        """
        Get the list of removed feature names.

        Returns
        -------
        List[str]
            Names of removed features

        Raises
        ------
        ModelNotFittedError
            If selector has not been fitted
        """
        if not self._is_fitted:
            raise ModelNotFittedError("FeatureSelector")
        assert self.removed_features_ is not None
        return self.removed_features_.copy()

    def get_feature_scores(self) -> Dict[str, float]:
        """
        Get scores for all features.

        Returns
        -------
        Dict[str, float]
            Mapping of feature name to score

        Raises
        ------
        ModelNotFittedError
            If selector has not been fitted
        """
        if not self._is_fitted:
            raise ModelNotFittedError("FeatureSelector")
        assert self.feature_scores_ is not None
        return self.feature_scores_.copy()

    @property
    def n_selected(self) -> int:
        """Number of selected features."""
        if not self._is_fitted or self.selected_features_ is None:
            return 0
        return len(self.selected_features_)

    @property
    def n_removed(self) -> int:
        """Number of removed features."""
        if not self._is_fitted or self.removed_features_ is None:
            return 0
        return len(self.removed_features_)

    def _finalize_selection(
        self,
        all_features: List[str],
        mask: np.ndarray,
        scores: Optional[Dict[str, float]] = None,
    ):
        """
        Finalize the feature selection based on a boolean mask.

        Parameters
        ----------
        all_features : List[str]
            All feature names
        mask : np.ndarray
            Boolean mask where True = keep feature
        scores : Dict[str, float], optional
            Feature scores
        """
        self.selected_features_ = [f for f, m in zip(all_features, mask) if m]
        self.removed_features_ = [f for f, m in zip(all_features, mask) if not m]
        self.feature_scores_ = scores or {f: 1.0 if m else 0.0 for f, m in zip(all_features, mask)}
        self._is_fitted = True
