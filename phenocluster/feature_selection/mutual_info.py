"""
PhenoCluster Mutual Information Feature Selection
==================================================

Select features based on mutual information with target variable.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from ..core.exceptions import FeatureSelectionError
from .base import BaseFeatureSelector


class MutualInfoSelector(BaseFeatureSelector):
    """
    Select features based on mutual information with a target variable.

    Supports both classification (discrete target) and regression (continuous target).
    Automatically detects target type based on number of unique values.

    Parameters
    ----------
    n_features : int, optional
        Number of top features to select. If None, uses percentile.
    percentile : float, optional
        Percentile of features to select (0-100). Default 50.
    discrete_threshold : int
        Maximum unique values to consider target as discrete (default 10)
    random_state : int
        Random seed for reproducibility

    Examples
    --------
    >>> selector = MutualInfoSelector(n_features=10)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(f"Selected: {selector.get_selected_features()}")
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        percentile: Optional[float] = 50.0,
        discrete_threshold: int = 10,
        random_state: int = 42,
    ):
        super().__init__()
        self.n_features = n_features
        self.percentile = percentile
        self.discrete_threshold = discrete_threshold
        self.random_state = random_state
        self.mi_scores_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MutualInfoSelector":
        """
        Fit the mutual information selector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target variable (required)

        Returns
        -------
        self : MutualInfoSelector
            Fitted selector

        Raises
        ------
        FeatureSelectionError
            If y is not provided
        """
        if y is None:
            raise FeatureSelectionError(
                "MutualInfoSelector requires a target variable y", method="mutual_info"
            )

        all_features = list(X.columns)

        # Prepare data (handle missing values with column-mean imputation)
        X_array = X.values.astype(float)
        col_means = np.nanmean(X_array, axis=0)
        nan_mask = np.isnan(X_array)
        if nan_mask.any():
            X_array[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Determine if target is discrete or continuous
        unique_values = np.unique(y[~np.isnan(y)] if np.any(np.isnan(y)) else y)
        is_discrete = len(unique_values) <= self.discrete_threshold

        # Compute mutual information
        if is_discrete:
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        self.mi_scores_ = mi_func(
            X_array, y, discrete_features="auto", random_state=self.random_state
        )

        # Select features
        if self.n_features is not None:
            # Select top k features
            k = min(self.n_features, len(all_features))
            top_indices = np.argsort(self.mi_scores_)[-k:]
            keep_mask = np.zeros(len(all_features), dtype=bool)
            keep_mask[top_indices] = True
        else:
            # Select by percentile
            threshold = np.percentile(self.mi_scores_, 100 - self.percentile)
            keep_mask = self.mi_scores_ >= threshold

        # Build scores dict
        scores = {feat: float(score) for feat, score in zip(all_features, self.mi_scores_)}

        self._finalize_selection(all_features, keep_mask, scores)
        return self
