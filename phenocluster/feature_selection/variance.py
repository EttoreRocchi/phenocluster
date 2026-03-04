"""
PhenoCluster Variance-Based Feature Selection
==============================================

Remove features with low variance (continuous) or low frequency (categorical).
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


class VarianceSelector(BaseFeatureSelector):
    """
    Remove features with variance below a threshold.

    For continuous features, uses sklearn's VarianceThreshold.
    For categorical features, removes features where dominant category
    exceeds a frequency threshold.

    Parameters
    ----------
    variance_threshold : float
        Minimum variance for continuous features (default 0.01)
    frequency_threshold : float
        Maximum frequency for dominant category in categorical features (default 0.99)

    Examples
    --------
    >>> selector = VarianceSelector(variance_threshold=0.01)
    >>> X_selected = selector.fit_transform(X)
    >>> print(f"Removed: {selector.get_removed_features()}")
    """

    def __init__(self, variance_threshold: float = 0.01, frequency_threshold: float = 0.99):
        super().__init__()
        self.variance_threshold = variance_threshold
        self.frequency_threshold = frequency_threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        continuous_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> "VarianceSelector":
        """
        Fit the variance selector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Ignored (for API compatibility)
        continuous_cols : List[str], optional
            Continuous column names (auto-detected if not provided)
        categorical_cols : List[str], optional
            Categorical column names (auto-detected if not provided)

        Returns
        -------
        self : VarianceSelector
            Fitted selector
        """
        all_features = list(X.columns)
        scores = {}
        keep_mask = np.ones(len(all_features), dtype=bool)

        # Auto-detect column types if not provided
        if continuous_cols is None:
            continuous_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Handle continuous features
        if continuous_cols:
            cont_data = X[continuous_cols].values

            # Use nanvar to correctly handle missing values without distortion
            cont_variances = np.nanvar(cont_data, axis=0)
            cont_mask = cont_variances > self.variance_threshold

            for i, col in enumerate(continuous_cols):
                col_idx = all_features.index(col)
                scores[col] = float(cont_variances[i])
                keep_mask[col_idx] = cont_mask[i]

        # Handle categorical features
        for col in categorical_cols:
            col_idx = all_features.index(col)
            value_counts = X[col].value_counts(normalize=True, dropna=False)

            if len(value_counts) == 0:
                # Empty column - remove
                scores[col] = 0.0
                keep_mask[col_idx] = False
            else:
                max_freq = value_counts.iloc[0]
                # Imbalance score: 1 - dominant_frequency
                # Higher score = more balanced distribution across categories
                # Note: This is NOT statistical variance; it's a simple metric
                # to filter highly skewed categorical features where one
                # category dominates (e.g., 99% "No", 1% "Yes")
                scores[col] = 1.0 - max_freq

                # Remove if dominant category exceeds threshold
                if max_freq > self.frequency_threshold:
                    keep_mask[col_idx] = False

        self._finalize_selection(all_features, keep_mask, scores)
        return self
