"""
PhenoCluster Correlation-Based Feature Selection
=================================================

Remove highly correlated features, keeping the one with higher variance.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseFeatureSelector


class CorrelationSelector(BaseFeatureSelector):
    """
    Remove features that are highly correlated with other features.

    When two features have correlation above the threshold, the feature
    with lower variance is removed.

    Parameters
    ----------
    threshold : float
        Correlation threshold above which features are considered redundant (default 0.95)
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall' (default 'pearson')

    Examples
    --------
    >>> selector = CorrelationSelector(threshold=0.95)
    >>> X_selected = selector.fit_transform(X)
    >>> print(f"Removed: {selector.get_removed_features()}")
    """

    def __init__(self, threshold: float = 0.95, method: str = "pearson"):
        super().__init__()
        self.threshold = threshold
        self.method = method
        self.correlation_matrix_: Optional[pd.DataFrame] = None
        self.correlated_pairs_: Optional[List[Tuple[str, str, float]]] = None

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "CorrelationSelector":
        """
        Fit the correlation selector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (numeric features only)
        y : np.ndarray, optional
            Ignored (for API compatibility)

        Returns
        -------
        self : CorrelationSelector
            Fitted selector
        """
        # Only use numeric columns for correlation
        numeric_X = X.select_dtypes(include=[np.number])
        all_features = list(X.columns)
        numeric_features = list(numeric_X.columns)

        self.correlation_matrix_ = numeric_X.corr(method=self.method)

        # Compute variances for tie-breaking
        variances = numeric_X.var()

        self.correlated_pairs_ = []
        features_to_remove = set()

        # Get upper triangle of correlation matrix
        corr_matrix = self.correlation_matrix_.values
        n_features = len(numeric_features)

        for i in range(n_features):
            if numeric_features[i] in features_to_remove:
                continue

            for j in range(i + 1, n_features):
                if numeric_features[j] in features_to_remove:
                    continue

                corr_val = abs(corr_matrix[i, j])

                if corr_val > self.threshold:
                    feat_i = numeric_features[i]
                    feat_j = numeric_features[j]

                    self.correlated_pairs_.append((feat_i, feat_j, corr_val))

                    # Remove feature with lower variance
                    if variances[feat_i] >= variances[feat_j]:
                        features_to_remove.add(feat_j)
                    else:
                        features_to_remove.add(feat_i)

        # Build mask for all features
        keep_mask = np.ones(len(all_features), dtype=bool)
        scores = {}

        numeric_index = {f: idx for idx, f in enumerate(numeric_features)}
        for i, feat in enumerate(all_features):
            if feat in features_to_remove:
                keep_mask[i] = False
                scores[feat] = 0.0
            elif feat in numeric_index:
                # Score based on maximum correlation with other features
                feat_idx = numeric_index[feat]
                max_corr = (
                    np.max(np.abs(np.delete(corr_matrix[feat_idx], feat_idx)))
                    if len(numeric_features) > 1
                    else 0
                )
                scores[feat] = 1.0 - max_corr  # Higher score = less correlated
            else:
                # Non-numeric features kept by default
                scores[feat] = 1.0

        self._finalize_selection(all_features, keep_mask, scores)
        return self
