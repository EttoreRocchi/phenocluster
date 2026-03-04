"""
PhenoCluster LASSO Feature Selection
=====================================

Select features using L1 regularization (LASSO).
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from ..core.exceptions import FeatureSelectionError
from .base import BaseFeatureSelector


class LassoSelector(BaseFeatureSelector):
    """
    Select features using L1 regularization (LASSO).

    Uses LassoCV for regression targets and LogisticRegressionCV with L1
    for classification targets. Features with non-zero coefficients are selected.

    Parameters
    ----------
    alpha : float, optional
        Regularization strength. If None, uses cross-validation to find optimal.
    n_features : int, optional
        Maximum number of features to select. If None, uses all non-zero.
    cv : int
        Number of CV folds for alpha selection (default 5)
    discrete_threshold : int
        Maximum unique values to consider target as discrete (default 10)
    random_state : int
        Random seed for reproducibility

    Examples
    --------
    >>> selector = LassoSelector(n_features=10)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(f"Selected: {selector.get_selected_features()}")
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        n_features: Optional[int] = None,
        cv: int = 5,
        discrete_threshold: int = 10,
        random_state: int = 42,
    ):
        super().__init__()
        self.alpha = alpha
        self.n_features = n_features
        self.cv = cv
        self.discrete_threshold = discrete_threshold
        self.random_state = random_state
        self.coefficients_: Optional[np.ndarray] = None
        self.alpha_: Optional[float] = None
        self._model = None
        self._scaler = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LassoSelector":
        """
        Fit the LASSO selector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (numeric features only)
        y : np.ndarray
            Target variable (required)

        Returns
        -------
        self : LassoSelector
            Fitted selector

        Raises
        ------
        FeatureSelectionError
            If y is not provided or if no numeric features
        """
        if y is None:
            raise FeatureSelectionError(
                "LassoSelector requires a target variable y", method="lasso"
            )

        # Only use numeric columns
        numeric_X = X.select_dtypes(include=[np.number])
        all_features = list(X.columns)
        numeric_features = list(numeric_X.columns)

        if len(numeric_features) == 0:
            raise FeatureSelectionError(
                "No numeric features found for LASSO selection", method="lasso"
            )

        # Prepare data
        X_array = numeric_X.values.astype(float)

        # Impute missing values with column means (preserves distribution better than 0)
        col_means = np.nanmean(X_array, axis=0)
        nan_mask = np.isnan(X_array)
        if nan_mask.any():
            X_array[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_array)

        # Determine if target is discrete or continuous
        unique_values = np.unique(y[~np.isnan(y)] if np.any(np.isnan(y)) else y)
        is_discrete = len(unique_values) <= self.discrete_threshold

        # Fit LASSO model
        if is_discrete:
            # Use LogisticRegression with L1 penalty
            if self.alpha is not None:
                Cs = [1.0 / self.alpha]
            else:
                Cs = 10  # Number of alphas to try

            self._model = LogisticRegressionCV(
                Cs=Cs,
                penalty="l1",
                solver="saga",
                cv=self.cv,
                random_state=self.random_state,
                max_iter=1000,
            )

            try:
                self._model.fit(X_scaled, y)
                self.coefficients_ = np.abs(self._model.coef_).mean(axis=0)
                self.alpha_ = 1.0 / self._model.C_[0]
            except Exception as e:
                raise FeatureSelectionError(f"LASSO fitting failed: {e}", method="lasso")
        else:
            # Use LassoCV for regression
            if self.alpha is not None:
                alphas = [self.alpha]
            else:
                alphas = None  # Let LassoCV choose

            self._model = LassoCV(
                alphas=alphas, cv=self.cv, random_state=self.random_state, max_iter=1000
            )

            try:
                self._model.fit(X_scaled, y)
                self.coefficients_ = np.abs(self._model.coef_)
                self.alpha_ = self._model.alpha_
            except Exception as e:
                raise FeatureSelectionError(f"LASSO fitting failed: {e}", method="lasso")

        # Select features with non-zero coefficients
        if self.n_features is not None:
            # Select top k features by coefficient magnitude
            k = min(self.n_features, len(numeric_features))
            top_indices = np.argsort(self.coefficients_)[-k:]
            keep_mask_numeric = np.zeros(len(numeric_features), dtype=bool)
            keep_mask_numeric[top_indices] = True
        else:
            # Select all non-zero
            keep_mask_numeric = self.coefficients_ > 1e-10

        # Build mask for all features
        keep_mask = np.ones(len(all_features), dtype=bool)
        scores = {}

        for i, feat in enumerate(all_features):
            if feat in numeric_features:
                num_idx = numeric_features.index(feat)
                keep_mask[i] = keep_mask_numeric[num_idx]
                scores[feat] = float(self.coefficients_[num_idx])
            else:
                # Non-numeric features kept by default
                scores[feat] = 1.0

        self._finalize_selection(all_features, keep_mask, scores)
        return self
