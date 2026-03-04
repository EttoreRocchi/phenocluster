"""
PhenoCluster Mixed Data Feature Selector
=========================================

Combined feature selection for mixed continuous/categorical data.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import FeatureSelectionError
from .base import BaseFeatureSelector
from .correlation import CorrelationSelector
from .lasso import LassoSelector
from .mutual_info import MutualInfoSelector
from .variance import VarianceSelector

if TYPE_CHECKING:
    from ..config import FeatureSelectionConfig


class MixedDataFeatureSelector(BaseFeatureSelector):
    """
    Feature selection for mixed continuous and categorical data.

    Combines multiple selection strategies and applies them sequentially
    or individually based on configuration.

    Parameters
    ----------
    config : FeatureSelectionConfig
        Configuration for feature selection
    continuous_cols : List[str], optional
        Names of continuous columns
    categorical_cols : List[str], optional
        Names of categorical columns

    Examples
    --------
    >>> config = FeatureSelectionConfig(method='combined')
    >>> selector = MixedDataFeatureSelector(config)
    >>> X_selected = selector.fit_transform(X)
    """

    def __init__(
        self,
        config: "FeatureSelectionConfig",
        continuous_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ):
        super().__init__()
        self.config = config
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self._selectors: List[BaseFeatureSelector] = []
        self._build_selector_pipeline()

    def _build_selector_pipeline(self):
        """Build the selector pipeline based on config."""
        cfg = self.config
        _SELECTOR_FACTORIES = {
            "variance": lambda: [
                VarianceSelector(
                    variance_threshold=cfg.variance_threshold,
                    frequency_threshold=cfg.frequency_threshold,
                )
            ],
            "correlation": lambda: [CorrelationSelector(threshold=cfg.correlation_threshold)],
            "mutual_info": lambda: [
                MutualInfoSelector(
                    n_features=cfg.n_features,
                    percentile=cfg.percentile,
                )
            ],
            "lasso": lambda: [LassoSelector(alpha=cfg.lasso_alpha, n_features=cfg.n_features)],
            "combined": lambda: [
                VarianceSelector(
                    variance_threshold=cfg.variance_threshold,
                    frequency_threshold=cfg.frequency_threshold,
                ),
                CorrelationSelector(threshold=cfg.correlation_threshold),
            ],
        }
        factory = _SELECTOR_FACTORIES.get(cfg.method)
        self._selectors = factory() if factory else []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> "MixedDataFeatureSelector":
        """
        Fit the mixed data feature selector.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Target variable (required for some methods)

        Returns
        -------
        self : MixedDataFeatureSelector
            Fitted selector
        """
        if not self.config.enabled:
            # Keep all features
            self.selected_features_ = list(X.columns)
            self.removed_features_ = []
            self.feature_scores_ = {f: 1.0 for f in X.columns}
            self._is_fitted = True
            return self

        # Check if target is required
        if self.config.require_target and y is None:
            raise FeatureSelectionError(
                f"Method '{self.config.method}' requires a target variable y",
                method=self.config.method,
            )

        # Apply selectors sequentially
        current_X = X.copy()
        all_scores = {}

        for selector in self._selectors:
            if self.config.method == "variance" and isinstance(selector, VarianceSelector):
                selector.fit(
                    current_X,
                    y,
                    continuous_cols=self.continuous_cols,
                    categorical_cols=self.categorical_cols,
                )
            else:
                selector.fit(current_X, y)

            # Update scores
            for feat, score in selector.get_feature_scores().items():
                if feat not in all_scores:
                    all_scores[feat] = score
                else:
                    all_scores[feat] = min(all_scores[feat], score)

            # Filter to selected features for next selector
            current_X = selector.transform(current_X)

        # Final selection
        all_features = list(X.columns)
        self.selected_features_ = list(current_X.columns)
        self.removed_features_ = [f for f in all_features if f not in self.selected_features_]
        self.feature_scores_ = all_scores
        self._is_fitted = True

        return self

    def get_selection_report(self) -> Dict:
        """
        Get a detailed report of the selection process.

        Returns
        -------
        Dict
            Report with selection details
        """
        if not self._is_fitted:
            from ..core.exceptions import ModelNotFittedError

            raise ModelNotFittedError("MixedDataFeatureSelector")

        return {
            "method": self.config.method,
            "n_original": len(self.selected_features_) + len(self.removed_features_),
            "n_selected": len(self.selected_features_),
            "n_removed": len(self.removed_features_),
            "selection_ratio": len(self.selected_features_)
            / (len(self.selected_features_) + len(self.removed_features_)),
            "selected_features": self.selected_features_,
            "removed_features": self.removed_features_,
            "feature_scores": self.feature_scores_,
        }
