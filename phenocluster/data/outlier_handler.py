"""Outlier detection and handling with fit/transform pattern."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class OutlierHandler:
    """Fit outlier detection parameters on training data and apply to any split.

    Note: For isolation_forest, the ``transform()`` method identifies outliers and
    stores the mask in ``self.outlier_mask`` but does NOT remove them from the
    returned DataFrame. The pipeline handles outlier removal separately.
    """

    def __init__(self, config: PhenoClusterConfig):
        self.config = config
        self.logger = get_logger("outlier", config)

        self.outlier_detector: Optional[IsolationForest] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self._winsorize_bounds: Dict[str, Tuple[float, float]] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit outlier detection parameters on training data."""
        if not self.config.outlier.enabled or not self.config.continuous_columns:
            self._is_fitted = True
            return

        self.logger.info(f"OUTLIER DETECTION (fit) - Method: {self.config.outlier.method.upper()}")
        cont_cols = [c for c in self.config.continuous_columns if c in df.columns]

        if self.config.outlier.method == "isolation_forest":
            self._fit_isolation_forest(df, cont_cols)
        elif self.config.outlier.method == "winsorize":
            self._fit_winsorize(df, cont_cols)

        self._is_fitted = True
        self.logger.info("Outlier handler fitted successfully")

    def _fit_isolation_forest(self, df, cont_cols):
        self.logger.info(f"  contamination={self.config.outlier.contamination}")
        self.outlier_detector = IsolationForest(
            contamination=self.config.outlier.contamination,
            random_state=self.config.random_state,
            n_jobs=1,
        )
        self.outlier_detector.fit(df[cont_cols].values)

    def _fit_winsorize(self, df, cont_cols):
        lower, upper = self.config.outlier.winsorize_limits
        self.logger.info(f"  Winsorization limits=({lower}, {upper})")
        for col in cont_cols:
            vals = df[col].dropna().values
            lower_val = np.nanpercentile(vals, lower * 100)
            upper_val = np.nanpercentile(vals, (1 - upper) * 100)
            self._winsorize_bounds[col] = (lower_val, upper_val)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted outlier handling to a dataframe."""
        if not self.config.outlier.enabled or not self.config.continuous_columns:
            return df
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before transform()")

        df_result = df.copy()
        cont_cols = [c for c in self.config.continuous_columns if c in df.columns]

        if self.config.outlier.method == "isolation_forest":
            if self.outlier_detector is None:
                raise RuntimeError("IsolationForest model not fitted. Call fit() first.")
            predictions = self.outlier_detector.predict(df[cont_cols].values)
            self.outlier_mask = predictions == -1
            n_outliers = self.outlier_mask.sum()
            self.logger.info(f"Detected {n_outliers} outliers ({n_outliers / len(df) * 100:.2f}%)")

        elif self.config.outlier.method == "winsorize":
            n_winsorized = 0
            for col in cont_cols:
                if col in self._winsorize_bounds:
                    lo, hi = self._winsorize_bounds[col]
                    original = df_result[col].values.copy()
                    df_result[col] = df_result[col].clip(lower=lo, upper=hi)
                    n_winsorized += int(np.sum(original != df_result[col].values))
            self.logger.info(f"Total values winsorized: {n_winsorized}")

        return df_result
