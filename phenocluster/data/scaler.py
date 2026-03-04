"""Continuous variable standardization with fit/transform pattern."""

from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class Scaler:
    """Fit standardization on training data and apply to any split."""

    def __init__(self, config: PhenoClusterConfig):
        self.config = config
        self.logger = get_logger("scaler", config)
        self.scaler: Optional[StandardScaler] = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit StandardScaler on continuous columns of training data."""
        if not self.config.continuous_columns:
            self._is_fitted = True
            return

        cont_cols = [c for c in self.config.continuous_columns if c in df.columns]
        self.logger.info(f"SCALING (fit) -{len(cont_cols)} continuous variables")
        self.scaler = StandardScaler()
        self.scaler.fit(df[cont_cols])
        self._is_fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted StandardScaler to continuous columns."""
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before transform()")
        if not self.config.continuous_columns or self.scaler is None:
            return df

        df_result = df.copy()
        cont_cols = [c for c in self.config.continuous_columns if c in df_result.columns]
        df_result[cont_cols] = self.scaler.transform(df_result[cont_cols])
        return df_result
