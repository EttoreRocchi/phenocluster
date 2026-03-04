"""
PhenoCluster Preprocessing Orchestrator
=========================================

Thin orchestrator composing Imputer, OutlierHandler, Encoder, and Scaler.

All preprocessing follows a fit/transform pattern to prevent data leakage:
- ``fit_*()`` methods learn parameters from training data only.
- ``transform_*()`` methods apply the learned parameters to any data.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger
from .encoder import Encoder
from .imputer import Imputer
from .outlier_handler import OutlierHandler
from .scaler import Scaler


class DataPreprocessor:
    """Orchestrates imputation, outlier handling, encoding, and scaling."""

    def __init__(self, config: PhenoClusterConfig):
        self.config = config
        self.logger = get_logger("preprocessing", config)

        self._imputer = Imputer(config)
        self._outlier_handler = OutlierHandler(config)
        self._encoder = Encoder(config)
        self._scaler = Scaler(config)

    @property
    def scaler(self):
        return self._scaler.scaler

    @property
    def label_encoders(self):
        return self._encoder.label_encoders

    @property
    def onehot_encoder(self):
        return self._encoder.onehot_encoder

    @property
    def frequency_encodings(self):
        return self._encoder.frequency_encodings

    @property
    def feature_columns(self):
        return self._encoder.feature_columns

    @property
    def outlier_detector(self):
        return self._outlier_handler.outlier_detector

    @property
    def outlier_mask(self):
        return self._outlier_handler.outlier_mask

    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect and report missing values in the dataset."""
        feature_cols = self.config.continuous_columns + self.config.categorical_columns
        missing_info = {}

        self.logger.info("MISSING VALUE ANALYSIS")
        for col in feature_cols:
            if col in df.columns:
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                missing_info[col] = missing_pct
                if missing_pct > 0:
                    self.logger.info(f"  {col}: {missing_pct:.2f}% missing")

        if missing_info:
            total_missing = sum(missing_info.values()) / len(missing_info)
            self.logger.info(f"Average missing rate: {total_missing:.2f}%")
        else:
            self.logger.info("No missing values detected")

        return missing_info

    def fit_imputer(self, df: pd.DataFrame) -> None:
        """Fit imputation parameters on training data."""
        self._imputer.fit(df)

    def transform_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation to a dataframe."""
        return self._imputer.transform(df)

    def fit_outlier_handler(self, df: pd.DataFrame) -> None:
        """Fit outlier detection parameters on training data."""
        self._outlier_handler.fit(df)

    def transform_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted outlier handling to a dataframe."""
        return self._outlier_handler.transform(df)

    def fit_preprocessor(self, df: pd.DataFrame) -> None:
        """Fit encoding and scaling on training data."""
        self._encoder.fit(df)
        self._scaler.fit(df)
        self.logger.info("Preprocessor fitted successfully")

    def transform_preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply fitted encoding and scaling, return processed df and feature matrix."""
        df_encoded = self._encoder.transform(df)
        df_scaled = self._scaler.transform(df_encoded)
        X = df_scaled[self._encoder.feature_columns].values
        self.logger.info(f"Feature matrix shape: {X.shape}")
        return df_scaled, X

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
    ) -> np.ndarray:
        """Extract feature matrix using fitted encoders."""
        return self._encoder.get_feature_matrix(df, continuous_cols, categorical_cols)
