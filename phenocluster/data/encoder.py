"""Categorical encoding with fit/transform pattern."""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class Encoder:
    """Fit categorical encoding on training data and apply to any split."""

    def __init__(self, config: PhenoClusterConfig):
        self.config = config
        self.logger = get_logger("encoding", config)

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.frequency_encodings: Dict[str, dict] = {}
        self.feature_columns: List[str] = []
        self._category_modes: Dict[str, Optional[str]] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit encoding parameters on training data."""
        df_processed = self._strip_categoricals(df)
        self.logger.info("ENCODING (fit)")

        if not self.config.categorical_columns:
            self.feature_columns = self.config.continuous_columns.copy()
            self._is_fitted = True
            return

        method = self.config.categorical_encoding.method
        self.logger.info(f"  Method: {method.upper()}")

        if method == "label":
            self._fit_label(df_processed)
        elif method == "onehot":
            self._fit_onehot(df_processed)
        elif method == "frequency":
            self._fit_frequency(df_processed)

        self._is_fitted = True

    def _fit_label(self, df):
        self._category_modes = {}
        for col in self.config.categorical_columns:
            if col in df.columns:
                nan_mask = df[col].isna()
                non_null = df.loc[~nan_mask, col].astype(str)
                self._category_modes[col] = non_null.mode().iloc[0] if len(non_null) > 0 else None
                le = LabelEncoder()
                le.fit(non_null)
                self.label_encoders[col] = le

        self.feature_columns = self.config.continuous_columns.copy()
        self.feature_columns += [f"{col}_encoded" for col in self.config.categorical_columns]

    def _fit_onehot(self, df):
        self.onehot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown=self.config.categorical_encoding.handle_unknown,
        )
        cat_data = df[self.config.categorical_columns].fillna("_MISSING").astype(str)
        self.onehot_encoder.fit(cat_data)

        encoded_cols = []
        for i, col in enumerate(self.config.categorical_columns):
            for cat in self.onehot_encoder.categories_[i]:
                encoded_cols.append(f"{col}_{cat}")

        self.feature_columns = self.config.continuous_columns.copy()
        self.feature_columns += encoded_cols

    def _fit_frequency(self, df):
        for col in self.config.categorical_columns:
            if col in df.columns:
                self.frequency_encodings[col] = df[col].value_counts(normalize=True).to_dict()

        self.feature_columns = self.config.continuous_columns.copy()
        self.feature_columns += [f"{col}_encoded" for col in self.config.categorical_columns]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoding to a dataframe."""
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before transform()")

        df_processed = self._strip_categoricals(df)

        if not self.config.categorical_columns:
            return df_processed

        method = self.config.categorical_encoding.method

        if method == "label":
            df_processed = self._transform_label(df_processed)
        elif method == "onehot" and self.onehot_encoder is not None:
            df_processed = self._transform_onehot(df_processed)
        elif method == "frequency":
            df_processed = self._transform_frequency(df_processed)

        return df_processed

    def _transform_label(self, df):
        for col in self.config.categorical_columns:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                nan_mask = df[col].isna()
                non_null = df.loc[~nan_mask, col].astype(str)

                if len(non_null) > 0:
                    known_classes = set(le.classes_)
                    unknown_mask = ~non_null.isin(known_classes)
                    if unknown_mask.any():
                        mode = self._category_modes.get(col)
                        self.logger.warning(
                            f"  {col}: {int(unknown_mask.sum())} values mapped to "
                            f"mode '{mode}' (unknown categories)"
                        )
                        non_null = non_null.copy()
                        non_null.loc[unknown_mask] = mode

                encoded = pd.Series(np.nan, index=df.index, dtype=float)
                if (~nan_mask).any():
                    encoded.loc[~nan_mask] = le.transform(non_null).astype(float)
                df[f"{col}_encoded"] = encoded
        return df

    def _transform_onehot(self, df):
        if self.onehot_encoder is None:
            raise RuntimeError("OneHotEncoder not fitted. Call fit() first.")
        cat_data = df[self.config.categorical_columns].fillna("_MISSING").astype(str)
        encoded_array = self.onehot_encoder.transform(cat_data)

        encoded_cols = []
        for i, col in enumerate(self.config.categorical_columns):
            for cat in self.onehot_encoder.categories_[i]:
                encoded_cols.append(f"{col}_{cat}")

        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        return pd.concat([df, encoded_df], axis=1)

    def _transform_frequency(self, df):
        for col in self.config.categorical_columns:
            if col in df.columns and col in self.frequency_encodings:
                mapped = df[col].map(self.frequency_encodings[col])
                n_unknown = mapped.isna().sum() - df[col].isna().sum()
                if n_unknown > 0:
                    warnings.warn(
                        f"Column '{col}': {n_unknown} unknown categories mapped to frequency 0.0",
                        stacklevel=2,
                    )
                df[f"{col}_encoded"] = mapped.fillna(0.0)
        return df

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        continuous_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Extract feature matrix for specified columns using fitted encoders."""
        if continuous_cols is None:
            continuous_cols = self.config.continuous_columns
        if categorical_cols is None:
            categorical_cols = self.config.categorical_columns

        feature_cols = continuous_cols.copy()
        method = self.config.categorical_encoding.method

        if method == "label":
            feature_cols += [f"{col}_encoded" for col in categorical_cols]
        elif method == "onehot" and self.onehot_encoder is not None:
            for i, col in enumerate(self.config.categorical_columns):
                if col in categorical_cols:
                    for cat in self.onehot_encoder.categories_[i]:
                        feature_cols.append(f"{col}_{cat}")
        elif method == "frequency":
            feature_cols += [f"{col}_encoded" for col in categorical_cols]

        available = [c for c in feature_cols if c in df.columns]
        return df[available].values

    def _strip_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        for col in self.config.categorical_columns:
            if col in df_out.columns and df_out[col].dtype == object:
                df_out[col] = df_out[col].str.strip().str.strip("'\"")
        return df_out
