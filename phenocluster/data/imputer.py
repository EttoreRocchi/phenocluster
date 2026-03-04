"""Imputation for missing values with fit/transform pattern."""

from typing import Dict, Optional

import pandas as pd
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401 - required to enable IterativeImputer
)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class Imputer:
    """Fit imputation statistics on training data and apply to any split.

    Note: This module performs single imputation. Multiple imputation
    (MI) with Rubin's rules for pooling would propagate imputation
    uncertainty into downstream estimates but is not currently supported.
    """

    def __init__(self, config: PhenoClusterConfig):
        self.config = config
        self.logger = get_logger("imputation", config)

        self._imputer_cont = None
        self._imputer_cat = None
        self._knn_temp_encoders: Dict[str, LabelEncoder] = {}
        self._knn_temp_modes: Dict[str, Optional[str]] = {}
        self._imputation_method: Optional[str] = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Fit imputation parameters on training data."""
        feature_cols = self.config.continuous_columns + self.config.categorical_columns
        df_impute = df[feature_cols].copy()

        if not self.config.imputation.enabled or not df_impute.isna().any().any():
            self._is_fitted = True
            return

        method = self.config.imputation.method
        self._imputation_method = method
        self.logger.info(f"IMPUTATION (fit) - Method: {method.upper()}")

        cont_cols = [c for c in self.config.continuous_columns if c in df_impute.columns]
        cat_cols = [c for c in self.config.categorical_columns if c in df_impute.columns]

        if method == "iterative":
            self._fit_iterative(df_impute, cont_cols, cat_cols)
        elif method == "simple":
            self._fit_simple(df_impute, cont_cols, cat_cols)
        elif method == "knn":
            self._fit_knn(df_impute)
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        self._is_fitted = True
        self.logger.info("Imputer fitted successfully")

    def _fit_iterative(self, df, cont_cols, cat_cols):
        estimator_type = self.config.imputation.estimator
        self.logger.info(f"  IterativeImputer max_iter={self.config.imputation.max_iter}")

        if estimator_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            estimator = RandomForestRegressor(
                n_estimators=10,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1,
            )
        else:
            from sklearn.linear_model import BayesianRidge

            estimator = BayesianRidge()

        if cont_cols:
            self._imputer_cont = IterativeImputer(
                estimator=estimator,
                max_iter=self.config.imputation.max_iter,
                random_state=self.config.random_state,
                initial_strategy="mean",
                n_nearest_features=self.config.imputation.n_nearest_features,
                verbose=0,
            )
            self._imputer_cont.fit(df[cont_cols])

        if cat_cols:
            # Limitation: categorical variables are imputed independently
            # via mode (most_frequent) rather than using the iterative
            # multivariate model.  Cross-variable information between
            # categorical and continuous features is therefore not
            # leveraged for categorical imputation.
            self._imputer_cat = SimpleImputer(strategy="most_frequent")
            self._imputer_cat.fit(df[cat_cols])

    def _fit_simple(self, df, cont_cols, cat_cols):
        self.logger.info("  SimpleImputer (mean/most_frequent)")
        if cont_cols:
            self._imputer_cont = SimpleImputer(strategy="mean")
            self._imputer_cont.fit(df[cont_cols])
        if cat_cols:
            self._imputer_cat = SimpleImputer(strategy="most_frequent")
            self._imputer_cat.fit(df[cat_cols])

    def _fit_knn(self, df_impute):
        # Limitation: categorical variables are label-encoded to arbitrary
        # integers before KNN imputation.  Euclidean distance on these
        # integers is not meaningful for nominal variables.  One-hot
        # encoding or Gower distance would be more appropriate but are
        # not currently supported.
        self.logger.info("  KNNImputer k=5")
        df_encoded = df_impute.copy()
        self._knn_temp_encoders = {}
        self._knn_temp_modes = {}

        for col in self.config.categorical_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                non_null_mask = df_encoded[col].notna()
                if non_null_mask.any():
                    vals = df_encoded.loc[non_null_mask, col].astype(str)
                    le.fit(vals)
                    df_encoded.loc[non_null_mask, col] = le.transform(vals)
                    self._knn_temp_encoders[col] = le
                    self._knn_temp_modes[col] = vals.mode().iloc[0] if len(vals) > 0 else None

        self._imputer_cont = KNNImputer(n_neighbors=5, weights="uniform", keep_empty_features=True)
        self._imputer_cont.fit(df_encoded)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputation to a dataframe."""
        feature_cols = self.config.continuous_columns + self.config.categorical_columns

        if not self.config.imputation.enabled:
            return df
        if not self._is_fitted:
            raise RuntimeError("fit() must be called before transform()")

        df_impute = df[feature_cols].copy()
        if not df_impute.isna().any().any():
            return df
        if self._imputer_cont is None and self._imputer_cat is None:
            return df

        method = self._imputation_method
        cont_cols = [c for c in self.config.continuous_columns if c in df_impute.columns]
        cat_cols = [c for c in self.config.categorical_columns if c in df_impute.columns]
        df_result = df.copy()

        if method in ("iterative", "simple"):
            if cont_cols and self._imputer_cont is not None:
                df_result[cont_cols] = self._imputer_cont.transform(df_impute[cont_cols])
            if cat_cols and self._imputer_cat is not None:
                df_result[cat_cols] = self._imputer_cat.transform(df_impute[cat_cols])
        elif method == "knn":
            df_result = self._transform_knn(df_result, df_impute, feature_cols)

        return df_result

    def _transform_knn(self, df_result, df_impute, feature_cols):
        df_encoded = df_impute.copy()
        for col, le in self._knn_temp_encoders.items():
            if col in df_encoded.columns:
                non_null_mask = df_encoded[col].notna()
                if non_null_mask.any():
                    vals = df_encoded.loc[non_null_mask, col].astype(str)
                    known = set(le.classes_)
                    unknown_mask = ~vals.isin(known)
                    if unknown_mask.any():
                        mode = self._knn_temp_modes.get(col)
                        if mode is not None:
                            self.logger.warning(
                                f"  KNN impute: {col}: {unknown_mask.sum()} unknown "
                                f"categories mapped to '{mode}'"
                            )
                            vals = vals.copy()
                            vals.loc[unknown_mask] = mode
                    df_encoded.loc[non_null_mask, col] = le.transform(vals)

        if self._imputer_cont is not None:
            imputed_array = self._imputer_cont.transform(df_encoded)
            df_result[feature_cols] = imputed_array

        return df_result
