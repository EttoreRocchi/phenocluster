"""
PhenoCluster Data Splitter
==========================

Train/test data splitting with stratification support.

Model selection fits each candidate on the full training set and scores
by information criterion, so a separate validation set is not needed.
The test set provides unbiased evaluation of the final model.
"""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..core.exceptions import DataSplitError, InsufficientDataError
from ..core.types import DataSplitResult

if TYPE_CHECKING:
    from ..config import DataSplitConfig


class DataSplitter:
    """
    Splits data into train and test sets.

    Supports stratified splitting and reproducible experiments.
    Model selection is handled separately on the training set.

    Parameters
    ----------
    config : DataSplitConfig
        Configuration for splitting

    Examples
    --------
    >>> config = DataSplitConfig(test_size=0.2)
    >>> splitter = DataSplitter(config)
    >>> result = splitter.split(df)
    >>> print(f"Train: {result.n_train}, Test: {result.n_test}")
    """

    def __init__(self, config: "DataSplitConfig"):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._stratification_fallback = False

    def split(self, df: pd.DataFrame, stratify_column: Optional[str] = None) -> DataSplitResult:
        """
        Split dataframe into train and test sets.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to split
        stratify_column : str, optional
            Column for stratification (overrides config)

        Returns
        -------
        DataSplitResult
            Container with train and test dataframes
        """
        n_samples = len(df)
        min_samples = 10  # Minimum samples needed

        if n_samples < min_samples:
            raise InsufficientDataError(
                f"Need at least {min_samples} samples for splitting",
                n_samples=n_samples,
                min_required=min_samples,
            )

        strat_col = stratify_column or self.config.stratify_by
        stratify = None
        stratification_used = False
        fallback_reason: Optional[str] = None

        if strat_col:
            if strat_col not in df.columns:
                raise DataSplitError(f"Stratification column '{strat_col}' not found in dataframe")
            stratify = df[strat_col].values

            # Check if stratification is possible
            unique, counts = np.unique(stratify, return_counts=True)
            min_count = int(np.min(counts))
            if min_count < 2:
                fallback_reason = (
                    f"Stratification column '{strat_col}' has a stratum with "
                    f"{min_count} sample(s); sklearn requires >=2."
                )
                self.logger.warning(f"{fallback_reason} Falling back to non-stratified split.")
                self._stratification_fallback = True
                stratify = None
            else:
                stratification_used = True

        # Split into train and test
        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle,
            stratify=stratify,
        )

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return DataSplitResult(
            train=train_df,
            test=test_df,
            train_indices=train_idx,
            test_indices=test_idx,
            stratification_used=stratification_used,
            stratification_fallback_reason=fallback_reason,
        )
