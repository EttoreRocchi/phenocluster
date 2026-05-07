"""
PhenoCluster Random Splitter
============================

Random or stratified train/test split. Reproduces the original v0.2.0
``DataSplitter`` behavior; remains the default for back-compatibility.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ...core.exceptions import DataSplitError, InsufficientDataError
from ...core.types import DataSplitResult
from ._base import BaseSplitter


class RandomSplitter(BaseSplitter):
    """
    Random split with optional stratification.

    Parameters
    ----------
    config : DataSplitConfig
        Configuration; ``test_size``, ``stratify_by``, ``shuffle`` and
        ``random_state`` are honored.

    Examples
    --------
    >>> splitter = RandomSplitter(config)
    >>> result = splitter.split(df)
    >>> print(f"Train: {result.n_train}, Test: {result.n_test}")
    """

    def split(
        self, df: pd.DataFrame, stratify_column: Optional[str] = None, **_kwargs
    ) -> DataSplitResult:
        """
        Split dataframe into train and test sets.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to split.
        stratify_column : str, optional
            Column for stratification (overrides ``config.stratify_by``).

        Returns
        -------
        DataSplitResult
            Train/test partitions with ``partition_kind="random"``.
        """
        n_samples = len(df)
        min_samples = 10

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

            unique, counts = np.unique(stratify, return_counts=True)
            min_count = int(np.min(counts))
            if min_count < 2:
                fallback_reason = (
                    f"Stratification column '{strat_col}' has a stratum with "
                    f"{min_count} sample(s); sklearn requires >=2."
                )
                self.logger.warning(f"{fallback_reason} Falling back to non-stratified split.")
                stratify = None
            else:
                stratification_used = True

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
            partition_kind="random",
        )
