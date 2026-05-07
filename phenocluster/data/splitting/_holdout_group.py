"""
PhenoCluster Holdout-Group Splitter
===================================

Holds out one or more named groups (e.g., a specific site/center) from the
derivation set. The remaining groups form derivation; the held-out groups
form the validation cohort.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from ...core.exceptions import DataSplitError, InsufficientDataError
from ...core.types import DataSplitResult
from ._base import BaseSplitter

if TYPE_CHECKING:
    from ...config import DataSplitConfig


class HoldoutGroupSplitter(BaseSplitter):
    """
    Group-aware split that holds out specified group values.

    Parameters
    ----------
    config : DataSplitConfig
        Configuration with ``strategy="holdout_group"``, ``group_column`` and
        ``holdout_values`` populated.

    Notes
    -----
    Rows whose ``group_column`` is missing (NaN) are excluded from the
    derivation set and reported in the result's
    :attr:`stratification_fallback_reason` field for transparency.
    """

    def __init__(self, config: "DataSplitConfig"):
        super().__init__(config)
        if not config.group_column:
            raise ValueError("HoldoutGroupSplitter requires config.group_column")
        if not config.holdout_values:
            raise ValueError("HoldoutGroupSplitter requires non-empty config.holdout_values")
        self._holdout_set = set(config.holdout_values)

    def split(self, df: pd.DataFrame, **_kwargs) -> DataSplitResult:
        """Hold out the configured ``holdout_values`` from ``group_column``.

        Returns
        -------
        DataSplitResult
            Derivation/validation partitions with ``partition_kind="site"``.
            Rows whose ``group_column`` is missing are excluded; the count is
            recorded on ``stratification_fallback_reason``.
        """
        col = self.config.group_column
        self._require_columns(df, [col])
        values = df[col]
        missing_mask = values.isna()
        n_missing = int(missing_mask.sum())

        test_mask = values.isin(self._holdout_set) & ~missing_mask
        train_mask = ~values.isin(self._holdout_set) & ~missing_mask

        train_idx = np.where(train_mask.values)[0]
        test_idx = np.where(test_mask.values)[0]

        if len(test_idx) < self.config.min_validation_size:
            raise InsufficientDataError(
                f"Holdout cohort for {col} in {sorted(self._holdout_set)} "
                f"has {len(test_idx)} rows (< min_validation_size="
                f"{self.config.min_validation_size})",
                n_samples=len(test_idx),
                min_required=self.config.min_validation_size,
            )
        if len(train_idx) == 0:
            raise DataSplitError(
                f"Derivation set is empty after holding out {sorted(self._holdout_set)}"
            )

        fallback_reason: Optional[str] = None
        if n_missing > 0:
            fallback_reason = f"{n_missing} row(s) had missing '{col}' and were excluded."
            self.logger.warning(fallback_reason)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        label = f"{col}=" + ",".join(str(v) for v in sorted(self._holdout_set, key=str))
        return DataSplitResult(
            train=train_df,
            test=test_df,
            train_indices=train_idx,
            test_indices=test_idx,
            stratification_fallback_reason=fallback_reason,
            partition_kind="site",
            partition_label=label,
        )
