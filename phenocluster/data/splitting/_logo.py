"""
PhenoCluster Leave-One-Group-Out Splitter
=========================================

Yields one (derivation, validation) pair per unique value of
``config.group_column``. Used for multi-site generalizability assessments.
"""

from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd

from ...core.exceptions import DataSplitError
from ...core.types import DataSplitResult
from ._base import BaseSplitter

if TYPE_CHECKING:
    from ...config import DataSplitConfig


class LeaveOneGroupOutSplitter(BaseSplitter):
    """
    Iterates leave-one-group-out splits.

    Each split holds out a single group value (e.g., one hospital) as the
    validation cohort and uses the remaining groups as the derivation set.

    Parameters
    ----------
    config : DataSplitConfig
        Configuration with ``strategy="leave_one_group_out"`` and
        ``group_column`` populated.
    """

    def __init__(self, config: "DataSplitConfig"):
        super().__init__(config)
        if not config.group_column:
            raise ValueError("LeaveOneGroupOutSplitter requires config.group_column")

    # `split()` is inherited from BaseSplitter and returns the first cohort
    # produced by `iter_splits` (the alphabetically-first group value,
    # respecting `min_validation_size`). Callers that want every cohort
    # should iterate via ``iter_splits``.

    def iter_splits(self, df: pd.DataFrame) -> Iterator[DataSplitResult]:
        """Yield one split per unique ``group_column`` value (sorted by string).

        Splits whose validation cohort is smaller than
        ``min_validation_size`` or whose derivation set is empty are skipped
        with a warning. Rows with missing ``group_column`` are excluded from
        every split.
        """
        col = self.config.group_column
        self._require_columns(df, [col])
        values = df[col]
        present = values.dropna().unique()
        if len(present) < 2:
            raise DataSplitError(
                f"Leave-one-group-out requires at least 2 groups in '{col}', found {len(present)}"
            )

        n_missing = int(values.isna().sum())
        if n_missing > 0:
            self.logger.warning(
                f"{n_missing} row(s) had missing '{col}' and will be excluded "
                "from every LOGO split."
            )

        ordered_groups = sorted(present, key=str)
        for group_value in ordered_groups:
            test_mask = (values == group_value).values
            train_mask = (values != group_value).values & ~values.isna().values
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(test_idx) < self.config.min_validation_size:
                self.logger.warning(
                    f"Skipping LOGO split for {col}={group_value!r}: "
                    f"{len(test_idx)} rows < min_validation_size="
                    f"{self.config.min_validation_size}"
                )
                continue
            if len(train_idx) == 0:
                self.logger.warning(
                    f"Skipping LOGO split for {col}={group_value!r}: derivation set empty"
                )
                continue

            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            yield DataSplitResult(
                train=train_df,
                test=test_df,
                train_indices=train_idx,
                test_indices=test_idx,
                partition_kind="site",
                partition_label=f"{col}={group_value}",
            )
