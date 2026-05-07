"""
PhenoCluster Temporal Splitter
==============================

Chronological train/test partitioning based on a date column. Used for
temporal generalizability analyses, where phenotypes derived on early data
are validated on later data.
"""

from typing import TYPE_CHECKING, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from ...core.exceptions import DataSplitError, InsufficientDataError
from ...core.types import DataSplitResult
from ._base import BaseSplitter

if TYPE_CHECKING:
    from ...config import DataSplitConfig


class TemporalSplitter(BaseSplitter):
    """
    Temporal split on a date/datetime column.

    Sub-strategies (selected via ``config.time_scheme``):

    - ``"cutoff"``: rows with ``time_column <= time_cutoff`` form the
      derivation set; later rows form the validation set.
    - ``"fraction"``: rows are sorted chronologically; the most recent
      ``time_test_fraction`` form the validation set.
    - ``"sliding"``: ``n_windows`` non-overlapping (or minimally overlapping)
      contiguous validation windows, each preceded by a fixed-length
      derivation window of equal extent.
    - ``"expanding"``: ``n_windows`` validation windows; each derivation set
      is everything from the start of the data up to that window.

    Parameters
    ----------
    config : DataSplitConfig
        Configuration with ``strategy="temporal"`` and the relevant temporal
        fields populated.
    """

    def __init__(self, config: "DataSplitConfig"):
        super().__init__(config)
        if not config.time_column:
            raise ValueError("TemporalSplitter requires config.time_column")
        self._cutoff: Optional[pd.Timestamp] = None
        if config.time_cutoff is not None:
            try:
                self._cutoff = pd.to_datetime(config.time_cutoff, errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Could not parse time_cutoff='{config.time_cutoff}' as a datetime"
                ) from exc

    def _coerce_times(self, df: pd.DataFrame) -> pd.Series:
        col = self.config.time_column
        self._require_columns(df, [col])
        times = pd.to_datetime(df[col], errors="raise")
        if times.isna().any():
            raise DataSplitError(
                f"Time column '{col}' contains missing values; "
                "filter or impute these before splitting."
            )
        return times

    def split(self, df: pd.DataFrame, **_kwargs) -> DataSplitResult:
        """Return the first (or only) temporal split."""
        scheme = self.config.time_scheme
        if scheme == "cutoff":
            return self._cutoff_split(df)
        if scheme == "fraction":
            return self._fraction_split(df)
        if scheme in ("sliding", "expanding"):
            return next(self._iter_window_splits(df))
        raise ValueError(f"Unknown time_scheme '{scheme}'")

    def iter_splits(self, df: pd.DataFrame) -> Iterator[DataSplitResult]:
        """Yield every temporal split produced by ``config.time_scheme``.

        For ``"sliding"`` and ``"expanding"`` this yields one split per
        rolling window. For ``"cutoff"`` and ``"fraction"`` it yields the
        single result of :meth:`split`.
        """
        scheme = self.config.time_scheme
        if scheme in ("sliding", "expanding"):
            yield from self._iter_window_splits(df)
        else:
            yield self.split(df)

    def _cutoff_split(self, df: pd.DataFrame) -> DataSplitResult:
        times = self._coerce_times(df)
        cutoff = self._cutoff
        train_mask = times <= cutoff
        test_mask = ~train_mask
        return self._build_result(
            df,
            np.where(train_mask)[0],
            np.where(test_mask)[0],
            label=f"cutoff={cutoff.date()}",
            deriv_window=f"<={cutoff.date()}",
            valid_window=f">{cutoff.date()}",
        )

    def _fraction_split(self, df: pd.DataFrame) -> DataSplitResult:
        if self.config.time_test_fraction is None:
            raise DataSplitError(
                "TemporalSplitter scheme='fraction' requires time_test_fraction; got None."
            )
        times = self._coerce_times(df)
        order = np.argsort(times.values, kind="stable")
        n = len(df)
        n_test = max(1, int(round(n * self.config.time_test_fraction)))
        test_pos = order[-n_test:]
        train_pos = order[:-n_test]
        train_idx, test_idx = self._snap_boundary(times, train_pos, test_pos)
        deriv_max = times.iloc[train_idx].max().date() if len(train_idx) else None
        valid_min = times.iloc[test_idx].min().date()
        valid_max = times.iloc[test_idx].max().date()
        return self._build_result(
            df,
            train_idx,
            test_idx,
            label=f"fraction={self.config.time_test_fraction:.2f}",
            deriv_window=f"<={deriv_max}" if deriv_max else None,
            valid_window=f"{valid_min}..{valid_max}",
        )

    @staticmethod
    def _snap_boundary(
        times: pd.Series, train_pos: np.ndarray, test_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Move every training-set row whose timestamp matches the smallest
        validation timestamp into the validation set. Ensures no single date
        appears on both sides of the split.
        """
        if len(test_pos) == 0 or len(train_pos) == 0:
            return np.sort(train_pos), np.sort(test_pos)
        boundary = times.iloc[test_pos].min()
        train_times = times.iloc[train_pos].values
        keep_mask = train_times != boundary
        if keep_mask.all():
            return np.sort(train_pos), np.sort(test_pos)
        moved = train_pos[~keep_mask]
        new_train = train_pos[keep_mask]
        new_test = np.concatenate([test_pos, moved])
        return np.sort(new_train), np.sort(np.unique(new_test))

    def _iter_window_splits(self, df: pd.DataFrame) -> Iterator[DataSplitResult]:
        if self.config.n_windows is None:
            raise DataSplitError(
                f"TemporalSplitter scheme='{self.config.time_scheme}' requires n_windows; got None."
            )
        times = self._coerce_times(df)
        order = np.argsort(times.values, kind="stable")
        n = len(df)
        k = self.config.n_windows
        scheme = self.config.time_scheme

        window_size = n // (k + 1)
        if window_size < self.config.min_validation_size:
            raise InsufficientDataError(
                f"Cannot build {k} {scheme} windows from {n} samples "
                f"(window_size={window_size} < "
                f"min_validation_size={self.config.min_validation_size})",
                n_samples=n,
                min_required=self.config.min_validation_size * (k + 1),
            )

        for i in range(k):
            test_start = (i + 1) * window_size
            test_end = test_start + window_size if i < k - 1 else n
            test_pos = order[test_start:test_end]
            if scheme == "sliding":
                train_start = max(0, test_start - window_size)
                train_pos = order[train_start:test_start]
            else:
                train_pos = order[:test_start]
            train_idx, test_idx = self._snap_boundary(times, train_pos, test_pos)
            valid_min = times.iloc[test_idx].min().date()
            valid_max = times.iloc[test_idx].max().date()
            deriv_min = times.iloc[train_idx].min().date()
            deriv_max = times.iloc[train_idx].max().date()
            yield self._build_result(
                df,
                train_idx,
                test_idx,
                label=f"{scheme}_window_{i + 1}_of_{k}",
                deriv_window=f"{deriv_min}..{deriv_max}",
                valid_window=f"{valid_min}..{valid_max}",
            )

    def _build_result(
        self,
        df: pd.DataFrame,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        label: str,
        deriv_window: Optional[str],
        valid_window: Optional[str],
    ) -> DataSplitResult:
        if len(test_idx) < self.config.min_validation_size:
            raise InsufficientDataError(
                f"Temporal validation cohort '{label}' has {len(test_idx)} rows "
                f"(< min_validation_size={self.config.min_validation_size})",
                n_samples=len(test_idx),
                min_required=self.config.min_validation_size,
            )
        if len(train_idx) == 0:
            raise InsufficientDataError(
                f"Temporal derivation cohort '{label}' is empty",
                n_samples=0,
                min_required=1,
            )
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        return DataSplitResult(
            train=train_df,
            test=test_df,
            train_indices=train_idx,
            test_indices=test_idx,
            partition_kind="temporal",
            partition_label=label,
            derivation_window=deriv_window,
            validation_window=valid_window,
        )
