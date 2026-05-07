"""
PhenoCluster Splitter Base
==========================

Base class shared by all splitting strategies.
"""

import logging
from abc import ABC
from typing import TYPE_CHECKING, Iterable, Iterator

import pandas as pd

from ...core.exceptions import DataSplitError
from ...core.types import DataSplitResult

if TYPE_CHECKING:
    from ...config import DataSplitConfig


class BaseSplitter(ABC):
    """
    Base class for train/test splitters.

    Subclasses must override **at least one** of :meth:`split` (single
    derivation/validation pair) or :meth:`iter_splits` (one or more pairs).
    The other method then defaults to delegating: ``split`` returns the first
    result of ``iter_splits``; ``iter_splits`` yields the single result of
    ``split``. This lets callers consume any splitter polymorphically without
    having to know whether the strategy produces one or many cohorts.

    Parameters
    ----------
    config : DataSplitConfig
        Configuration object describing the split.
    """

    def __init__(self, config: "DataSplitConfig"):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__module__)
        self._verify_overrides()

    def _verify_overrides(self) -> None:
        # At least one of split/iter_splits must be overridden, otherwise
        # the default delegation would recurse infinitely.
        cls = type(self)
        split_overridden = cls.split is not BaseSplitter.split
        iter_overridden = cls.iter_splits is not BaseSplitter.iter_splits
        if not (split_overridden or iter_overridden):
            raise TypeError(
                f"{cls.__name__} must override at least one of 'split' or 'iter_splits'."
            )

    def split(self, df: pd.DataFrame, **kwargs) -> DataSplitResult:
        """Return a single train/test split.

        Defaults to ``next(self.iter_splits(df))`` so that splitters which
        naturally produce many cohorts (e.g. leave-one-group-out) can satisfy
        the single-split contract without raising.
        """
        return next(self.iter_splits(df))

    def iter_splits(self, df: pd.DataFrame) -> Iterator[DataSplitResult]:
        """Yield one or more splits.

        Defaults to yielding :meth:`split`'s single result. Strategies that
        produce multiple cohorts override this.
        """
        yield self.split(df)

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise DataSplitError(
                f"Required column(s) not found in dataframe: {missing}",
            )
