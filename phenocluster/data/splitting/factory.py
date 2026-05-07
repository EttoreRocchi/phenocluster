"""
PhenoCluster Splitter Factory
=============================

Dispatches to the appropriate :class:`BaseSplitter` subclass based on
``config.strategy``.
"""

from typing import TYPE_CHECKING

from ._base import BaseSplitter
from ._holdout_group import HoldoutGroupSplitter
from ._logo import LeaveOneGroupOutSplitter
from ._random import RandomSplitter
from ._temporal import TemporalSplitter

if TYPE_CHECKING:
    from ...config import DataSplitConfig


def make_splitter(config: "DataSplitConfig") -> BaseSplitter:
    """
    Build the splitter implementation requested by ``config.strategy``.

    Parameters
    ----------
    config : DataSplitConfig
        Validated configuration. ``config.strategy`` selects the
        implementation.

    Returns
    -------
    BaseSplitter
        A splitter instance ready to consume a dataframe.
    """
    strategy = config.strategy
    if strategy == "random":
        return RandomSplitter(config)
    if strategy == "temporal":
        return TemporalSplitter(config)
    if strategy == "holdout_group":
        return HoldoutGroupSplitter(config)
    if strategy == "leave_one_group_out":
        return LeaveOneGroupOutSplitter(config)
    raise ValueError(f"Unknown split strategy '{strategy}'")
