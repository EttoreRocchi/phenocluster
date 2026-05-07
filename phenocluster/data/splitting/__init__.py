"""
PhenoCluster Splitting Strategies
=================================

Train/test partitioning strategies for derivation/validation workflows.

The default :class:`RandomSplitter` reproduces the original v0.2.0 behavior
(random or stratified split). The other strategies enable temporal and
multi-site generalizability assessments introduced in v0.3.0.
"""

from ._base import BaseSplitter
from ._holdout_group import HoldoutGroupSplitter
from ._logo import LeaveOneGroupOutSplitter
from ._random import RandomSplitter
from ._temporal import TemporalSplitter
from .factory import make_splitter

__all__ = [
    "BaseSplitter",
    "RandomSplitter",
    "TemporalSplitter",
    "HoldoutGroupSplitter",
    "LeaveOneGroupOutSplitter",
    "make_splitter",
]
