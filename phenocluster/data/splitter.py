"""
PhenoCluster Data Splitter
==========================

Backwards-compatible alias for :class:`phenocluster.data.splitting.RandomSplitter`.

Historically this module exposed ``DataSplitter`` as the only splitting
strategy. Since v0.3.0 it is one of several strategies; the original name is
retained so existing imports keep working.
"""

from .splitting import RandomSplitter

DataSplitter = RandomSplitter

__all__ = ["DataSplitter"]
