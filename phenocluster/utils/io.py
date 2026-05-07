"""
PhenoCluster Safe I/O Helpers
=============================

Utilities for reading user-supplied CSV files with light defence against
foot-guns: pseudo-files (``/dev/zero``, named pipes), oversized files that
would exhaust memory, and unresolved relative paths. Used for external
generalizability cohort CSVs where the path comes from YAML config.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from ..core.exceptions import DataSplitError

DEFAULT_MAX_BYTES = 2 * 1024**3


def safe_read_csv(
    path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    **read_csv_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV after validating the path is a regular file under ``max_bytes``.

    Parameters
    ----------
    path : Path
        Filesystem path to a CSV. Symlinks are resolved before validation;
        the resolved target must be a regular file.
    max_bytes : int, default 2 GB
        Reject the read if the file's size exceeds this. Guards against
        accidentally pointing at a 100 GB log file or ``/dev/zero``.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.

    Raises
    ------
    DataSplitError
        If the path is missing, not a regular file, or exceeds ``max_bytes``.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise DataSplitError(f"CSV not found at {resolved}")
    if not resolved.is_file():
        raise DataSplitError(
            f"CSV path is not a regular file: {resolved} "
            "(symlinks must resolve to a regular file; pseudo-files are rejected)"
        )
    size = resolved.stat().st_size
    if size > max_bytes:
        raise DataSplitError(
            f"CSV at {resolved} is {size} bytes, larger than the "
            f"{max_bytes}-byte cap; pass max_bytes= to override."
        )
    return pd.read_csv(resolved, **read_csv_kwargs)
