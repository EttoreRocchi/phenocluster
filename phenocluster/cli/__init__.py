"""PhenoCluster CLI package.

Re-exports the Typer app and backwards-compatible symbols so that
`phenocluster.cli:main` (declared in `pyproject.toml`) keeps working and
existing tests that import validation helpers continue to pass.
"""

import pandas as pd  # noqa: F401  re-exported for test patching

from ..config import PhenoClusterConfig  # noqa: F401  re-exported for test patching
from ..pipeline import PhenoClusterPipeline  # noqa: F401  re-exported for test patching
from .app import app, main, typer_click_object
from .validation import (
    _check_columns,
    _display_validation_results,
    _validate_against_data,
    _validate_multistate_structure,
    _validate_structure,
)

__all__ = [
    "app",
    "main",
    "typer_click_object",
    "PhenoClusterPipeline",
    "PhenoClusterConfig",
    "pd",
    "_check_columns",
    "_display_validation_results",
    "_validate_against_data",
    "_validate_multistate_structure",
    "_validate_structure",
]
