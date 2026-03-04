"""
PhenoCluster Model Selection Module
===================================

Model selection using information criteria for Latent Class / Profile Analysis.
"""

from .grid_search import StepMixModelSelector
from .scorers import (
    AVAILABLE_CRITERIA,
    aic_score,
    bic_score,
    caic_score,
    create_scorer,
    get_all_criteria,
    icl_score,
    relative_entropy_score,
    sabic_score,
)

__all__ = [
    # Scorers
    "bic_score",
    "aic_score",
    "caic_score",
    "sabic_score",
    "icl_score",
    "relative_entropy_score",
    "create_scorer",
    "get_all_criteria",
    "AVAILABLE_CRITERIA",
    # Model Selection
    "StepMixModelSelector",
]
