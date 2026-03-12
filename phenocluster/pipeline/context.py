"""Typed pipeline context replacing the untyped ctx dict."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PipelineContext:
    """Carries intermediate results between pipeline stages.

    Replaces the untyped ``ctx: Dict`` that was threaded through
    the monolithic pipeline methods.
    """

    # Raw / filtered data
    data_raw: Optional[pd.DataFrame] = None
    data_filtered: Optional[pd.DataFrame] = None
    n_rows_original: int = 0

    # Missing info
    missing_info: Optional[Dict] = None

    # Data quality
    quality_report: Dict = field(default_factory=dict)

    # Split info
    split_info: Dict = field(default_factory=dict)

    # Preprocessing outputs
    X_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    data_train: Optional[pd.DataFrame] = None
    data_test: Optional[pd.DataFrame] = None

    # Feature selection
    feature_selection_report: Dict = field(default_factory=dict)

    # Model training
    model: Any = None
    selection_results: Dict = field(default_factory=dict)

    # Evaluation
    labels: Optional[np.ndarray] = None
    labels_test: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None
    proba_test: Optional[np.ndarray] = None
    cluster_stats: Optional[Dict] = None
    model_fit_metrics: Dict = field(default_factory=dict)
    test_metrics: Dict = field(default_factory=dict)
    n_clusters: int = 0
    validation_metrics: Dict = field(default_factory=dict)
    classification_quality: Dict = field(default_factory=dict)
    classification_quality_test: Dict = field(default_factory=dict)
    data_processed: Optional[pd.DataFrame] = None
    X: Optional[np.ndarray] = None
    original_continuous_data: Optional[pd.DataFrame] = None

    # Analyses
    stability_results: Dict = field(default_factory=dict)
    outcome_results: Dict = field(default_factory=dict)
    survival_results: Dict = field(default_factory=dict)
    multistate_results: Dict = field(default_factory=dict)

    # External validation
    external_validation_results: Dict = field(default_factory=dict)
