"""
PhenoCluster Evaluation - Cluster Evaluator Facade
====================================================

Composes ClusterStatistics, OutcomeAnalyzer, and FeatureCharacterizer
into a single interface for the pipeline.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import FeatureCharacterizationConfig, PhenoClusterConfig
from ..utils.logging import get_logger
from .cluster_statistics import ClusterStatistics
from .feature_characterization import FeatureCharacterizer
from .outcome_analysis import OutcomeAnalyzer

# Re-export for any remaining import sites
from .stats_utils import create_phenotype_dummies  # noqa: F401


class ClusterEvaluator:
    """Facade composing cluster statistics, outcome analysis, and feature characterization."""

    def __init__(self, config: PhenoClusterConfig, model):
        self.config = config
        self.model = model
        self.n_clusters = model.n_components
        self.logger = get_logger("evaluation", config)

        self._stats = ClusterStatistics(config, self.n_clusters)
        self._outcomes = OutcomeAnalyzer(config, self.n_clusters)
        self._features = FeatureCharacterizer(config, self.n_clusters)

    def compute_cluster_statistics(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        original_df: Optional[pd.DataFrame] = None,
    ) -> Dict[int, Dict]:
        return self._stats.compute_cluster_statistics(df, labels, original_df)

    def analyze_outcomes(
        self, df: pd.DataFrame, labels: np.ndarray, reference_phenotype: int = 0
    ) -> Dict[str, Dict]:
        return self._outcomes.analyze_outcomes(df, labels, reference_phenotype)

    def compute_feature_importance(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Dict]:
        return self._features.compute_feature_importance(df, labels)

    def get_top_features_per_cluster(
        self,
        feature_importance: Dict,
        n_top: int = 10,
        feature_char_config: Optional[FeatureCharacterizationConfig] = None,
    ) -> Dict[int, List[Dict]]:
        return self._features.get_top_features_per_cluster(
            feature_importance, n_top, feature_char_config
        )
