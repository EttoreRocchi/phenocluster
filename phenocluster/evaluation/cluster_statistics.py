"""Descriptive statistics per cluster."""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class ClusterStatistics:
    """Compute descriptive statistics for each cluster."""

    def __init__(self, config: PhenoClusterConfig, n_clusters: int):
        self.config = config
        self.n_clusters = n_clusters
        self.logger = get_logger("evaluation", config)

    def compute_cluster_statistics(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        original_df: Optional[pd.DataFrame] = None,
    ) -> Dict[int, Dict]:
        """Compute descriptive statistics for each cluster."""
        self.logger.info("CLUSTER CHARACTERIZATION")

        df_with_labels = df.copy()
        df_with_labels["cluster"] = labels
        cluster_stats = {}

        for cluster_id in range(self.n_clusters):
            cluster_data = df_with_labels[df_with_labels["cluster"] == cluster_id]
            stats_dict = {"n_samples": len(cluster_data)}
            self.logger.info(f"CLUSTER {cluster_id} (n={len(cluster_data)})")

            if self.config.continuous_columns:
                stats_dict["continuous"] = self._continuous_stats(
                    cluster_data, cluster_id, df_with_labels, original_df
                )

            if self.config.categorical_columns:
                stats_dict["categorical"] = self._categorical_stats(cluster_data)

            cluster_stats[cluster_id] = stats_dict

        return cluster_stats

    def _continuous_stats(self, cluster_data, cluster_id, df_with_labels, original_df):
        cont_stats = {}
        for col in self.config.continuous_columns:
            if col not in cluster_data.columns:
                continue

            mean_val = cluster_data[col].mean()
            median_val = cluster_data[col].median()
            std_val = cluster_data[col].std()

            cont_stats[col] = {
                "mean": float(mean_val),
                "median": float(median_val),
                "std": float(std_val),
            }

            if original_df is not None and col in original_df.columns:
                orig_cluster = original_df[df_with_labels["cluster"] == cluster_id]
                cont_stats[col]["mean_original"] = float(orig_cluster[col].mean())
                cont_stats[col]["median_original"] = float(orig_cluster[col].median())
                cont_stats[col]["std_original"] = float(orig_cluster[col].std())

        return cont_stats

    def _categorical_stats(self, cluster_data):
        cat_stats = {}
        for col in self.config.categorical_columns:
            if col in cluster_data.columns:
                value_counts = cluster_data[col].value_counts(dropna=False)
                cat_stats[col] = (value_counts / len(cluster_data) * 100).to_dict()
        return cat_stats
