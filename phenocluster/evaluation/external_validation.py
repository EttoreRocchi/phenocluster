"""
PhenoCluster External Validation Module
========================================

Framework for validating phenotypes on independent external cohorts.

Key features:
- Apply fitted StepMix model to external cohort
- Compare cluster distributions across cohorts
- Compare outcome associations across cohorts
- Generate validation report
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class ExternalValidator:
    """
    Validates phenotypes on independent external cohorts.

    This class provides methods for external validation of phenotypes,
    including application of fitted models, cluster distribution comparison,
    and outcome comparison across cohorts.

    Parameters
    ----------
    config : PhenoClusterConfig
        Configuration object
    n_clusters : int
        Number of phenotype clusters

    Examples
    --------
    >>> validator = ExternalValidator(config, n_clusters=3)
    >>> results = validator.validate_with_model(X_external, model)
    """

    def __init__(self, config: PhenoClusterConfig, n_clusters: int):
        """
        Initialize the external validator.

        Parameters
        ----------
        config : PhenoClusterConfig
            Configuration object
        n_clusters : int
            Number of phenotype clusters
        """
        self.config = config
        self.n_clusters = n_clusters
        self.logger = get_logger("external_validation", config)

    def validate_with_model(
        self,
        X_external: np.ndarray,
        model,
        derivation_labels: Optional[np.ndarray] = None,
        derivation_outcomes: Optional[Dict] = None,
        n_external: int = 0,
        external_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Validate phenotypes on external cohort using fitted LCA/LPA model.

        The external data must be pre-processed using the same fitted
        preprocessor as the derivation cohort (pipeline handles this).

        Note on label alignment (S13a): since the same fitted StepMix model
        is applied to both cohorts, the predicted labels are inherently
        aligned - no Hungarian-algorithm relabelling is needed.

        Parameters
        ----------
        X_external : np.ndarray
            Pre-processed feature matrix for external cohort
        model : StepMix
            Fitted StepMix model from derivation cohort
        derivation_labels : np.ndarray, optional
            Labels from derivation cohort (for distribution comparison)
        derivation_outcomes : Dict, optional
            Outcome results from derivation cohort (for comparison)
        n_external : int
            Original sample size of external cohort (before preprocessing)
        external_df : pd.DataFrame, optional
            Pre-processed external DataFrame (for outcome comparison)

        Returns
        -------
        Dict
            Validation results including:
            - external_labels: Predicted phenotype labels
            - cluster_distribution: Distribution of phenotypes
            - log_likelihood: Model fit on external data
            - outcome_comparison: Comparison with derivation cohort (if available)
        """
        self.logger.info("EXTERNAL VALIDATION (Full Model)")
        n_samples = n_external if n_external > 0 else len(X_external)
        self.logger.info(f"External cohort size: {n_samples}")

        # Predict phenotypes
        self.logger.info("Predicting phenotypes for external cohort...")
        external_labels = model.predict(X_external)

        cluster_dist = self._compute_cluster_distribution(external_labels)
        self.logger.info("External cohort phenotype distribution:")
        for cluster_id, info in cluster_dist.items():
            self.logger.info(
                f"  Phenotype {cluster_id}: n={info['count']} ({info['percentage']:.1f}%)"
            )

        # Compute model fit on external data
        external_ll = model.score(X_external)
        self.logger.info(f"External cohort log-likelihood: {external_ll:.2f}")

        # Compute derivation cluster distribution for comparison
        derivation_dist = None
        if derivation_labels is not None:
            derivation_dist = self._compute_cluster_distribution(derivation_labels)

        results = {
            "external_labels": external_labels.tolist(),
            "cluster_distribution": cluster_dist,
            "derivation_distribution": derivation_dist,
            "log_likelihood": float(external_ll),
            "n_samples": n_samples,
        }

        # Compare outcomes if derivation results available
        if derivation_outcomes is not None:
            self.logger.info("Comparing outcomes across cohorts...")
            outcome_comparison = self._compare_outcomes(
                external_labels, derivation_outcomes, external_df=external_df
            )
            results["outcome_comparison"] = outcome_comparison

        return results

    def _compute_cluster_distribution(self, labels: np.ndarray) -> Dict[int, Dict]:
        """Compute cluster distribution statistics."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        distribution = {}
        for cluster_id, count in zip(unique, counts):
            distribution[int(cluster_id)] = {
                "count": int(count),
                "percentage": float(count / total * 100),
            }

        return distribution

    def _compare_outcomes(
        self,
        external_labels: np.ndarray,
        derivation_outcomes: Dict,
        external_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Compare outcome associations between derivation and external cohorts.

        When *external_df* is provided and contains the relevant outcome
        columns, prevalence per cluster is computed for the external cohort
        and a chi-square test is used to assess distributional shift between
        derivation and external outcome profiles.

        Parameters
        ----------
        external_labels : np.ndarray
            Predicted phenotype labels for external cohort
        derivation_outcomes : Dict
            Outcome results from derivation cohort
        external_df : pd.DataFrame, optional
            Pre-processed external DataFrame with outcome columns

        Returns
        -------
        Dict
            Comparison results for each outcome, including external
            prevalence and chi-square test when external data is available.
        """
        comparison = {}

        # Get derivation prevalence from full_cohort results
        cohort_outcomes = derivation_outcomes.get("full_cohort", {})

        for outcome, outcome_results in cohort_outcomes.items():
            deriv_prevalence = {}
            deriv_counts: Dict[int, Dict] = {}
            for cluster_id, result in outcome_results.items():
                if isinstance(result, dict):
                    deriv_prevalence[cluster_id] = result.get("prevalence", np.nan)
                    deriv_counts[cluster_id] = {
                        "n_positive": result.get("n_positive", 0),
                        "n_total": result.get("n_total", 0),
                    }

            # External cluster sizes
            ext_cluster_sizes = {}
            for cluster_id in range(self.n_clusters):
                cluster_mask = external_labels == cluster_id
                ext_cluster_sizes[int(cluster_id)] = int(cluster_mask.sum())

            outcome_entry: Dict = {
                "derivation_prevalence": deriv_prevalence,
                "external_cluster_sizes": ext_cluster_sizes,
            }

            if external_df is not None and outcome in external_df.columns:
                ext_outcome = external_df[outcome].values
                ext_prevalence: Dict[int, float] = {}
                ext_outcome_counts: Dict[int, Dict] = {}

                for cluster_id in range(self.n_clusters):
                    mask = external_labels == cluster_id
                    if mask.sum() > 0:
                        cluster_vals = ext_outcome[mask]
                        valid = cluster_vals[~np.isnan(cluster_vals)]
                        n_total = len(valid)
                        n_positive = int(valid.sum()) if n_total > 0 else 0
                        prev = float(valid.mean()) if n_total > 0 else np.nan
                    else:
                        n_total = 0
                        n_positive = 0
                        prev = np.nan

                    ext_prevalence[int(cluster_id)] = prev
                    ext_outcome_counts[int(cluster_id)] = {
                        "n_positive": n_positive,
                        "n_total": n_total,
                    }

                outcome_entry["external_prevalence"] = ext_prevalence
                outcome_entry["external_outcome_counts"] = ext_outcome_counts

                # Chi-square test comparing outcome distribution across cohorts
                chi2_result = self._chi2_cohort_comparison(deriv_counts, ext_outcome_counts)
                if chi2_result is not None:
                    outcome_entry["chi2_statistic"] = chi2_result["statistic"]
                    outcome_entry["chi2_p_value"] = chi2_result["p_value"]

                self.logger.info(f"  Outcome '{outcome}' external prevalence per cluster:")
                for cid in sorted(ext_prevalence.keys()):
                    p = ext_prevalence[cid]
                    p_str = f"{p:.1%}" if not np.isnan(p) else "N/A"
                    self.logger.info(f"    Phenotype {cid}: {p_str}")
                if chi2_result is not None:
                    self.logger.info(
                        f"    Chi-square test (derivation vs external): "
                        f"chi2={chi2_result['statistic']:.2f}, "
                        f"p={chi2_result['p_value']:.4f}"
                    )

            comparison[outcome] = outcome_entry

        return comparison

    @staticmethod
    def _chi2_cohort_comparison(
        deriv_counts: Dict[int, Dict],
        ext_counts: Dict[int, Dict],
    ) -> Optional[Dict]:
        """
        Run a chi-square test comparing outcome counts across cohorts.

        Builds a 2 x K contingency table (rows = cohort, columns = cluster)
        with cells [n_positive, n_total - n_positive].

        Returns None when the table cannot be constructed (e.g. zero totals).
        """
        cluster_ids = sorted(set(deriv_counts.keys()) & set(ext_counts.keys()))
        if not cluster_ids:
            return None

        # Build contingency table: rows = [deriv_positive, ext_positive]
        #                                  [deriv_negative, ext_negative]
        # Actually we want a 2 x K table where each column is a cluster
        # and rows are [derivation, external], with cells = n_positive.
        # But a standard chi-square for distribution shift should compare
        # the 2 x 2K table (positive/negative per cluster per cohort).
        # Simplest: 2-row (cohorts) x K-col (clusters) table of positive counts
        # with expected counts derived from totals.
        # More informative: per-cluster 2x2 table. But for a single summary
        # test we use a 2 x K contingency table of [n_positive, n_negative].

        # Build 2 x (2*K) table: for each cluster, [positive, negative] per cohort
        row_deriv = []
        row_ext = []
        for cid in cluster_ids:
            d = deriv_counts[cid]
            e = ext_counts[cid]
            d_pos = d.get("n_positive", 0)
            d_tot = d.get("n_total", 0)
            e_pos = e.get("n_positive", 0)
            e_tot = e.get("n_total", 0)
            row_deriv.extend([d_pos, d_tot - d_pos])
            row_ext.extend([e_pos, e_tot - e_pos])

        table = np.array([row_deriv, row_ext])

        # Skip if any column sums are zero (chi2 undefined)
        col_sums = table.sum(axis=0)
        if np.any(col_sums == 0) or table.sum() == 0:
            return None

        try:
            stat, p, _, _ = chi2_contingency(table)
            return {"statistic": float(stat), "p_value": float(p)}
        except ValueError:
            return None
