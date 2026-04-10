"""Outcome association analysis using logistic regression."""

from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger
from .stats_utils import apply_fdr_correction, create_phenotype_dummies


class OutcomeAnalyzer:
    """Logistic regression for binary outcome associations."""

    def __init__(self, config: PhenoClusterConfig, n_clusters: int):
        self.config = config
        self.n_clusters = n_clusters
        self.logger = get_logger("evaluation", config)

    def analyze_outcomes(
        self, df: pd.DataFrame, labels: np.ndarray, reference_phenotype: int = 0
    ) -> Dict[str, Dict]:
        """
        Perform outcome association analysis using logistic regression.

        Fits a logistic regression of each binary outcome in
        ``config.outcome_columns`` on phenotype dummies, using
        ``reference_phenotype`` as the reference level. When
        ``config.inference.fdr_correction`` is enabled, Benjamini-Hochberg
        FDR is applied globally across all outcome/phenotype p-values.

        Parameters
        ----------
        df : pd.DataFrame
            Patient-level dataframe containing the outcome columns.
        labels : np.ndarray
            Hard phenotype assignments aligned with the rows of ``df``.
        reference_phenotype : int, default 0
            Phenotype id to use as the logistic regression reference.

        Returns
        -------
        Dict[str, Dict]
            Mapping from outcome column name to its per-phenotype
            regression results. Returns an empty dict when
            ``config.outcome_columns`` is not configured.
        """
        if not self.config.outcome_columns:
            self.logger.info("No outcome columns specified. Skipping outcome analysis.")
            return {}

        self.logger.info("OUTCOME ASSOCIATION ANALYSIS")
        outcome_results = {}

        for outcome in self.config.outcome_columns:
            if outcome not in df.columns:
                self.logger.warning(f"Outcome '{outcome}' not found in dataframe")
                continue

            result = self._analyze_single_outcome(df, labels, outcome, reference_phenotype)
            if result is not None:
                outcome_results[outcome] = result

        if self.config.inference.enabled and self.config.inference.fdr_correction:
            self._apply_global_fdr(outcome_results)

        return outcome_results

    def _analyze_single_outcome(self, df, labels, outcome, reference_phenotype):
        """Analyze a single binary outcome."""
        self.logger.info(f"Outcome: {outcome}")
        # pd.isna handles float, integer, nullable-integer and object dtypes
        valid_idx = ~pd.isna(df[outcome].values)
        n_missing = int((~valid_idx).sum())
        if n_missing > 0:
            pct = n_missing / len(valid_idx) * 100
            self.logger.info(f"  Missing: {n_missing} ({pct:.1f}%) - complete-case (MCAR)")
        y = df[outcome].values[valid_idx].astype(int)
        labels_valid = labels[valid_idx]

        unique_vals = np.unique(y)
        if len(unique_vals) != 2:
            self.logger.warning(
                f"Outcome '{outcome}' not binary ({len(unique_vals)} values). Skip."
            )
            return None

        dummies, non_ref_ids = create_phenotype_dummies(
            labels_valid, self.n_clusters, reference=reference_phenotype
        )
        self.logger.info(f"  Prevalence: {y.mean() * 100:.1f}%")

        results = {}
        ref_mask = labels_valid == reference_phenotype
        results[reference_phenotype] = {
            "OR": 1.0,
            "CI_lower": 1.0,
            "CI_upper": 1.0,
            "p_value": None,
            "test_method": "Reference",
            "prevalence": float(y[ref_mask].mean() if ref_mask.sum() > 0 else 0),
        }

        inference = self.config.inference
        if inference.enabled:
            self._fit_logistic(y, dummies, non_ref_ids, labels_valid, results, inference)
            self._contingency_tests(y, labels_valid, non_ref_ids, reference_phenotype, results)
        else:
            for cid in non_ref_ids:
                mask = labels_valid == cid
                results[cid] = {"prevalence": float(y[mask].mean() if mask.sum() > 0 else 0)}

        return results

    def _fit_logistic(self, y, dummies, non_ref_ids, labels_valid, results, inference):
        """Fit logistic regression and extract ORs."""
        confidence_level = inference.confidence_level
        z = norm.ppf(1 - (1 - confidence_level) / 2)

        try:
            X_design = sm.add_constant(dummies)
            model = sm.Logit(y, X_design)
            try:
                fit_result = model.fit(disp=0, maxiter=100)
            except Exception:
                self.logger.warning(
                    "Logistic regression did not converge; "
                    "falling back to L1-regularised fit. "
                    "Reported ORs may be biased toward 1.0 "
                    "and CIs may be too narrow."
                )
                fit_result = model.fit_regularized(method="l1", alpha=0.1, disp=0)

            for i, cid in enumerate(non_ref_ids):
                mask = labels_valid == cid
                prevalence = float(y[mask].mean() if mask.sum() > 0 else 0)
                coef = fit_result.params[i + 1]
                se = fit_result.bse[i + 1]
                results[cid] = {
                    "OR": float(np.exp(coef)),
                    "CI_lower": float(np.exp(coef - z * se)),
                    "CI_upper": float(np.exp(coef + z * se)),
                    "p_value": float(fit_result.pvalues[i + 1]),
                    "test_method": "logistic regression",
                    "prevalence": prevalence,
                }
        except Exception as e:
            self.logger.warning(f"Logistic regression failed: {e}")
            for cid in non_ref_ids:
                mask = labels_valid == cid
                results[cid] = {
                    "OR": np.nan,
                    "CI_lower": np.nan,
                    "CI_upper": np.nan,
                    "p_value": np.nan,
                    "test_method": "logistic regression (failed)",
                    "prevalence": float(y[mask].mean() if mask.sum() > 0 else 0),
                }

    def _contingency_tests(self, y, labels_valid, non_ref_ids, ref, results):
        """Run chi-square / Fisher's exact tests per phenotype."""
        for cid in non_ref_ids:
            try:
                mask_k = labels_valid == cid
                mask_ref = labels_valid == ref
                table = np.array(
                    [
                        [int((mask_k & (y == 1)).sum()), int((mask_k & (y == 0)).sum())],
                        [int((mask_ref & (y == 1)).sum()), int((mask_ref & (y == 0)).sum())],
                    ]
                )
                outcome_test = self.config.inference.outcome_test
                small_cells = table.min() < 5
                use_fisher = outcome_test == "fisher" or (outcome_test == "auto" and small_cells)
                if use_fisher:
                    _, p_val = stats.fisher_exact(table)
                    method = "Fisher's exact"
                else:
                    apply_yates = outcome_test == "auto" and small_cells
                    chi2, p_val, _, _ = stats.chi2_contingency(table, correction=apply_yates)
                    method = "chi-square (Yates)" if apply_yates else "chi-square"

                if cid not in results:
                    results[cid] = {}
                results[cid]["p_value_contingency"] = float(p_val)
                results[cid]["p_value_contingency_method"] = method
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Contingency test failed for cluster {cid}: {e}")
                if cid not in results:
                    results[cid] = {}
                results[cid]["p_value_contingency"] = None
                results[cid]["p_value_contingency_method"] = None

    def _apply_global_fdr(self, outcome_results):
        """Apply BH FDR correction across all outcomes and clusters."""
        all_p_entries = []
        for outcome_name, results_per_cluster in outcome_results.items():
            for k, cluster_data in results_per_cluster.items():
                if isinstance(cluster_data, dict) and "p_value" in cluster_data:
                    all_p_entries.append((outcome_name, k))
        if all_p_entries:
            raw_p = [outcome_results[o][k]["p_value"] for o, k in all_p_entries]
            adjusted = apply_fdr_correction(raw_p)
            for (o, k), q in zip(all_p_entries, adjusted):
                outcome_results[o][k]["p_value_fdr"] = q
