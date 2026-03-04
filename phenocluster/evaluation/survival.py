"""
PhenoCluster Survival Analysis Module
======================================

Survival analysis with Kaplan-Meier/Nelson-Aalen descriptive curves
and Cox proportional hazards models for phenotype comparison.
"""

import contextlib
import io
from typing import Dict

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class SurvivalAnalyzer:
    """Compare cluster survival using KM curves and Cox PH models."""

    MAX_REASONABLE_HR = 1000

    def __init__(self, config: PhenoClusterConfig, n_clusters: int, reference_phenotype: int = 0):
        self.config = config
        self.n_clusters = n_clusters
        self.reference_phenotype = reference_phenotype
        self.logger = get_logger("survival", config)

    def analyze_survival(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        time_column: str,
        event_column: str,
    ) -> Dict:
        """Perform full survival analysis: KM/NA curves, log-rank, Cox PH."""
        self.logger.info("SURVIVAL ANALYSIS")
        survival_data = self._prepare_data(data, labels, time_column, event_column)
        if survival_data is None:
            return {}

        km_results, na_results, medians = self._fit_km_na(survival_data, time_column, event_column)
        ph_diag = self._check_ph(survival_data, time_column, event_column)
        logrank_p, pairwise_lr = self._log_rank_tests(survival_data, time_column, event_column)
        comparison = self._cox_comparison(survival_data, time_column, event_column, km_results)

        for key, lr in pairwise_lr.items():
            comparison.setdefault(key, {}).update(lr)

        time_points = np.linspace(0, survival_data[time_column].max(), 10)
        surv_at_t = self._survival_at_times(km_results, time_points)

        return {
            "survival_data": km_results,
            "nelson_aalen_data": na_results,
            "comparison": comparison,
            "median_survival": medians,
            "survival_at_times": surv_at_t,
            "ph_diagnostics": ph_diag,
            "ph_violated": ph_diag.get("ph_violated"),
            "time_column": time_column,
            "event_column": event_column,
            "logrank_p_value": logrank_p,
        }

    def analyze_weighted_survival(
        self,
        data: pd.DataFrame,
        posterior_probs: np.ndarray,
        time_column: str,
        event_column: str,
        min_weight: float = 0.01,
    ) -> Dict:
        """Perform weighted survival analysis using posterior probabilities."""
        self.logger.info("WEIGHTED SURVIVAL ANALYSIS")
        self._validate_columns(data, time_column, event_column)
        if posterior_probs.shape[0] != len(data):
            raise ValueError("Posterior probability rows must match data rows")
        if posterior_probs.shape[1] != self.n_clusters:
            raise ValueError("Posterior probability columns must match n_clusters")

        mask = data[[time_column, event_column]].notna().all(axis=1)
        survival_data = data[mask].copy().reset_index(drop=True)
        probs = posterior_probs[mask]
        if len(survival_data) == 0:
            self.logger.warning("No valid survival data")
            return {}

        self.logger.info(f"Analyzing weighted survival for {len(survival_data)} patients")
        wkm, medians = self._fit_weighted_km(
            survival_data, probs, time_column, event_column, min_weight
        )

        labels = probs.argmax(axis=1)
        survival_data["phenotype"] = labels
        comparison = {}
        if len(wkm) >= 2:
            try:
                comparison = _fit_cox_model(
                    survival_data,
                    time_column,
                    event_column,
                    self.n_clusters,
                    self.reference_phenotype,
                    self.config.inference,
                    self.MAX_REASONABLE_HR,
                    self.logger,
                )
            except Exception as e:
                self.logger.error(f"Cox PH weighted survival model failed: {e}")

        return {
            "weighted_km": wkm,
            "comparison": comparison,
            "median_survival": medians,
            "time_column": time_column,
            "event_column": event_column,
            "analysis_type": "weighted",
        }

    def _validate_columns(self, data, time_column, event_column):
        if time_column not in data.columns:
            raise ValueError(f"Time column '{time_column}' not found")
        if event_column not in data.columns:
            raise ValueError(f"Event column '{event_column}' not found")

    def _prepare_data(self, data, labels, time_column, event_column):
        self._validate_columns(data, time_column, event_column)
        df = data.copy()
        df["phenotype"] = labels
        mask = df[[time_column, event_column]].notna().all(axis=1)
        df = df[mask].copy()
        if len(df) == 0:
            self.logger.warning("No valid survival data available")
            return None
        self.logger.info(f"Analyzing survival for {len(df)} patients")
        return df

    def _fit_km_na(self, survival_data, time_col, event_col):
        """Fit Kaplan-Meier and Nelson-Aalen curves per cluster."""
        km_results, na_results, medians = {}, {}, {}
        for cid in range(self.n_clusters):
            cd = survival_data[survival_data["phenotype"] == cid]
            if len(cd) == 0:
                continue
            try:
                km = _fit_km_for_cluster(cd, time_col, event_col, cid)
                km_results[cid] = km["km"]
                na_results[cid] = km["na"]
                medians[cid] = km["median"]
                self.logger.info(
                    f"Cluster {cid}: n={len(cd)}, events={int(cd[event_col].sum())}, "
                    f"median={km['median']:.2f}"
                )
            except Exception as e:
                self.logger.error(f"Failed KM/NA for Cluster {cid}: {e}")
        return km_results, na_results, medians

    def _fit_weighted_km(self, data, probs, time_col, event_col, min_weight):
        """Fit weighted KM curves per cluster."""
        results, medians = {}, {}
        for cid in range(self.n_clusters):
            weights = probs[:, cid]
            valid = weights >= min_weight
            if valid.sum() < 10:
                continue
            try:
                kmf = KaplanMeierFitter()
                kmf.fit(
                    durations=data.loc[valid, time_col].values,
                    event_observed=data.loc[valid, event_col].values,
                    weights=weights[valid],
                    label=f"Cluster {cid}",
                )
                eff_n = (weights[valid].sum() ** 2) / (weights[valid] ** 2).sum()
                results[cid] = {
                    "timeline": kmf.survival_function_.index.values,
                    "survival_function": kmf.survival_function_.values.flatten(),
                    "confidence_interval_lower": kmf.confidence_interval_.iloc[:, 0].values,
                    "confidence_interval_upper": kmf.confidence_interval_.iloc[:, 1].values,
                    "n_patients": int(valid.sum()),
                    "effective_n": float(eff_n),
                    "total_weight": float(weights[valid].sum()),
                    "n_events": float((data.loc[valid, event_col].values * weights[valid]).sum()),
                }
                medians[cid] = kmf.median_survival_time_
            except Exception as e:
                self.logger.error(f"Failed weighted KM for Cluster {cid}: {e}")
        return results, medians

    def _log_rank_tests(self, survival_data, time_col, event_col):
        """Run overall + pairwise log-rank tests."""
        if not self.config.inference.enabled:
            return None, {}
        logrank_p = None
        pairwise = {}
        try:
            from lifelines.statistics import logrank_test, multivariate_logrank_test

            lr = multivariate_logrank_test(
                survival_data[time_col],
                survival_data["phenotype"],
                survival_data[event_col],
            )
            logrank_p = float(lr.p_value)
            self.logger.info(f"  Log-rank: chi2={lr.test_statistic:.2f}, p={logrank_p:.4f}")

            ref = self.reference_phenotype
            for k in sorted(survival_data["phenotype"].unique()):
                if k == ref:
                    continue
                try:
                    m_r = survival_data["phenotype"] == ref
                    m_k = survival_data["phenotype"] == k
                    lr_p = logrank_test(
                        survival_data.loc[m_r, time_col],
                        survival_data.loc[m_k, time_col],
                        survival_data.loc[m_r, event_col],
                        survival_data.loc[m_k, event_col],
                    )
                    pairwise[f"{k}_vs_{ref}"] = {
                        "p_value": float(lr_p.p_value),
                        "p_value_method": "log-rank",
                    }
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"  Log-rank test failed: {e}")
        return logrank_p, pairwise

    def _check_ph(self, data, time_col, event_col):
        """Check proportional hazards assumption."""
        return _check_proportional_hazards(
            data,
            time_col,
            event_col,
            self.n_clusters,
            self.reference_phenotype,
            self.logger,
        )

    def _cox_comparison(self, survival_data, time_col, event_col, km_results):
        """Fit Cox PH model for HR inference."""
        if not self.config.inference.enabled or len(km_results) < 2:
            if not self.config.inference.enabled:
                self.logger.warning("  Inference disabled. Descriptive only.")
            return {}
        try:
            return _fit_cox_model(
                survival_data,
                time_col,
                event_col,
                self.n_clusters,
                self.reference_phenotype,
                self.config.inference,
                self.MAX_REASONABLE_HR,
                self.logger,
            )
        except Exception as e:
            self.logger.error(f"Cox PH model failed: {e}")
            return {}

    @staticmethod
    def _survival_at_times(km_results, time_points):
        """Calculate survival probabilities at specific time points."""
        from scipy.interpolate import interp1d

        result = {}
        for cid, km in km_results.items():
            sf = interp1d(
                km["timeline"],
                km["survival_function"],
                kind="previous",
                bounds_error=False,
                fill_value=(1.0, float(km["survival_function"][-1])),
            )
            result[cid] = sf(time_points).tolist()
        result["time_points"] = time_points.tolist()
        return result


def _fit_km_for_cluster(cluster_data, time_col, event_col, cluster_id):
    """Fit KM and NA for a single cluster."""
    kmf = KaplanMeierFitter()
    naf = NelsonAalenFitter()
    kmf.fit(cluster_data[time_col], cluster_data[event_col], label=f"Cluster {cluster_id}")
    naf.fit(cluster_data[time_col], cluster_data[event_col], label=f"Cluster {cluster_id}")
    return {
        "km": {
            "timeline": kmf.survival_function_.index.values,
            "survival_function": kmf.survival_function_.values.flatten(),
            "confidence_interval_lower": kmf.confidence_interval_.iloc[:, 0].values,
            "confidence_interval_upper": kmf.confidence_interval_.iloc[:, 1].values,
            "n_patients": len(cluster_data),
            "n_events": int(cluster_data[event_col].sum()),
        },
        "na": {
            "timeline": naf.cumulative_hazard_.index.values,
            "cumulative_hazard": naf.cumulative_hazard_.values.flatten(),
            "confidence_interval_lower": naf.confidence_interval_.iloc[:, 0].values,
            "confidence_interval_upper": naf.confidence_interval_.iloc[:, 1].values,
            "n_patients": len(cluster_data),
            "n_events": int(cluster_data[event_col].sum()),
        },
        "median": kmf.median_survival_time_,
    }


def _check_proportional_hazards(data, time_col, event_col, n_clusters, ref, logger):
    """Check PH assumption using lifelines diagnostics."""
    logger.info("  Checking proportional hazards assumption...")
    ph = {"tested": False, "violations": [], "summary": ""}
    try:
        cox_df = data[[time_col, event_col, "phenotype"]].copy().dropna()
        non_ref = [k for k in range(n_clusters) if k != ref]
        for k in non_ref:
            cox_df[f"phenotype_{k}"] = (cox_df["phenotype"] == k).astype(int)
        cox_df = cox_df.drop(columns=["phenotype"])

        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col=time_col, event_col=event_col, show_progress=False)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                test_results = cph.check_assumptions(cox_df, p_value_threshold=0.05)
        except Exception:
            test_results = None

        ph["tested"] = True
        if test_results is not None and len(test_results) > 0:
            for cov, rdf in test_results:
                ph["violations"].append({"covariate": cov, "test_stat": "see log"})
                logger.warning(f"  PH may be violated for: {cov}")
            ph["summary"] = f"PH potentially violated for {len(ph['violations'])} covariate(s)."
            ph["ph_violated"] = True
        else:
            ph["summary"] = "No PH violation detected."
            ph["ph_violated"] = False
            logger.info(f"  {ph['summary']}")
    except Exception as e:
        logger.debug(f"  PH check skipped: {e}")
        ph["summary"] = f"PH check skipped: {e}"
        ph["ph_violated"] = None
    return ph


def _fit_cox_model(data, time_col, event_col, n_clusters, ref, inference_cfg, max_hr, logger):
    """Fit Cox PH model and extract HRs."""
    cox_df = data[[time_col, event_col, "phenotype"]].copy().dropna()
    non_ref = [k for k in range(n_clusters) if k != ref]
    for k in non_ref:
        cox_df[f"phenotype_{k}"] = (cox_df["phenotype"] == k).astype(int)
    cox_df = cox_df.drop(columns=["phenotype"])

    n_events = int(cox_df[event_col].sum())
    if n_events < 2:
        logger.warning(f"Insufficient events ({n_events}) for Cox PH.")
        return {}

    n_covariates = len(non_ref)
    epv = n_events / n_covariates if n_covariates > 0 else n_events
    if epv < 10:
        logger.warning(
            f"Low events-per-variable ({epv:.1f}) for Cox PH model. "
            f"Results may be unstable (recommended EPV >= 10)."
        )

    cph = CoxPHFitter(
        penalizer=inference_cfg.cox_penalizer, alpha=1 - inference_cfg.confidence_level
    )
    cph.fit(cox_df, duration_col=time_col, event_col=event_col, show_progress=False)

    comparison = {}
    for k in non_ref:
        cov = f"phenotype_{k}"
        if cov not in cph.summary.index:
            continue
        hr = float(cph.summary.loc[cov, "exp(coef)"])
        ci = cph.confidence_intervals_
        ci_lo = float(np.exp(ci.loc[cov].iloc[0]))
        ci_hi = float(np.exp(ci.loc[cov].iloc[1]))
        p = float(cph.summary.loc[cov, "p"])
        bad = hr > max_hr or hr < 1 / max_hr

        comparison[f"{k}_vs_{ref}"] = {
            "HR": hr,
            "CI_lower": ci_lo,
            "CI_upper": ci_hi,
            "p_value": p,
            "p_value_method": "Cox PH",
            "unreliable": bad,
        }
        if bad:
            logger.warning(f"  Phenotype {k} vs {ref}: HR={hr:.2f} - unreliable")
        logger.info(
            f"  Phenotype {k} vs {ref}: HR={hr:.2f} "
            f"({inference_cfg.confidence_level * 100:.0f}% CI: {ci_lo:.2f}-{ci_hi:.2f}), p={p:.4f}"
        )
    return comparison
