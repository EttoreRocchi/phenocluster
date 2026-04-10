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
from scipy import stats

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


class SurvivalAnalyzer:
    """Compare cluster survival using KM curves and Cox PH models."""

    # Sanity bound used to flag (not discard) implausible Cox PH point
    # estimates. A hazard ratio of 1000 corresponds to a log-HR of ~6.9; in
    # clinical survival data this almost always indicates separation, an
    # ill-conditioned design matrix, or a fitting failure rather than a true
    # effect. Estimates outside [1/MAX_REASONABLE_HR, MAX_REASONABLE_HR] are
    # tagged ``unreliable=True``, but the underlying coefficient is left intact.
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
        """
        Perform full survival analysis: KM/NA curves, log-rank, Cox PH.

        Parameters
        ----------
        data : pd.DataFrame
            Patient-level dataframe containing follow-up time and event columns.
        labels : np.ndarray
            Hard phenotype assignments aligned with the rows of ``data``.
        time_column : str
            Name of the follow-up time column in ``data``.
        event_column : str
            Name of the binary event indicator column in ``data``
            (1 = event, 0 = censored).

        Returns
        -------
        Dict
            Results dictionary with keys ``survival_data`` (Kaplan-Meier
            fits), ``nelson_aalen_data``, ``comparison`` (Cox PH and
            pairwise log-rank), ``median_survival``, ``survival_at_times``,
            ``ph_diagnostics``, ``ph_violated``, ``time_column``,
            ``event_column``, and ``logrank_p_value``. Returns an empty
            dict when no valid survival data is available.
        """
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
        """
        Perform weighted survival analysis using posterior probabilities.

        Each patient contributes to every phenotype with a weight equal to
        their posterior probability of belonging to it, rather than being
        assigned to a single hard cluster.

        Parameters
        ----------
        data : pd.DataFrame
            Patient-level dataframe containing follow-up time and event columns.
        posterior_probs : np.ndarray
            Array of shape ``(n_samples, n_clusters)`` with posterior
            phenotype probabilities. Rows must align with ``data``; columns
            must match ``self.n_clusters``.
        time_column : str
            Name of the follow-up time column in ``data``.
        event_column : str
            Name of the binary event indicator column in ``data``.
        min_weight : float, default 0.01
            Samples with posterior weight below this threshold for a given
            phenotype are dropped from that phenotype's weighted fit.

        Returns
        -------
        Dict
            Weighted survival results including weighted KM curves,
            median survival estimates, and pairwise comparisons. Returns
            an empty dict when no valid rows remain after filtering.

        Raises
        ------
        ValueError
            If ``posterior_probs`` has a row count that does not match
            ``data`` or a column count that does not match ``n_clusters``.
        """
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


def _grambsch_therneau_global(cph, cox_df, transform="log"):
    """Compute the Grambsch-Therneau global Schoenfeld test.

    Implements the Grambsch & Therneau global statistic:

    .. math:: T = \\frac{1}{d}\\, G^T I^{-1} G

    where :math:`G` is the vector of covariance-weighted sums of scaled
    Schoenfeld residuals over the :math:`d` distinct event times,
    :math:`I` is the information matrix on the normalised scale, and the
    time-transform values are centred via
    :math:`\\text{scalar} = \\sum g^2 - (\\sum g)^2 / d`.
    """
    schoenfeld = cph.compute_residuals(cox_df, kind="schoenfeld")
    d = schoenfeld.shape[0]
    p = schoenfeld.shape[1]
    if d == 0 or p == 0:
        return None

    event_times = schoenfeld.index.values
    if transform == "log":
        g = np.log(event_times)
    else:
        g = np.asarray(event_times, dtype=float)

    r = schoenfeld.values
    Gr = (g[:, None] * r).sum(axis=0)
    scalar = float((g**2).sum() - (g.sum() ** 2) / d)
    if scalar <= 0:
        return None

    norm_std = cph._norm_std.values
    V_normalised = cph.variance_matrix_.values * np.outer(norm_std, norm_std)
    try:
        I_normalised = np.linalg.inv(V_normalised)
    except np.linalg.LinAlgError:
        I_normalised = np.linalg.pinv(V_normalised)

    T_global = float((Gr @ I_normalised @ Gr) / (scalar * d))
    p_value = float(stats.chi2.sf(T_global, df=p))
    return {"test_statistic": T_global, "df": int(p), "p_value": p_value}


def _check_proportional_hazards(data, time_col, event_col, n_clusters, ref, logger):
    """Check PH assumption using lifelines' Schoenfeld-residual test."""
    from lifelines.statistics import proportional_hazard_test

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
                test_results = proportional_hazard_test(cph, cox_df, time_transform="rank")
        except Exception:
            test_results = None

        try:
            global_test = _grambsch_therneau_global(cph, cox_df, transform="log")
        except Exception as e:
            logger.debug(f"  Global Grambsch-Therneau computation failed: {e}")
            global_test = None

        ph["tested"] = True
        if test_results is not None and hasattr(test_results, "summary"):
            summary = test_results.summary
            covariates = list(summary.index)
            raw_p = [float(summary.loc[c, "p"]) for c in covariates]
            test_statistics = [float(summary.loc[c, "test_statistic"]) for c in covariates]

            ph["per_covariate"] = [
                {"covariate": cov, "test_statistic": ts, "p_value": p}
                for cov, ts, p in zip(covariates, test_statistics, raw_p)
            ]

            ph["global_test"] = global_test
            global_stat = global_test["test_statistic"] if global_test else None
            global_df = global_test["df"] if global_test else None
            global_p = global_test["p_value"] if global_test else None

            for entry in ph["per_covariate"]:
                if entry["p_value"] < 0.05:
                    ph["violations"].append(
                        {
                            "covariate": entry["covariate"],
                            "test_statistic": entry["test_statistic"],
                            "p_value": entry["p_value"],
                        }
                    )
                    logger.warning(
                        f"  PH may be violated for: {entry['covariate']} "
                        f"(raw p={entry['p_value']:.4f})"
                    )

            if global_p is not None and global_p < 0.05:
                ph["summary"] = (
                    f"Global Schoenfeld test rejects PH "
                    f"(chi2={global_stat:.2f}, df={global_df}, p={global_p:.4f})."
                )
                ph["ph_violated"] = True
                logger.warning(f"  {ph['summary']}")
            elif ph["violations"]:
                ph["summary"] = (
                    f"Global PH test not rejected, but raw per-covariate "
                    f"Schoenfeld p<0.05 for {len(ph['violations'])} covariate(s) "
                    "(exploratory; not multiplicity-adjusted)."
                )
                ph["ph_violated"] = False
                logger.info(f"  {ph['summary']}")
            else:
                ph["summary"] = "No PH violation detected (global Schoenfeld test)."
                ph["ph_violated"] = False
                logger.info(f"  {ph['summary']}")
        else:
            ph["summary"] = "PH diagnostic returned no results."
            ph["ph_violated"] = None
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
