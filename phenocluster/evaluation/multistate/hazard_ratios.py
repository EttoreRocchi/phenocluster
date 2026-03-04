"""Hazard ratio extraction from transition-specific Cox PH models."""

import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from ..stats_utils import apply_fdr_correction, create_phenotype_dummies
from .types import MIN_TRANSITION_TIME, PatientTrajectory, TransitionResult


def extract_hazard_ratios(
    transitions,
    trajectories: List[PatientTrajectory],
    n_clusters: int,
    reference_phenotype: int,
    inference_cfg,
    max_hr: float,
    logger,
    min_events_per_transition: int = 5,
    baseline_confounders: Optional[List[str]] = None,
) -> Dict[str, TransitionResult]:
    """Fit Cox PH models for each transition and extract hazard ratios.

    When ``baseline_confounders`` is provided, the Cox models include
    these covariates alongside phenotype dummies so that the reported
    HRs are adjusted for potential confounders.
    """
    logger.info("Fitting transition-specific Cox PH models...")
    results = {}

    for trans in transitions:
        result = _fit_single_transition_hr(
            trans,
            trajectories,
            n_clusters,
            reference_phenotype,
            inference_cfg,
            max_hr,
            logger,
            min_events_per_transition,
            baseline_confounders=baseline_confounders or [],
        )
        if result is not None:
            results[trans.name] = result

    if inference_cfg.fdr_correction:
        _apply_transition_fdr(results)

    logger.info(f"Fitted models for {len(results)} transitions")
    return results


def _fit_single_transition_hr(
    trans,
    trajectories,
    n_clusters,
    reference_phenotype,
    inference_cfg,
    max_hr,
    logger,
    min_events,
    baseline_confounders=None,
):
    """Extract hazard ratios for a single transition."""
    times_arr, events_arr, phen_arr = _collect_transition_data(
        trans,
        trajectories,
        n_clusters,
        reference_phenotype,
    )

    if len(times_arr) == 0:
        logger.info(f"No data for transition {trans.name}")
        return None

    n_events = int(events_arr.sum())
    n_at_risk = len(times_arr)

    if n_events < min_events:
        logger.info(f"Skipping transition {trans.name}: {n_events} events < {min_events} required")
        return None

    logger.info(f"  Transition {trans.name}: {n_events} events, {n_at_risk} at risk")

    _, non_ref_ids = create_phenotype_dummies(phen_arr, n_clusters, reference=reference_phenotype)

    # Collect confounder values aligned with transition data
    confounder_arrays: Dict[str, np.ndarray] = {}
    if baseline_confounders:
        confounder_arrays = _collect_confounder_data(
            trans,
            trajectories,
            baseline_confounders,
        )

    phenotype_effects = {
        reference_phenotype: {
            "HR": 1.0,
            "CI_lower": 1.0,
            "CI_upper": 1.0,
            "p_value": None,
            "test_method": "Reference",
        }
    }

    _fit_cox_for_transition(
        times_arr,
        events_arr,
        phen_arr,
        non_ref_ids,
        reference_phenotype,
        inference_cfg,
        max_hr,
        phenotype_effects,
        logger,
        trans.name,
        confounder_arrays=confounder_arrays,
    )

    return TransitionResult(
        transition_name=trans.name,
        from_state=trans.from_state,
        to_state=trans.to_state,
        n_events=n_events,
        n_at_risk=n_at_risk,
        phenotype_effects=phenotype_effects,
        covariate_effects={},
    )


def _collect_transition_data(trans, trajectories, n_clusters, reference_phenotype):
    """Collect time/event/phenotype arrays for a single transition."""
    transition_times = []
    transition_events = []
    transition_phenotypes = []

    for traj in trajectories:
        for i in range(len(traj.states) - 1):
            if traj.states[i] != trans.from_state:
                continue
            time_in_state = (
                traj.time_at_each_state[i]
                if i < len(traj.time_at_each_state)
                else MIN_TRANSITION_TIME
            )
            event = 1 if traj.states[i + 1] == trans.to_state else 0

            phenotype = reference_phenotype
            for p in range(n_clusters):
                if p == reference_phenotype:
                    continue
                cov_name = f"phenotype_{p}"
                if cov_name in traj.covariates.index and traj.covariates[cov_name] == 1.0:
                    phenotype = p
                    break

            transition_times.append(max(MIN_TRANSITION_TIME, time_in_state))
            transition_events.append(event)
            transition_phenotypes.append(phenotype)

    return np.array(transition_times), np.array(transition_events), np.array(transition_phenotypes)


def _collect_confounder_data(trans, trajectories, confounders):
    """Collect confounder values aligned with transition records."""
    conf_values: Dict[str, list] = {c: [] for c in confounders}
    for traj in trajectories:
        for i in range(len(traj.states) - 1):
            if traj.states[i] != trans.from_state:
                continue
            for c in confounders:
                val = traj.covariates[c] if c in traj.covariates.index else np.nan
                conf_values[c].append(val)
    return {c: np.array(v) for c, v in conf_values.items()}


def _fit_cox_for_transition(
    times_arr,
    events_arr,
    phen_arr,
    non_ref_ids,
    reference_phenotype,
    inference_cfg,
    max_hr,
    phenotype_effects,
    logger,
    trans_name,
    confounder_arrays=None,
):
    """Fit Cox PH and populate phenotype_effects dict."""
    try:
        cox_df = pd.DataFrame({"time": times_arr, "event": events_arr})
        non_ref_cox = [p for p in sorted(np.unique(phen_arr)) if p != reference_phenotype]
        for p in non_ref_cox:
            cox_df[f"phenotype_{p}"] = (phen_arr == p).astype(float)

        # Include baseline confounders if available
        if confounder_arrays:
            for cname, cvals in confounder_arrays.items():
                if len(cvals) == len(cox_df):
                    cox_df[cname] = cvals

        n_covariates = len(non_ref_cox)
        epv = int(events_arr.sum()) / max(n_covariates, 1)
        if epv < 10:
            logger.warning(f"Low EPV ({epv:.1f}) for {trans_name}. HR estimates may be unstable.")

        alpha = 1.0 - inference_cfg.confidence_level
        cph = CoxPHFitter(penalizer=inference_cfg.cox_penalizer, alpha=alpha)
        cph.fit(cox_df, duration_col="time", event_col="event", show_progress=False)

        ci = cph.confidence_intervals_
        for k in non_ref_ids:
            cov_name = f"phenotype_{k}"
            if cov_name in cph.summary.index:
                hr = float(cph.summary.loc[cov_name, "exp(coef)"])
                p_val = float(cph.summary.loc[cov_name, "p"])
                ci_lo = float(np.exp(ci.loc[cov_name].iloc[0])) if cov_name in ci.index else np.nan
                ci_hi = float(np.exp(ci.loc[cov_name].iloc[1])) if cov_name in ci.index else np.nan

                phenotype_effects[k] = {
                    "HR": hr,
                    "CI_lower": ci_lo,
                    "CI_upper": ci_hi,
                    "p_value": p_val,
                    "p_value_method": "Cox PH (Wald)",
                    "unreliable": hr > max_hr or hr < 1 / max_hr,
                }
                logger.info(
                    f"    Phenotype {k} vs {reference_phenotype}: "
                    f"HR={hr:.2f} (CI: {ci_lo:.2f}-{ci_hi:.2f}) p={p_val:.4f}"
                )
            else:
                phenotype_effects[k] = _nan_effect()

        for k in non_ref_ids:
            if k not in phenotype_effects:
                phenotype_effects[k] = _nan_effect()

    except Exception as e:
        logger.warning(f"Cox PH model failed for {trans_name}: {type(e).__name__}: {e}")
        logger.warning(traceback.format_exc())
        for k in non_ref_ids:
            phenotype_effects[k] = _nan_effect()


def _nan_effect():
    """Return a NaN hazard ratio entry for missing/failed estimates."""
    return {
        "HR": np.nan,
        "CI_lower": np.nan,
        "CI_upper": np.nan,
        "p_value": np.nan,
        "p_value_method": "Cox PH (Wald)",
        "unreliable": True,
    }


def _apply_transition_fdr(results: Dict[str, TransitionResult]):
    """Apply FDR correction across all transitions and phenotypes."""
    fdr_keys = []
    fdr_pvals = []
    for t_name, t_result in results.items():
        for pheno_id, effect_dict in t_result.phenotype_effects.items():
            pv = effect_dict.get("p_value")
            if pv is not None and not np.isnan(pv):
                fdr_keys.append((t_name, pheno_id))
                fdr_pvals.append(pv)

    if fdr_pvals:
        corrected = apply_fdr_correction(fdr_pvals)
        for (t_name, pheno_id), q_val in zip(fdr_keys, corrected):
            results[t_name].phenotype_effects[pheno_id]["q_value"] = q_val
