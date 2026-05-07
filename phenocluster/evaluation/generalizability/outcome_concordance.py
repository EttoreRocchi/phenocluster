"""
PhenoCluster Outcome-Association Concordance
============================================

Compares per-phenotype effect-size estimates between a derivation cohort
and a validation cohort.

For each outcome (binary regression OR, or survival Cox HR) we recover
``log(effect)`` and its standard error from the upstream analyzers'
``CI_lower`` / ``CI_upper``. The recovery assumes the upstream produced a
symmetric Wald CI on the log scale at the given ``z`` (default 1.96, i.e.
95% Wald). When the analyzer falls back to a non-Wald CI (e.g. the
L1-regularised logistic fallback in :class:`OutcomeAnalyzer`), the
recovered SE is approximate and those entries should be treated cautiously.

Reported metrics:

- Pearson r and Spearman rho across phenotypes,
- Lin's concordance correlation coefficient,
- Sign agreement (fraction of phenotypes whose log-effects agree in sign
  with absolute value above ``effect_floor``),
- A per-phenotype Wald delta test on ``log(OR)_d - log(OR)_v`` with
  ``SE = sqrt(SE_d^2 + SE_v^2)``, BH-FDR-corrected across phenotypes
  within the outcome family.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

from ..stats_utils import apply_fdr_correction

DEFAULT_Z = float(norm.ppf(0.975))


def _se_from_ci(
    point: float,
    ci_lower: Optional[float],
    ci_upper: Optional[float],
    z: float,
) -> Optional[float]:
    if (
        ci_lower is None
        or ci_upper is None
        or not np.isfinite(point)
        or not np.isfinite(ci_lower)
        or not np.isfinite(ci_upper)
        or point <= 0
        or ci_lower <= 0
        or ci_upper <= 0
    ):
        return None
    log_lo = float(np.log(ci_lower))
    log_hi = float(np.log(ci_upper))
    if not np.isfinite(log_lo) or not np.isfinite(log_hi) or log_hi <= log_lo:
        return None
    return float((log_hi - log_lo) / (2.0 * z))


def _extract_log_effect(
    entry: Dict[str, Any], effect_key: str, z: float
) -> Optional[Tuple[float, float]]:
    """Return ``(log_effect, se_log_effect)`` from a single phenotype result, or None."""
    if entry is None:
        return None
    point = entry.get(effect_key)
    if point is None or not np.isfinite(point) or point <= 0:
        return None
    log_eff = float(np.log(point))
    se = _se_from_ci(point, entry.get("CI_lower"), entry.get("CI_upper"), z)
    if se is None or not np.isfinite(se):
        return None
    return log_eff, se


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    if len(x) < 2:
        return float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=0)), float(np.var(y, ddof=0))
    cov = float(np.mean((x - mx) * (y - my)))
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float(2.0 * cov / denom)


def _vector_summary(log_d: np.ndarray, log_v: np.ndarray, effect_floor: float) -> Dict[str, float]:
    if len(log_d) < 2:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "lin_ccc": float("nan"),
            "sign_agreement": float("nan"),
            "n_phenotypes": int(len(log_d)),
        }
    pr = pearsonr(log_d, log_v)
    sr = spearmanr(log_d, log_v)
    abs_above = (np.abs(log_d) > effect_floor) & (np.abs(log_v) > effect_floor)
    if abs_above.sum() == 0:
        sign_agreement = float("nan")
    else:
        agree = np.sign(log_d[abs_above]) == np.sign(log_v[abs_above])
        sign_agreement = float(agree.mean())
    return {
        "pearson_r": float(pr.statistic) if hasattr(pr, "statistic") else float(pr[0]),
        "pearson_p": float(pr.pvalue) if hasattr(pr, "pvalue") else float(pr[1]),
        "spearman_rho": float(sr.statistic) if hasattr(sr, "statistic") else float(sr[0]),
        "spearman_p": float(sr.pvalue) if hasattr(sr, "pvalue") else float(sr[1]),
        "lin_ccc": lin_ccc(log_d, log_v),
        "sign_agreement": sign_agreement,
        "n_phenotypes": int(len(log_d)),
    }


def _per_phenotype_delta_tests(
    keys: List[Any],
    log_d: np.ndarray,
    se_d: np.ndarray,
    log_v: np.ndarray,
    se_v: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    raw_p: List[Optional[float]] = []
    for k, ld, sd, lv, sv in zip(keys, log_d, se_d, log_v, se_v):
        delta = float(ld - lv)
        pooled_se = float(np.sqrt(sd**2 + sv**2))
        if pooled_se == 0:
            z = float("nan")
            p = float("nan")
        else:
            z = float(delta / pooled_se)
            p = float(2.0 * norm.sf(abs(z)))
        rows.append(
            {
                "phenotype": k,
                "log_effect_derivation": ld,
                "log_effect_validation": lv,
                "delta_log_effect": delta,
                "pooled_se": pooled_se,
                "z": z,
                "p_value": p,
            }
        )
        raw_p.append(p if np.isfinite(p) else None)
    fdr = apply_fdr_correction(raw_p)
    for row, q in zip(rows, fdr):
        row["p_value_fdr"] = q
    return rows


def _compare_one_outcome(
    deriv_outcome: Dict[Any, Dict],
    val_outcome: Dict[Any, Dict],
    effect_key: str,
    z: float,
    effect_floor: float,
) -> Dict[str, Any]:
    common_keys = sorted(set(deriv_outcome) & set(val_outcome), key=str)
    log_d: List[float] = []
    se_d: List[float] = []
    log_v: List[float] = []
    se_v: List[float] = []
    keys: List[Any] = []
    for k in common_keys:
        d = _extract_log_effect(deriv_outcome[k], effect_key, z)
        v = _extract_log_effect(val_outcome[k], effect_key, z)
        if d is None or v is None:
            continue
        keys.append(k)
        log_d.append(d[0])
        se_d.append(d[1])
        log_v.append(v[0])
        se_v.append(v[1])
    log_d_arr = np.asarray(log_d)
    log_v_arr = np.asarray(log_v)
    se_d_arr = np.asarray(se_d)
    se_v_arr = np.asarray(se_v)
    return {
        "summary": _vector_summary(log_d_arr, log_v_arr, effect_floor),
        "per_phenotype": _per_phenotype_delta_tests(keys, log_d_arr, se_d_arr, log_v_arr, se_v_arr),
    }


def compare_outcomes(
    derivation_outcomes: Dict[str, Dict],
    validation_outcomes: Dict[str, Dict],
    *,
    z: float = DEFAULT_Z,
    effect_floor: float = 0.1,
) -> Dict[str, Any]:
    """Compare per-phenotype OR vectors between two cohorts.

    ``derivation_outcomes`` and ``validation_outcomes`` are expected to be
    nested dicts of the shape ``{outcome_name: {phenotype_id: {"OR": ...,
    "CI_lower": ..., "CI_upper": ...}}}`` produced by
    :class:`OutcomeAnalyzer`.
    """
    return _compare_outcome_dicts(
        derivation_outcomes, validation_outcomes, effect_key="OR", z=z, effect_floor=effect_floor
    )


def compare_survival(
    derivation_survival: Dict[str, Dict],
    validation_survival: Dict[str, Dict],
    *,
    z: float = DEFAULT_Z,
    effect_floor: float = 0.1,
) -> Dict[str, Any]:
    """Compare per-phenotype HR vectors between two cohorts.

    Mirrors :func:`compare_outcomes` but reads ``HR`` rather than ``OR``.
    """
    return _compare_outcome_dicts(
        derivation_survival,
        validation_survival,
        effect_key="HR",
        z=z,
        effect_floor=effect_floor,
    )


def _compare_outcome_dicts(
    deriv: Dict[str, Dict],
    val: Dict[str, Dict],
    *,
    effect_key: str,
    z: float,
    effect_floor: float,
) -> Dict[str, Any]:
    common = _common_outcome_names(deriv, val)
    return {
        outcome: _compare_one_outcome(
            deriv.get(outcome, {}),
            val.get(outcome, {}),
            effect_key=effect_key,
            z=z,
            effect_floor=effect_floor,
        )
        for outcome in common
    }


def _common_outcome_names(deriv: Dict[str, Dict], val: Dict[str, Dict]) -> Iterable[str]:
    return sorted(set(deriv.keys()) & set(val.keys()))
