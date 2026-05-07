"""
PhenoCluster Generalizability Evaluation
========================================

Temporal and multi-site generalizability evaluation for LCA/LPA-derived
phenotypes (introduced in v0.3.0). The subpackage exposes:

- :class:`GeneralizabilityEvaluator`: top-level facade.
- :class:`CohortReport`, :class:`GeneralizabilityReport`: result containers.
- :func:`apply_to_cohort`: replay the fitted derivation pipeline on a new
  cohort.
- :func:`refit_and_match`: refit the LCA model on a validation cohort and
  match labels via Hungarian alignment.
- :func:`feature_drift`: PSI / KS / chi-square per-feature drift table.
- :func:`compare_outcomes`, :func:`compare_survival`: per-phenotype OR/HR
  concordance with FDR-corrected delta tests.
- :func:`compute_calibration_block`: Brier, ECE and reliability data.
- :func:`cluster_distribution`: phenotype prevalence helper.
- :func:`chi2_cohort_comparison`: chi-square test on per-phenotype outcome
  counts between two cohorts (re-used by :class:`ExternalValidator`).
"""

from ._types import CohortReport, GeneralizabilityReport
from .apply_validator import apply_to_cohort, safe_apply_to_cohort
from .calibration import (
    brier_multiclass,
    compute_calibration_block,
    expected_calibration_error,
    reliability_curve,
)
from .drift import feature_drift, population_stability_index, top_drifted
from .evaluator import GeneralizabilityEvaluator
from .outcome_concordance import compare_outcomes, compare_survival, lin_ccc
from .prevalence import chi2_cohort_comparison, cluster_distribution
from .refit_validator import hungarian_alignment, refit_and_match

__all__ = [
    "GeneralizabilityEvaluator",
    "CohortReport",
    "GeneralizabilityReport",
    "apply_to_cohort",
    "safe_apply_to_cohort",
    "refit_and_match",
    "hungarian_alignment",
    "feature_drift",
    "population_stability_index",
    "top_drifted",
    "compare_outcomes",
    "compare_survival",
    "lin_ccc",
    "compute_calibration_block",
    "brier_multiclass",
    "expected_calibration_error",
    "reliability_curve",
    "cluster_distribution",
    "chi2_cohort_comparison",
]
