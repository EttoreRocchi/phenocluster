"""
PhenoCluster Generalizability Configuration
===========================================

Dataclass schema for the v0.3.0 temporal and multi-site generalizability
analyses. The top-level :class:`GeneralizabilityConfig` is plugged into
:class:`PhenoClusterConfig` and consumed by ``GeneralizationStage``.

YAML layout::

    generalizability:
      enabled: true
      refit: true
      min_validation_size_for_refit: 100
      temporal:
        time_column: "admission_date"
        scheme: "cutoff"
        time_cutoff: "2020-12-31"
      multisite:
        site_column: "center"
        scheme: "logo"
        min_site_size: 30
      calibration: { enabled: true, n_bins: 10, strategy: "quantile" }
      drift:        { enabled: true, n_bins: 10, top_k: 20 }
      outcome_concordance: { enabled: true, fdr_method: "bh", alpha: 0.05 }
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

VALID_TEMPORAL_SCHEMES = ("cutoff", "fraction", "sliding", "expanding")
VALID_MULTISITE_SCHEMES = ("logo", "holdout", "pairwise")
VALID_CALIBRATION_STRATEGIES = ("quantile", "uniform")
VALID_TRAINING_SCOPES = ("per_split", "global")
VALID_FEATURE_SELECTOR_SCOPES = ("auto", "global", "per_split")


@dataclass
class TemporalSpec:
    """Configuration for the temporal generalizability scheme.

    Parameters
    ----------
    time_column : str
        Date or datetime column on the input dataframe.
    scheme : {"cutoff", "fraction", "sliding", "expanding"}
        Sub-strategy.
    time_cutoff : str or datetime, optional
        Required when ``scheme="cutoff"``. Rows with
        ``time_column <= time_cutoff`` form the derivation cohort.
    time_test_fraction : float, optional
        Required when ``scheme="fraction"``. Fraction of the most recent
        rows used as the validation cohort (in ``(0, 1)``).
    n_windows : int, optional
        Required when ``scheme in {"sliding", "expanding"}``. Number of
        rolling validation windows.
    """

    time_column: str
    scheme: str = "cutoff"
    time_cutoff: Optional[Any] = None
    time_test_fraction: Optional[float] = None
    n_windows: Optional[int] = None

    def __post_init__(self):
        if not self.time_column:
            raise ValueError("TemporalSpec.time_column must not be empty")
        if self.scheme not in VALID_TEMPORAL_SCHEMES:
            raise ValueError(
                f"TemporalSpec.scheme must be one of {VALID_TEMPORAL_SCHEMES}, got '{self.scheme}'"
            )
        if self.scheme == "cutoff" and self.time_cutoff is None:
            raise ValueError("TemporalSpec scheme='cutoff' requires time_cutoff")
        if self.scheme == "fraction":
            if self.time_test_fraction is None:
                raise ValueError("TemporalSpec scheme='fraction' requires time_test_fraction")
            if not 0 < self.time_test_fraction < 1:
                raise ValueError("TemporalSpec.time_test_fraction must be in (0, 1)")
        if self.scheme in ("sliding", "expanding"):
            if not self.n_windows or self.n_windows < 2:
                raise ValueError(f"TemporalSpec scheme='{self.scheme}' requires n_windows >= 2")


@dataclass
class MultiSiteSpec:
    """Configuration for the multi-site generalizability scheme.

    Parameters
    ----------
    site_column : str
        Categorical column identifying the site/center of each row.
    scheme : {"logo", "holdout", "pairwise"}
        ``"logo"`` runs leave-one-site-out across every site;
        ``"holdout"`` holds out the single (or several) sites listed in
        ``holdout_sites``; ``"pairwise"`` is reserved for future expansion
        and currently behaves like LOGO.
    holdout_sites : list, optional
        Required when ``scheme="holdout"``. Site values to hold out.
    min_site_size : int, default 30
        Sites with fewer rows than this are skipped (LOGO and pairwise) or
        rejected (holdout).
    """

    site_column: str
    scheme: str = "logo"
    holdout_sites: Optional[List[Any]] = None
    min_site_size: int = 30

    def __post_init__(self):
        if not self.site_column:
            raise ValueError("MultiSiteSpec.site_column must not be empty")
        if self.scheme not in VALID_MULTISITE_SCHEMES:
            raise ValueError(
                f"MultiSiteSpec.scheme must be one of {VALID_MULTISITE_SCHEMES}, "
                f"got '{self.scheme}'"
            )
        if self.scheme == "holdout" and not self.holdout_sites:
            raise ValueError("MultiSiteSpec scheme='holdout' requires non-empty holdout_sites")
        if self.min_site_size < 1:
            raise ValueError("MultiSiteSpec.min_site_size must be >= 1")


@dataclass
class CalibrationSubConfig:
    """Calibration metric settings (Brier, ECE, reliability)."""

    enabled: bool = True
    n_bins: int = 10
    strategy: str = "quantile"

    def __post_init__(self):
        if self.n_bins < 2:
            raise ValueError("CalibrationSubConfig.n_bins must be >= 2")
        if self.strategy not in VALID_CALIBRATION_STRATEGIES:
            raise ValueError(
                f"CalibrationSubConfig.strategy must be one of "
                f"{VALID_CALIBRATION_STRATEGIES}, got '{self.strategy}'"
            )


@dataclass
class DriftSubConfig:
    """Distribution drift settings."""

    enabled: bool = True
    n_bins: int = 10
    top_k: int = 20

    def __post_init__(self):
        if self.n_bins < 2:
            raise ValueError("DriftSubConfig.n_bins must be >= 2")
        if self.top_k < 1:
            raise ValueError("DriftSubConfig.top_k must be >= 1")


@dataclass
class OutcomeConcordanceSubConfig:
    """Cross-cohort outcome concordance settings."""

    enabled: bool = True
    fdr_method: str = "bh"
    alpha: float = 0.05
    effect_floor: float = 0.1

    def __post_init__(self):
        if not 0 < self.alpha < 1:
            raise ValueError("OutcomeConcordanceSubConfig.alpha must be in (0, 1)")
        if self.effect_floor < 0:
            raise ValueError("OutcomeConcordanceSubConfig.effect_floor must be >= 0")


VALID_EXTERNAL_KINDS = ("temporal", "site", "external")


@dataclass
class ExternalCohortSpec:
    """A single external validation cohort supplied as a separate CSV file.

    Parameters
    ----------
    path : str
        Path to a CSV with the same column schema as the derivation data.
    label : str
        Human-readable identifier shown in JSON, plots, and the dashboard.
    kind : {"temporal", "site", "external"}
        Whether this cohort is grouped under temporal validation, multi-site
        validation, or treated as a generic external cohort. Routing only
        affects which JSON file the result lands in and which section of the
        report it appears under.
    """

    path: str
    label: str
    kind: str = "external"

    def __post_init__(self):
        if not self.path:
            raise ValueError("ExternalCohortSpec.path must not be empty")
        if not self.label:
            raise ValueError("ExternalCohortSpec.label must not be empty")
        if self.kind not in VALID_EXTERNAL_KINDS:
            raise ValueError(
                f"ExternalCohortSpec.kind must be one of {VALID_EXTERNAL_KINDS}, got '{self.kind}'"
            )


@dataclass
class GeneralizabilityConfig:
    """Top-level generalizability configuration.

    At least one source of validation cohorts must be provided when
    ``enabled=True``: an in-CSV split (``temporal`` or ``multisite``) or
    an external CSV (via ``external_cohorts``).

    Notes
    -----
    ``training_scope`` controls how the model used for generalizability
    metrics on in-CSV splits is fit:

    - ``"per_split"`` (default): for each (derivation, validation) split,
      a fresh preprocessor and StepMix model are fit on derivation rows
      only, then applied to the validation rows. The pipeline's
      full-cohort model stays untouched for descriptive analyses.
    - ``"global"``: the validation rows are scored by the pipeline's
      full-cohort model. Faster but only appropriate when the full-cohort
      model is the intended evaluation reference.

    Outcome ORs are reported in both scopes simultaneously: the existing
    full-cohort block stays as the descriptive output, and a
    derivation-only block is computed per split for cross-cohort
    concordance comparisons. External cohorts (separate CSVs) always use
    the global model since they were never seen during training.
    """

    enabled: bool = False
    refit: bool = True
    min_validation_size_for_refit: int = 100
    training_scope: str = "per_split"
    feature_selector_scope: str = "auto"
    temporal: Optional[TemporalSpec] = None
    multisite: Optional[MultiSiteSpec] = None
    external_cohorts: List[ExternalCohortSpec] = field(default_factory=list)
    calibration: CalibrationSubConfig = field(default_factory=CalibrationSubConfig)
    drift: DriftSubConfig = field(default_factory=DriftSubConfig)
    outcome_concordance: OutcomeConcordanceSubConfig = field(
        default_factory=OutcomeConcordanceSubConfig
    )

    def __post_init__(self):
        if self.min_validation_size_for_refit < 1:
            raise ValueError("GeneralizabilityConfig.min_validation_size_for_refit must be >= 1")
        if self.training_scope not in VALID_TRAINING_SCOPES:
            raise ValueError(
                f"GeneralizabilityConfig.training_scope must be one of "
                f"{VALID_TRAINING_SCOPES}, got '{self.training_scope}'"
            )
        if self.feature_selector_scope not in VALID_FEATURE_SELECTOR_SCOPES:
            raise ValueError(
                f"GeneralizabilityConfig.feature_selector_scope must be one of "
                f"{VALID_FEATURE_SELECTOR_SCOPES}, got '{self.feature_selector_scope}'"
            )
        if self.enabled:
            has_in_csv = self.temporal is not None or self.multisite is not None
            has_external = bool(self.external_cohorts)
            if not (has_in_csv or has_external):
                raise ValueError(
                    "GeneralizabilityConfig.enabled=True requires at least one of "
                    "'temporal', 'multisite', or 'external_cohorts'"
                )
