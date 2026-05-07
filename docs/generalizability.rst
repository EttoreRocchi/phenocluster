Generalizability
================

Version 0.3.0 introduces a first-class generalizability assessment that
runs after the main pipeline. Three complementary cohort sources are
supported:

- **Temporal generalizability** - validate phenotypes derived on early
  data against later data via a fixed cutoff, a chronological fraction,
  or sliding/expanding rolling windows.
- **Multi-site generalizability** - validate phenotypes across
  hospitals or centers via leave-one-site-out (LOGO), holdout, or
  pairwise schemes.
- **External cohort CSVs** - validate against one or more separate CSV
  files that share the derivation schema. Configured via
  ``generalizability.external_cohorts``.

Training scope
--------------

The ``generalizability.training_scope`` setting controls how the model
used for in-CSV split metrics is fit:

- ``"per_split"`` (default): for every **in-CSV** split (temporal or
  multi-site), a fresh ``DataPreprocessor`` and StepMix model are fit
  on derivation rows only and then applied to the validation rows.
  Each cohort report carries ``fit_mode: per_split``. The pipeline's
  full-cohort model is left untouched for descriptive analyses
  elsewhere in the report.
- ``"global"``: in-CSV splits are scored by the pipeline's full-cohort
  model. Each cohort report carries ``fit_mode: global``. Faster, but
  appropriate only when the full-cohort model is the intended
  evaluation reference (e.g., descriptive transport check rather than
  a held-out validation claim).

For **external cohorts** (separate CSV files) the global model is
always used: the pipeline never saw those rows during training, so
the global-model path is the correct evaluation reference regardless
of ``training_scope``.

Feature-selector scope
~~~~~~~~~~~~~~~~~~~~~~

The feature selector is a separate concern from the model. By default
(``feature_selector_scope: auto``) the per-split path also refits the
feature selector on each split's derivation rows when that is safe:

- For unsupervised selectors (``variance``, ``correlation``,
  ``mutual_info`` over inter-feature MI), the selector is refit on the
  derivation rows.
- For supervised LASSO, the selector is refit when the target column
  is present in the derivation rows AND is **not** also one of the
  cross-cohort concordance outcomes (i.e., not an entry in
  ``outcome.outcome_columns`` or a survival ``time_column`` /
  ``event_column``). When the target collides with an outcome, the
  global selector is reused and a warning is recorded on the cohort
  report; selecting features that explicitly predict the outcome we
  are then comparing across cohorts would be a separate leakage
  concern.

Each cohort's report carries a ``feature_selector_mode`` field with one
of ``per_split_refit``, ``global_reused``,
``global_reused_with_warning``, or ``none`` (when feature selection is
disabled). Override the default with
``feature_selector_scope: per_split`` to make any unsafe situation a
hard error, or ``feature_selector_scope: global`` to always reuse the
pipeline's single selector.

Modes
-----

Apply-only
~~~~~~~~~~

Set ``refit: false`` to apply the derivation model to each validation
cohort without refitting. This is the fastest mode and reports cohort
size, log-likelihood, phenotype distribution, drift, and outcome
concordance. Calibration is not computed in this mode (the predicted
class label is, by construction, the argmax of the posterior).

Refit-and-match
~~~~~~~~~~~~~~~

Set ``refit: true`` to refit a fresh StepMix on the validation cohort
with the same hyperparameters as the derivation model. The resulting
cluster labels are aligned with the derivation labels using a padded
Hungarian assignment (``scipy.optimize.linear_sum_assignment``), which
gracefully handles unequal cluster counts (extra clusters become
"unmatched novel"; missing ones become "unmatched absent"). ARI and NMI
are computed on the **raw** (pre-alignment) labels because both metrics
are permutation-invariant; alignment is used only for matched-accuracy
and reporting.

Refit is automatically skipped (with a warning) when the validation
cohort is smaller than ``min_validation_size_for_refit`` (default 100).

Metrics
-------

Calibration
~~~~~~~~~~~

- **Brier score** (one-vs-rest per class, plus mean).
- **Expected calibration error** with 10 quantile bins by default.
- **Reliability curve** data (per-bin confidence, accuracy, count).

Calibration is meaningful only in refit-and-match mode, where the
refit-aligned labels serve as proxy ground truth.

Drift
~~~~~

A tidy per-feature table is computed against the derivation cohort:

- **PSI** (Population Stability Index) using equal-mass bin edges
  fitted on the derivation distribution, with Laplace smoothing
  (default floor ``0.5 / min(n_deriv, n_val)``) so empty bins do not
  blow up the log-ratio.
- **Kolmogorov-Smirnov** test for continuous variables.
- **Chi-square** test for categorical variables (with unseen
  categories folded into a ``"<unseen>"`` bucket).
- **Missing-rate difference** between cohorts.

Outcome concordance
~~~~~~~~~~~~~~~~~~~

For each outcome present in both cohorts, ``compare_outcomes``
extracts ``(log effect, SE)`` from the existing analyzers' Wald CIs
and reports:

- Pearson r and Spearman rho across phenotypes.
- Lin's concordance correlation coefficient.
- Sign agreement above a configurable absolute-effect floor.
- Per-phenotype Wald delta test on
  ``log(OR)_d - log(OR)_v`` with pooled SE
  ``sqrt(SE_d^2 + SE_v^2)``, BH-FDR-corrected within the outcome
  family.

The same machinery applies to Cox HRs via ``compare_survival``.

Configuration
-------------

.. code-block:: yaml

   generalizability:
     enabled: true
     training_scope: per_split        # per_split (default) | global
     refit: true                      # refit-and-match against the derivation-only fit
     min_validation_size_for_refit: 100
     temporal:
       time_column: admission_date
       scheme: cutoff                 # cutoff | fraction | sliding | expanding
       time_cutoff: "2020-12-31"
       # time_test_fraction: 0.2      # for scheme=fraction
       # n_windows: 3                 # for scheme=sliding|expanding
     multisite:
       site_column: center
       scheme: logo                   # logo | holdout | pairwise
       # holdout_sites: [SITE_A]      # for scheme=holdout
       min_site_size: 30
     external_cohorts:                # one or more separate CSVs
       - { path: ./cohort_B.csv, label: hospital_X, kind: site }
       - { path: ./cohort_2024.csv, label: era_2024, kind: temporal }
     calibration:        { enabled: true, n_bins: 10, strategy: quantile }
     drift:              { enabled: true, n_bins: 10, top_k: 20 }
     outcome_concordance: { enabled: true, fdr_method: bh, alpha: 0.05 }

At least one of ``temporal``, ``multisite``, or ``external_cohorts``
must be provided when ``enabled: true``; otherwise the stage raises a
configuration error.

Outputs
-------

- ``results/temporal_validation_results.json`` - one entry per
  temporal cohort.
- ``results/multisite_validation_results.json`` - one entry per
  multi-site cohort.
- ``results/external_cohorts_results.json`` - one entry per external
  CSV cohort listed under ``generalizability.external_cohorts``.
- ``results/generalizability_summary.json`` - aggregate ARI / PSI per
  kind, plus ``training_scope`` flag and
  ``mean_derivation_only_ari_to_global``.
- ``data/generalizability/cluster_distribution_<label>.csv`` and
  ``data/generalizability/drift_<label>.csv`` per cohort.
- New plots under ``plots/`` (cohort prevalence heatmaps, drift bar
  charts, OR concordance scatter, ARI forest).

Each cohort's JSON entry carries:

- ``fit_mode``: ``per_split`` (default, for in-CSV splits) or
  ``global`` (external cohorts and the legacy permissive path).
- ``derivation_only_ari``: ARI between the fresh per-split
  derivation-only fit and the global full-cohort model on the same
  rows. A high value means the per-split fit recovers the same
  structure as the descriptive global model.
- ``derivation_only_outcomes``: ORs computed against the
  derivation-only labels, fed into the cross-cohort outcome
  concordance comparison.

A new "Generalizability" section is appended to the static HTML
report when generalizability results are present.

Pitfalls
--------

- **Refit instability** with very small validation cohorts: the
  ``min_validation_size_for_refit`` guardrail (default 100) skips
  refit and emits a warning rather than producing meaningless ARI/NMI.
- **Time-column leakage**: do not include the time column itself
  among ``continuous_columns`` or ``categorical_columns``.
- **Missing values in the partition column** (``time_column`` or
  ``site_column``) are excluded from the partitioning step and tracked
  in the result's ``stratification_fallback_reason``.
