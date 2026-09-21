Output Files
============

PhenoCluster writes all results to the configured output directory
(default: ``results/``).

Main outputs
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``analysis_report.html``
     - Comprehensive interactive HTML report with all results and visualisations. Skipped when ``generate_html_report: false`` (or ``--no-html-report`` on the CLI). JSON/CSV outputs are still written.
   * - ``results/cluster_statistics.json``
     - Phenotype sizes, feature distributions, classification quality metrics
   * - ``results/outcome_results.json``
     - Odds ratios with confidence intervals and FDR-corrected p-values
   * - ``results/survival_results.json``
     - Kaplan-Meier estimates, Nelson-Aalen curves, and Cox PH hazard ratios with confidence intervals
   * - ``results/multistate_results.json``
     - Transition hazard ratios, state occupation probabilities, and pathway analysis
   * - ``results/classification_quality.json``
     - Per-phenotype Average Posterior Probability and assignment confidence on the full cohort
   * - ``results/classification_quality_test.json``
     - The same classification quality metrics restricted to the held-out test rows
   * - ``results/model_selection_summary.json``
     - Best number of clusters, the criterion used, and its value at the selected K
   * - ``results/feature_importance.json``
     - Feature characterisation per phenotype (effect sizes, dominant categories)
   * - ``results/feature_selection.json``
     - Selected and removed features with their scores (when ``feature_selection.enabled: true``)
   * - ``results/validation_report.json``
     - Internal validation metrics (train/test log-likelihood, cluster proportions)
   * - ``results/stability_results.json``
     - Consensus clustering stability metrics
   * - ``results/consensus_matrix.npy``
     - Consensus co-assignment matrix from the stability analysis
   * - ``results/split_info.json``
     - Train/test split details (sample counts, stratification)
   * - ``results/external_validation_results.json``
     - External validation results (when ``external_validation.enabled: true``)
   * - ``results/temporal_validation_results.json``
     - Temporal generalizability cohorts (v0.3.0, when ``generalizability.temporal`` is set)
   * - ``results/multisite_validation_results.json``
     - Multi-site (LOGO / holdout) cohorts (v0.3.0, when ``generalizability.multisite`` is set)
   * - ``results/external_cohorts_results.json``
     - External-CSV cohorts (v0.3.0, one entry per file listed under ``generalizability.external_cohorts``)
   * - ``results/generalizability_summary.json``
     - Aggregate ARI / PSI per kind plus the resolved ``training_scope`` (v0.3.0)
   * - ``data/phenotypes_data.csv``
     - Original dataset augmented with phenotype assignments
   * - ``data/posterior_probabilities.csv``
     - Posterior class membership probabilities per patient
   * - ``data/model_fit_metrics.csv``
     - Information criteria (BIC, AIC, etc.), entropy, and average posterior probabilities
   * - ``data/model_selection_results.csv``
     - One row per candidate K with its criterion value and rank (when model selection runs)
   * - ``data/generalizability/``
     - Per-cohort ``cluster_distribution_<label>.csv`` and ``drift_<label>.csv`` (v0.3.0), plus ``phenotypes_<label>.csv`` with one row per validation patient: assigned phenotype and posterior probability per phenotype (v0.4.0)
   * - ``plots/``
     - Every Plotly figure, written twice: ``<name>.html`` for standalone viewing and the static report, ``<name>.json`` for the dashboard
   * - ``quality/``
     - ``data_quality_report.json`` plus the standalone data quality figures (missing data, outliers, correlation, variance) when ``data_quality.generate_report: true``
   * - ``logs/phenocluster.log``
     - Pipeline execution log (when ``logging.log_to_file: true``; the file name follows ``logging.log_file``)
   * - ``artifacts/``
     - ``config_used.yaml`` with the resolved configuration, the fitted ``preprocessor.joblib``, ``label_encoders.joblib`` and ``feature_selector.joblib``, and the ``cache_*.joblib`` files plus ``cache_manifest.json`` used for incremental re-runs

Visualisations
--------------

All figures are Plotly objects saved under ``plots/`` as ``<name>.html`` and
``<name>.json``, unless ``visualization.save_plots`` is ``false``. PhenoCluster
uses the colorblind-safe Wong (2011) palette, extended with Tol colours when a
model has more than eight phenotypes.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Plot
     - Description
   * - Model selection
     - Information criterion (e.g. BIC) vs number of clusters with best-k annotation
   * - Phenotype size distribution
     - Bar chart of patient counts per phenotype
   * - Classification quality
     - Posterior probability distributions per phenotype
   * - Continuous heatmap
     - Z-score standardised continuous features by phenotype
   * - Categorical heatmap
     - Within-phenotype proportions for categorical features
   * - Forest plots (OR)
     - Odds ratios with confidence interval bars
   * - Forest plots (HR)
     - Hazard ratios with confidence interval bars
   * - Kaplan-Meier curves
     - Survival curves per phenotype with step-function interpolation
   * - Nelson-Aalen curves
     - Cumulative hazard estimates per phenotype
   * - Cumulative incidence functions
     - Transition-specific hazard curves per transition
   * - State occupation probabilities
     - Time-varying probability of being in each state
   * - Pathway frequency
     - Most common clinical pathways from Monte Carlo simulation
   * - Cohort prevalence heatmap
     - Phenotype prevalence (%) across temporal or multi-site cohorts (v0.3.0)
   * - Drift bar chart
     - Top-K features by absolute PSI per cohort (v0.3.0)
   * - OR concordance scatter
     - Derivation vs validation log(OR) per phenotype with identity line (v0.3.0)
   * - LOGO / window forest
     - Per-cohort refit-and-match ARI dot plot (v0.3.0)
