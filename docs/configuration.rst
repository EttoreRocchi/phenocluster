Configuration Reference
=======================

PhenoCluster is configured via a YAML file with nested sections. All parameters
have sensible defaults; only ``data.continuous_columns`` and/or
``data.categorical_columns`` are strictly required.

.. note::

   Default values shown below are from the base profile template. Python
   dataclass defaults (used when instantiating ``PhenoClusterConfig``
   programmatically without a profile) may differ for some parameters.

Generate a starter config from any profile (see :doc:`profiles`):

.. code-block:: bash

   phenocluster create-config -p <profile> -o config.yaml

global
------

Project-level settings.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``project_name``
     - str
     - ``"PhenoCluster"``
     - Project identifier shown in report titles and headers
   * - ``output_dir``
     - str
     - ``"results"``
     - Directory where all output files, plots, and cached artifacts are written
   * - ``random_state``
     - int
     - ``42``
     - Global random seed, automatically propagated to model selection, data splitting, and feature selection for full reproducibility
   * - ``generate_html_report``
     - bool
     - ``true``
     - Whether to render the static HTML analysis report at the end of a run. JSON, CSV, and Plotly figure outputs are written either way. Can be overridden at the command line with ``phenocluster run --html-report`` / ``--no-html-report``.

data
----

Dataset schema and train/test splitting.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``continuous_columns``
     - list[str]
     - ``[]``
     - Names of continuous (numeric) feature columns used for phenotype discovery
   * - ``categorical_columns``
     - list[str]
     - ``[]``
     - Names of categorical (discrete) feature columns used for phenotype discovery
   * - ``split.strategy``
     - str
     - ``"random"``
     - Splitting strategy. One of ``random``, ``temporal``, ``holdout_group``, ``leave_one_group_out``
   * - ``split.test_size``
     - float
     - ``0.2``
     - Fraction of data held out for testing (0 to 1 exclusive). Used by ``random``
   * - ``split.stratify_by``
     - str | null
     - ``null``
     - Column name to stratify the train/test split by (ensures balanced representation); ``null`` for random split. Used by ``random``
   * - ``split.shuffle``
     - bool
     - ``true``
     - Whether to shuffle the data before splitting. Used by ``random``
   * - ``split.time_column``
     - str | null
     - ``null``
     - Name of the date/datetime column. Required when ``strategy="temporal"``
   * - ``split.time_scheme``
     - str
     - ``"cutoff"``
     - Sub-strategy for ``temporal``. One of ``cutoff``, ``fraction``, ``sliding``, ``expanding``
   * - ``split.time_cutoff``
     - str | null
     - ``null``
     - Cutoff timestamp; rows with ``time_column <= time_cutoff`` form the derivation set. Required when ``time_scheme="cutoff"``
   * - ``split.time_test_fraction``
     - float | null
     - ``null``
     - Fraction (0 to 1 exclusive) of the most recent rows used for validation. Required when ``time_scheme="fraction"``
   * - ``split.n_windows``
     - int | null
     - ``null``
     - Number of validation windows (must be >= 2). Required when ``time_scheme`` is ``sliding`` or ``expanding``
   * - ``split.group_column``
     - str | null
     - ``null``
     - Column whose values define the groups. Required for ``holdout_group`` and ``leave_one_group_out``
   * - ``split.holdout_values``
     - list | null
     - ``null``
     - Group values placed in the validation set; all other groups go to derivation. Required for ``holdout_group``
   * - ``split.min_validation_size``
     - int
     - ``25``
     - Minimum number of rows a validation cohort must contain; smaller cohorts raise an error

.. note::

   The train/test split is performed **before** any preprocessing.
   Imputation, outlier handling, encoding, and scaling are fit on the
   training set only for model selection. Once K is chosen, the full
   pipeline is refitted on the entire cohort for final analysis.

.. note::

   Legacy configurations that omit ``split.strategy`` keep working unchanged:
   ``random`` is the default and the only fields it consults are
   ``test_size``, ``stratify_by``, and ``shuffle``.

preprocessing.row_filter
------------------------

Row-level missing data filtering, applied before any imputation.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable row filtering
   * - ``max_missing_pct``
     - float
     - ``0.30``
     - Maximum fraction of missing values allowed per row; rows exceeding this threshold are dropped

preprocessing.imputation
------------------------

Missing data imputation for remaining missing values after row filtering.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Enable imputation. When disabled, StepMix handles missing values natively via FIML
   * - ``method``
     - str
     - ``"iterative"``
     - Imputation strategy: ``iterative`` (multivariate chained equations), ``knn`` (k-nearest neighbours), ``simple`` (mean/mode)
   * - ``estimator``
     - str
     - ``"bayesian_ridge"``
     - Regression estimator for iterative imputation: ``bayesian_ridge`` or ``random_forest``
   * - ``max_iter``
     - int
     - ``10``
     - Maximum number of imputation rounds (iterative method only)

preprocessing.categorical_encoding
-----------------------------------

Categorical variable encoding applied before LCA/LPA.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``method``
     - str
     - ``"label"``
     - Encoding strategy: ``label`` (ordinal integers), ``onehot`` (one-hot dummy variables), ``frequency`` (replace categories with their frequency)
   * - ``handle_unknown``
     - str
     - ``"ignore"``
     - Behaviour when encountering unseen categories at test time: ``ignore`` produces zeros (one-hot) or maps to mode (label)

preprocessing.outlier
---------------------

Outlier detection and handling for continuous features.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable outlier handling
   * - ``method``
     - str
     - ``"winsorize"``
     - Strategy: ``winsorize`` (clip extreme values to percentile bounds) or ``isolation_forest`` (detect and remove anomalous observations)
   * - ``contamination``
     - float | "auto"
     - ``"auto"``
     - Expected proportion of outliers in the data (isolation forest only); ``"auto"`` lets the algorithm decide
   * - ``winsorize_limits``
     - [float, float]
     - ``[0.01, 0.01]``
     - Lower and upper percentile bounds for winsorization (e.g., ``[0.01, 0.01]`` clips the bottom 1% and top 1%)

preprocessing.feature_selection
-------------------------------

Optional feature selection to reduce dimensionality before LCA/LPA.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Enable feature selection
   * - ``method``
     - str
     - ``"lasso"``
     - Selection method: ``variance`` (remove low-variance features), ``correlation`` (remove highly correlated pairs), ``mutual_info`` (rank by mutual information with target), ``lasso`` (L1-penalised logistic regression), ``combined`` (apply all filters sequentially)
   * - ``variance_threshold``
     - float
     - ``0.01``
     - Minimum variance required to keep a feature (variance method)
   * - ``frequency_threshold``
     - float
     - ``0.99``
     - Drop features where a single value accounts for more than this fraction of observations
   * - ``correlation_threshold``
     - float
     - ``0.9``
     - Maximum allowed pairwise Pearson correlation; one feature from each correlated pair is dropped
   * - ``n_features``
     - int | null
     - ``null``
     - Target number of features to select (mutual info and lasso methods); ``null`` uses method-specific defaults
   * - ``percentile``
     - float
     - ``50.0``
     - Percentile threshold for feature ranking (mutual info method)
   * - ``lasso_alpha``
     - float | null
     - ``null``
     - L1 regularisation strength for lasso; ``null`` selects automatically via cross-validation
   * - ``target_column``
     - str | null
     - ``null``
     - Target column name required by supervised methods (``mutual_info``, ``lasso``). When this column is also an outcome (``outcome.outcome_columns``) or a survival ``time_column``/``event_column``, the pipeline emits a warning at validation time because supervised feature selection then biases cluster-vs-outcome estimates toward optimistic associations.
   * - ``error_on_outcome_collision``
     - bool
     - ``false``
     - When ``true``, the warning above is promoted to a hard ``ValueError``. Leave at ``false`` for sensitivity analyses that intentionally mix the target with an outcome.

model
-----

Latent Class / Profile Analysis model parameters and automatic selection.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``n_clusters``
     - int
     - ``3``
     - Fixed number of latent classes; only used when ``selection.enabled`` is ``false``
   * - ``selection.enabled``
     - bool
     - ``true``
     - Enable automatic model selection by searching over a range of cluster counts
   * - ``selection.min_clusters``
     - int
     - ``2``
     - Minimum number of clusters to evaluate during model selection
   * - ``selection.max_clusters``
     - int
     - ``6``
     - Maximum number of clusters to evaluate during model selection
   * - ``selection.criterion``
     - str
     - ``"BIC"``
     - Information criterion used to rank models: ``BIC``, ``AIC``, ``ICL``, ``CAIC``, ``SABIC``, ``ENTROPY``
   * - ``selection.min_cluster_size``
     - int | float
     - ``0.05``
     - Minimum acceptable cluster size; integer for absolute count, float in (0, 1) for proportion of total samples. Models with any cluster below this threshold are rejected
   * - ``selection.n_init``
     - list[int]
     - ``[100]``
     - Number of random EM initialisations per cluster count to avoid local optima
   * - ``selection.n_jobs``
     - int
     - ``-1``
     - Number of parallel jobs; ``-1`` uses all available CPU cores
   * - ``selection.refit``
     - bool
     - ``true``
     - Refit best model on full training data after selection
   * - ``stepmix.max_iter``
     - int
     - ``1000``
     - Maximum number of EM algorithm iterations per fit
   * - ``stepmix.abs_tol``
     - float
     - ``1e-7``
     - Absolute convergence tolerance for the EM log-likelihood
   * - ``stepmix.rel_tol``
     - float
     - ``1e-5``
     - Relative convergence tolerance for the EM log-likelihood

outcome
-------

Binary outcome association analysis. When enabled, a logistic regression
is fitted for each outcome column, comparing each phenotype against the reference.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable outcome association analysis
   * - ``outcome_columns``
     - list[str]
     - ``[]``
     - Names of binary (0/1) outcome columns in the dataset

stability
---------

Consensus clustering stability analysis via repeated subsampled LCA/LPA fits.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable stability analysis
   * - ``n_runs``
     - int
     - ``100``
     - Number of subsampled LCA/LPA fits; higher values give more reliable stability estimates
   * - ``subsample_fraction``
     - float
     - ``0.8``
     - Fraction of training data randomly sampled for each run
   * - ``n_jobs``
     - int
     - ``-1``
     - Number of parallel jobs; ``-1`` uses all available CPU cores

survival
--------

Survival analysis with Kaplan-Meier curves, Nelson-Aalen estimators, log-rank
tests, and Cox PH hazard ratios.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable survival analysis
   * - ``use_weighted``
     - bool
     - ``false``
     - Weight survival curves by class membership probabilities instead of hard assignments
   * - ``targets``
     - list
     - ``[]``
     - List of survival endpoints, each with ``name`` (label), ``time_column`` (follow-up duration), and ``event_column`` (censoring indicator, 1 = event)

multistate
----------

Multistate transition modelling with transition-specific Cox PH models,
hazard ratios, and Monte Carlo trajectory simulation.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Enable multistate analysis
   * - ``states``
     - list
     - ``[]``
     - State definitions, each with ``id`` (integer), ``name``, ``state_type`` (``initial``, ``transient``, ``absorbing``), and optionally ``event_column`` / ``time_column`` for transient states
   * - ``transitions``
     - list
     - ``[]``
     - Allowed transitions between states, each with ``name``, ``from_state`` (id), ``to_state`` (id)
   * - ``baseline_confounders``
     - list[str]
     - ``[]``
     - Column names of baseline covariates to adjust for in the Cox models
   * - ``min_events_per_transition``
     - int
     - ``3``
     - Minimum observed events required to fit a model for a given transition
   * - ``default_followup_time``
     - float
     - ``30``
     - Maximum follow-up horizon (in the same time unit as your data) for Monte Carlo simulation
   * - ``monte_carlo.n_simulations``
     - int
     - ``1000``
     - Number of Monte Carlo patient trajectories simulated per phenotype
   * - ``monte_carlo.time_points``
     - list[float]
     - ``[5,10,15,20,25,30]``
     - Time points at which state occupation probabilities are evaluated
   * - ``monte_carlo.max_transitions_per_path``
     - int
     - ``10``
     - Safety limit on the maximum number of transitions in a single simulated trajectory

inference
---------

Statistical inference settings for outcome, survival, and multistate analyses.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable statistical inference (logistic regression, Cox PH, log-rank tests)
   * - ``confidence_level``
     - float
     - ``0.95``
     - Width of confidence intervals (e.g., 0.95 for 95% CI)
   * - ``fdr_correction``
     - bool
     - ``true``
     - Apply Benjamini-Hochberg FDR correction for multiple comparisons
   * - ``outcome_test``
     - str
     - ``"auto"``
     - Test for binary outcomes: ``auto`` (selects based on expected cell counts), ``chi-square``, ``fisher``
   * - ``cox_penalizer``
     - float
     - ``0.0``
     - L2 penalizer for Cox PH models (survival and multistate analyses); helps with convergence when events are sparse


reference_phenotype
-------------------

Strategy for selecting the reference phenotype against which all other
phenotypes are compared in outcome and survival analyses.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``strategy``
     - str
     - ``"largest"``
     - Selection strategy: ``largest`` (most patients), ``healthiest`` (lowest rate of a specified outcome), ``specific`` (user-specified ID)
   * - ``specific_id``
     - int | null
     - ``null``
     - Phenotype ID to use as reference; required when ``strategy`` is ``specific``
   * - ``health_outcome``
     - str | null
     - ``null``
     - Outcome column name used to determine the healthiest phenotype; required when ``strategy`` is ``healthiest``

external_validation
-------------------

External validation on an independent cohort. When enabled, the fitted model
is applied to an external dataset to assess phenotype reproducibility.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Enable external validation on an independent cohort
   * - ``external_data_path``
     - str | null
     - ``null``
     - Path to the external cohort CSV file

generalizability
----------------

Temporal and multi-site generalizability assessment (v0.3.0). See
:doc:`generalizability` for the full reference; the table below lists the
top-level toggles. Sub-blocks ``temporal``, ``multisite``,
``external_cohorts``, ``calibration``, ``drift``, and
``outcome_concordance`` are documented in detail there.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``false``
     - Run the generalizability stage. Requires at least one of ``temporal``, ``multisite``, or ``external_cohorts`` to be set.
   * - ``training_scope``
     - ``"per_split"`` | ``"global"``
     - ``"per_split"``
     - For each in-CSV (derivation, validation) split, ``per_split`` fits a fresh preprocessor and StepMix on derivation rows only. ``global`` reuses the pipeline's full-cohort model. External-CSV cohorts always use the global model.
   * - ``feature_selector_scope``
     - ``"auto"`` | ``"global"`` | ``"per_split"``
     - ``"auto"``
     - Whether to refit the feature selector per split. ``auto`` refits when safe (unsupervised methods, or supervised methods whose target column does not collide with concordance outcomes) and reuses the global selector with a warning otherwise. ``per_split`` raises when a safe refit is not possible. ``global`` always reuses the pipeline's selector. Each cohort report carries a ``feature_selector_mode`` field reflecting the resolved choice.
   * - ``refit``
     - bool
     - ``true``
     - When ``true``, refit StepMix on the validation cohort and Hungarian-align with the derivation labels (yields ARI / NMI / matched accuracy plus calibration). When ``false``, apply-only.
   * - ``min_validation_size_for_refit``
     - int
     - ``100``
     - Skip the validation-side refit when the validation cohort has fewer rows than this.
   * - ``temporal``
     - object | null
     - ``null``
     - Temporal split spec. Fields: ``time_column``, ``scheme`` (``cutoff`` | ``fraction`` | ``sliding`` | ``expanding``), ``time_cutoff``, ``time_test_fraction``, ``n_windows``.
   * - ``multisite``
     - object | null
     - ``null``
     - Multi-site split spec. Fields: ``site_column``, ``scheme`` (``logo`` | ``holdout`` | ``pairwise``), ``holdout_sites``, ``min_site_size``.
   * - ``external_cohorts``
     - list of objects
     - ``[]``
     - Each entry: ``{path, label, kind}`` with ``kind`` in ``{"temporal", "site", "external"}``.
   * - ``calibration``
     - object
     - ``{enabled: true, n_bins: 10, strategy: "quantile"}``
     - Brier / ECE / reliability-curve settings (refit-and-match mode only).
   * - ``drift``
     - object
     - ``{enabled: true, n_bins: 10, top_k: 20}``
     - Per-feature drift table (PSI, KS, chi-square).
   * - ``outcome_concordance``
     - object
     - ``{enabled: true, fdr_method: "bh", alpha: 0.05}``
     - Cross-cohort OR/HR concordance with FDR-corrected per-phenotype delta tests.

cache
-----

Artifact caching for incremental re-runs. Cached artifacts allow skipping
completed pipeline steps when re-running with the same data and config.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable caching of intermediate pipeline results to ``artifacts/``
   * - ``compress_level``
     - int
     - ``3``
     - Gzip compression level for cached files (0 = no compression, 9 = maximum)

visualization
-------------

Plot output settings.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``save_plots``
     - bool
     - ``true``
     - Save generated plots to the output directory
   * - ``dpi``
     - int
     - ``300``
     - Resolution in dots per inch for raster plot formats

logging
-------

Logging configuration.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``level``
     - str
     - ``"INFO"``
     - Minimum log level: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
   * - ``format``
     - str
     - ``"detailed"``
     - Log message format: ``minimal`` (message only), ``standard`` (level + message), ``detailed`` (timestamp + level + module + message)
   * - ``log_to_file``
     - bool
     - ``true``
     - Write log messages to a file in the output directory
   * - ``log_file``
     - str
     - ``"phenocluster.log"``
     - Name of the log file
   * - ``quiet_mode``
     - bool
     - ``false``
     - Suppress all console output (log file is still written if enabled)

data_quality
------------

Automated data quality assessment run before preprocessing.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``enabled``
     - bool
     - ``true``
     - Enable data quality checks
   * - ``missing_threshold``
     - float
     - ``0.15``
     - Flag columns with more than this fraction of missing values in the report
   * - ``correlation_threshold``
     - float
     - ``0.9``
     - Flag feature pairs with Pearson correlation exceeding this value
   * - ``variance_threshold``
     - float
     - ``0.01``
     - Flag features with variance below this value as near-constant
   * - ``generate_report``
     - bool
     - ``true``
     - Include a data quality summary section in the HTML report

categorical_flow
----------------

Categorical variable flow visualisation settings.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``group_by_prefix``
     - bool
     - ``true``
     - Group variables by name prefix for cleaner visualisation
   * - ``prefix_separator``
     - str
     - ``"_"``
     - Character used to split variable names into prefix groups
   * - ``custom_groups``
     - dict
     - ``{}``
     - Manual variable groupings (e.g., ``Recipient: [R_*]``)
   * - ``show_sankey``
     - bool
     - ``false``
     - Show Sankey diagrams (can be cluttered with many variables)
   * - ``show_proportion_heatmap``
     - bool
     - ``true``
     - Show proportion heatmap (recommended)
   * - ``min_category_pct``
     - float
     - ``0.03``
     - Group categories below this proportion as "Other"

feature_characterization
------------------------

Descriptive feature characterisation settings for the report.

.. list-table::
   :header-rows: 1
   :widths: 25 10 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``group_by_prefix``
     - bool
     - ``true``
     - Group features by name prefix
   * - ``prefix_separator``
     - str
     - ``"_"``
     - Character used to split feature names into prefix groups
   * - ``custom_groups``
     - dict
     - ``{}``
     - Manual feature groupings (e.g., ``Recipient: [R_*]``)
   * - ``n_top_per_group``
     - int
     - ``5``
     - Number of top features to show per group per cluster
   * - ``n_top_overall``
     - int
     - ``20``
     - Number of top features overall (when grouping is disabled)
