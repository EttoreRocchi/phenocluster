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
   * - ``split.test_size``
     - float
     - ``0.2``
     - Fraction of data held out for testing (0 to 1 exclusive)
   * - ``split.stratify_by``
     - str | null
     - ``null``
     - Column name to stratify the train/test split by (ensures balanced representation); ``null`` for random split
   * - ``split.shuffle``
     - bool
     - ``true``
     - Whether to shuffle the data before splitting

.. note::

   The train/test split is performed **before** any preprocessing.
   Imputation, outlier handling, encoding, and scaling are fit on the
   training set only for model selection. Once K is chosen, the full
   pipeline is refitted on the entire cohort for final analysis.

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
     - Target column name required by supervised methods (``mutual_info``, ``lasso``)

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
