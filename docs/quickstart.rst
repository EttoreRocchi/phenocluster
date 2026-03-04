Quick Start
===========

This guide walks through a typical PhenoCluster workflow in four steps.

1. Generate a configuration file
---------------------------------

.. code-block:: bash

   phenocluster create-config -p complete -o config.yaml

This creates a fully commented YAML file with sensible defaults from the
``complete`` profile. See :doc:`profiles` for all available profiles.

2. Edit the configuration
--------------------------

Open ``config.yaml`` and fill in your dataset-specific parameters:

.. code-block:: yaml

   global:
     project_name: "My Study"
     output_dir: "results"
     random_state: 42

   data:
     continuous_columns:
       - age
       - bmi
       - lab_value_1
     categorical_columns:
       - sex
       - smoking_status
       - disease_stage
     split:
       test_size: 0.2

   outcome:
     enabled: true
     outcome_columns:
       - mortality_30d
       - readmission_30d

   survival:
     enabled: true
     targets:
       - name: "overall_survival"
         time_column: "time_to_death"
         event_column: "death_indicator"

You can validate the config before running:

.. code-block:: bash

   phenocluster validate-config -c config.yaml -d data.csv

3. Run the pipeline
--------------------

.. code-block:: bash

   phenocluster run -d data.csv -c config.yaml

Use ``--force-rerun`` to ignore cached intermediate results.

4. Inspect results
-------------------

Results are written to the output directory (default ``results/``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``analysis_report.html``
     - Comprehensive HTML report with all results and visualisations
   * - ``cluster_statistics.json``
     - Phenotype sizes, feature distributions, and classification quality
   * - ``outcome_results.json``
     - Odds ratios with confidence intervals and p-values
   * - ``survival_results.json``
     - Kaplan-Meier estimates and Cox PH hazard ratios
   * - ``multistate_results.json``
     - Transition hazard ratios, pathways, and state occupation
   * - ``data/model_fit_metrics.csv``
     - Information criteria, entropy, and posterior probabilities
   * - ``data/phenotypes_data.csv``
     - Original data augmented with phenotype assignments
   * - ``data/posterior_probabilities.csv``
     - Posterior class membership probabilities per patient
   * - ``results/model_selection_summary.json``
     - Model selection comparison table and best model info
   * - ``results/feature_importance.json``
     - Feature characterisation per phenotype
   * - ``results/validation_report.json``
     - Internal validation metrics (train/test comparison)
   * - ``results/stability_results.json``
     - Consensus clustering stability metrics
   * - ``results/split_info.json``
     - Train/test split details
   * - ``results/external_validation_results.json``
     - External validation results (when enabled)
   * - ``phenocluster.log``
     - Pipeline execution log
   * - ``artifacts/``
     - Cached intermediate results for incremental re-runs
