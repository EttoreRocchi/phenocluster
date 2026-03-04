Configuration Profiles
======================

Profiles provide pre-configured defaults for common use-cases. Generate a
config file from a profile with:

.. code-block:: bash

   phenocluster create-config -p <profile> -o config.yaml

Available profiles
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 40 10 10 15

   * - Profile
     - Description
     - Inference
     - Stability
     - Multistate
   * - ``descriptive``
     - Phenotype discovery only, no statistical inference
     - off
     - on
     - off
   * - ``complete``
     - All analyses enabled (outcomes, survival, multistate)
     - on
     - on
     - on
   * - ``quick``
     - Fast iteration for development
     - on
     - off
     - off

Profile details
---------------

descriptive
^^^^^^^^^^^

Designed for exploratory phenotype discovery without statistical inference.
Outcome analysis and inference are disabled. Stability analysis and
survival curves (Kaplan-Meier, Nelson-Aalen) are still computed.

complete
^^^^^^^^

The recommended profile for most analyses. Enables every analysis module:
outcome association (logistic regression), survival analysis (Cox PH,
log-rank), multistate modelling, and stability assessment. FDR correction
is enabled by default.

quick
^^^^^

Designed for fast iteration during development and debugging. Disables the
heaviest optional analyses (stability and multistate) while keeping outcome
association and survival analysis enabled. Not suitable for publication-quality
results.

Customising profiles
--------------------

The generated config file is a complete YAML that you can edit freely. The
profile only sets the initial defaults --- you can override any parameter after
generation. For example, to start from the ``complete`` profile but disable
multistate modelling:

.. code-block:: bash

   phenocluster create-config -p complete -o config.yaml

Then edit ``config.yaml``:

.. code-block:: yaml

   multistate:
     enabled: false
