Interactive Dashboard
=====================

Version 0.3.0 ships an optional Streamlit dashboard for exploring saved
pipeline outputs interactively. The dashboard reads from a results
directory; it does not re-run the pipeline.

Installation
------------

Streamlit is **not** a runtime dependency of the core package. Install
the optional ``[dashboard]`` extras:

.. code-block:: bash

   pip install 'phenocluster[dashboard]'

This pulls in ``streamlit`` and ``watchdog``. If the extras are not
installed, ``phenocluster dashboard`` exits cleanly with an actionable
install hint.

Launch
------

After running the pipeline once and saving results:

.. code-block:: bash

   phenocluster dashboard ./results/

Options:

- ``--port`` (default ``8501``) - local port to bind.
- ``--host`` (default ``127.0.0.1``) - host interface.
- ``--headless`` / ``--browser`` - whether to auto-open a browser
  (``--headless`` is the default).

Sidebar controls
----------------

The sidebar settings apply across every tab, not per tab:

- **Numeric precision** (2 to 6 decimals, default 3) and **plot height**
  (300 to 1000 px, default 520).
- **FDR threshold** (default 0.05) filters the Outcomes tab and the
  cross-cohort concordance views.
- **PSI threshold** (default 0.10) filters the drift bar charts in the
  Generalizability and Drift explorer tabs.
- **Highlight phenotypes** restricts tables and plots to the selected
  phenotypes.
- **Show cohort warnings** toggles the per-cohort warning panels, which
  carry the v0.4.0 schema check findings.
- **Reset settings** restores all of the above to their defaults.

Tabs
----

Overview
~~~~~~~~

Run summary, pipeline configuration block, model-selection plot, and a
glossary of recurring terms.

Phenotypes
~~~~~~~~~~

Cluster sizes, Average Posterior Probability, relative entropy, and
per-phenotype classification quality.

Outcomes
~~~~~~~~

Per-phenotype odds ratios with confidence intervals, restricted to the
outcome-phenotype pairs whose FDR-adjusted p-value is below the sidebar
FDR threshold.

Survival
~~~~~~~~

Per-target Cox HRs and Kaplan-Meier curves embedded from the saved
plotly HTML files.

Multistate
~~~~~~~~~~

Saved multistate plots (pathway frequency, transition hazards, state
occupation uncertainty, state diagram) and a per-transition table of
hazard ratios when ``multistate_results.json`` is present.

Generalizability
~~~~~~~~~~~~~~~~

Aggregate ARI / PSI summary, a per-cohort table (kind, label, sample
size, log-likelihood, ARI, NMI, matched accuracy, mean PSI, max PSI),
and a cohort-detail view with phenotype distribution, refit metrics,
cohort warnings, and a drift chart showing the features above the
sidebar PSI threshold, capped by a local **Top features** slider.

Drift explorer
~~~~~~~~~~~~~~

Per-cohort drift table viewed in isolation, with its own **Top
features** slider and a feature-kind filter on top of the sidebar PSI
threshold. The full table is available unfiltered in an expander.

Notes
-----

- The dashboard is intentionally **read-only**: it consumes the JSON
  and CSV artifacts written under the ``results/`` directory. To
  re-run the analysis, use ``phenocluster run``.
- The ``phenocluster.dashboard`` Python subpackage is not re-exported
  at the top level; importing it does not pull Streamlit.
