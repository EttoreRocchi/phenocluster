Installation
============

Requirements
------------

- Python >= 3.11
- pip

From PyPI
---------

.. code-block:: bash

   pip install phenocluster

To enable the optional Streamlit dashboard
(:doc:`dashboard`):

.. code-block:: bash

   pip install 'phenocluster[dashboard]'

The core package does **not** depend on Streamlit; the extras pull in
``streamlit`` and ``watchdog`` only when you opt in.

From source
-----------

.. code-block:: bash

   git clone https://github.com/EttoreRocchi/phenocluster.git
   cd phenocluster
   pip install -e ".[dev]"

This installs PhenoCluster in editable mode along with development dependencies
(``pytest``, ``pytest-cov``, ``ruff``).

Building the documentation
--------------------------

The documentation dependencies live in a separate ``docs`` extra:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build docs/ docs/_build/html -W

CI builds the same way, with ``-W`` turning any Sphinx warning into an error.
