"""Sphinx configuration for PhenoCluster documentation."""

import sys
from pathlib import Path

# Add project root to path so sphinx-click can import phenocluster.cli
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import phenocluster  # noqa: E402

project = "PhenoCluster"
author = "Ettore Rocchi"
copyright = "2026, Ettore Rocchi"
release = phenocluster.__version__

extensions = [
    "sphinx_click",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_logo = "phenocluster_logo.png"
html_title = "PhenoCluster"

html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/EttoreRocchi/PhenoCluster",
    "source_branch": "main",
    "source_directory": "docs/",
}
