"""
PhenoCluster Dashboard Loader
=============================

Reads pipeline output artifacts (JSON, CSV) from a saved results
directory into a frozen :class:`DashboardData` container. The dashboard
never recomputes; it only renders what the pipeline already saved.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardData:
    """Container for pipeline outputs consumed by the dashboard.

    Every field is optional; the dashboard is expected to gracefully
    skip tabs that lack data.
    """

    results_dir: Path
    config: Optional[Dict[str, Any]] = None
    cluster_stats: Optional[Dict[str, Any]] = None
    model_selection: Optional[Dict[str, Any]] = None
    outcome_results: Optional[Dict[str, Any]] = None
    survival_results: Optional[Dict[str, Any]] = None
    multistate_results: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, Any]] = None
    classification_quality: Optional[Dict[str, Any]] = None
    external_validation_results: Optional[Dict[str, Any]] = None
    temporal_validation_results: List[Dict[str, Any]] = field(default_factory=list)
    multisite_validation_results: List[Dict[str, Any]] = field(default_factory=list)
    external_cohorts_results: List[Dict[str, Any]] = field(default_factory=list)
    generalizability_summary: Optional[Dict[str, Any]] = None
    drift_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    cluster_distribution_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    plot_files: List[Path] = field(default_factory=list)
    plot_json_files: List[Path] = field(default_factory=list)


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s (%s): %s", path, type(exc).__name__, exc)
        return None


def _read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        import yaml
    except ImportError as exc:
        logger.warning("PyYAML not installed; cannot read %s: %s", path, exc)
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Failed to read %s (%s): %s", path, type(exc).__name__, exc)
        return None


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read %s (%s): %s", path, type(exc).__name__, exc)
        return None


def load_results(results_dir: Path) -> DashboardData:
    """Load all known artifacts from ``results_dir`` into a DashboardData.

    Missing files are tolerated; the corresponding fields stay empty so
    the dashboard can hide whichever tabs are not populated.
    """
    results_dir = Path(results_dir).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    results = results_dir / "results"
    data = results_dir / "data"
    plots = results_dir / "plots"

    config = _read_yaml(results_dir / "artifacts" / "config_used.yaml")

    json_load = {
        "cluster_stats": results / "cluster_statistics.json",
        "model_selection": results / "model_selection_summary.json",
        "outcome_results": results / "outcome_results.json",
        "survival_results": results / "survival_results.json",
        "multistate_results": results / "multistate_results.json",
        "feature_importance": results / "feature_importance.json",
        "classification_quality": results / "classification_quality.json",
        "external_validation_results": results / "external_validation_results.json",
        "temporal_validation_results": results / "temporal_validation_results.json",
        "multisite_validation_results": results / "multisite_validation_results.json",
        "external_cohorts_results": results / "external_cohorts_results.json",
        "generalizability_summary": results / "generalizability_summary.json",
    }
    loaded: Dict[str, Any] = {key: _read_json(path) for key, path in json_load.items()}

    drift_tables: Dict[str, pd.DataFrame] = {}
    cluster_distribution_tables: Dict[str, pd.DataFrame] = {}
    gen_data_dir = data / "generalizability"
    if gen_data_dir.exists():
        for csv_path in sorted(gen_data_dir.glob("drift_*.csv")):
            label = csv_path.stem.replace("drift_", "")
            df = _read_csv(csv_path)
            if df is not None:
                drift_tables[label] = df
        for csv_path in sorted(gen_data_dir.glob("cluster_distribution_*.csv")):
            label = csv_path.stem.replace("cluster_distribution_", "")
            df = _read_csv(csv_path)
            if df is not None:
                cluster_distribution_tables[label] = df

    plot_files = sorted(plots.glob("*.html")) if plots.exists() else []
    plot_json_files = sorted(plots.glob("*.json")) if plots.exists() else []

    return DashboardData(
        results_dir=results_dir,
        config=config,
        cluster_stats=loaded["cluster_stats"],
        model_selection=loaded["model_selection"],
        outcome_results=loaded["outcome_results"],
        survival_results=loaded["survival_results"],
        multistate_results=loaded["multistate_results"],
        feature_importance=loaded["feature_importance"],
        classification_quality=loaded["classification_quality"],
        external_validation_results=loaded["external_validation_results"],
        temporal_validation_results=loaded["temporal_validation_results"] or [],
        multisite_validation_results=loaded["multisite_validation_results"] or [],
        external_cohorts_results=loaded["external_cohorts_results"] or [],
        generalizability_summary=loaded["generalizability_summary"],
        drift_tables=drift_tables,
        cluster_distribution_tables=cluster_distribution_tables,
        plot_files=plot_files,
        plot_json_files=plot_json_files,
    )
