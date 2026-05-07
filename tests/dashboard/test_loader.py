"""Tests for the dashboard loader's tolerance to missing artifacts."""

import json
from pathlib import Path

import pandas as pd
import pytest

from phenocluster.dashboard.loader import DashboardData, load_results


def _seed_minimal(results_dir: Path) -> None:
    """Create the minimal directory layout with a few JSON/CSV artifacts."""
    (results_dir / "results").mkdir(parents=True)
    (results_dir / "data" / "generalizability").mkdir(parents=True)
    (results_dir / "plots").mkdir(parents=True)
    (results_dir / "artifacts").mkdir(parents=True)

    cluster_stats = {"sizes": {"0": 50, "1": 30}}
    with open(results_dir / "results" / "cluster_statistics.json", "w") as f:
        json.dump(cluster_stats, f)

    temporal = [
        {
            "label": "fraction=0.20",
            "kind": "temporal",
            "n_samples": 40,
            "log_likelihood": -123.4,
            "cluster_distribution": {"0": {"count": 20, "percentage": 50.0}},
        }
    ]
    with open(results_dir / "results" / "temporal_validation_results.json", "w") as f:
        json.dump(temporal, f)

    drift_df = pd.DataFrame({"feature": ["x", "y"], "kind": ["continuous"] * 2, "psi": [0.1, 0.05]})
    drift_df.to_csv(
        results_dir / "data" / "generalizability" / "drift_fraction-0.20.csv", index=False
    )

    (results_dir / "plots" / "kaplan_meier_demo.html").write_text("<html></html>")
    (results_dir / "plots" / "kaplan_meier_demo.json").write_text("{}")


def test_loader_parses_minimal_layout(tmp_path):
    _seed_minimal(tmp_path)
    data = load_results(tmp_path)
    assert isinstance(data, DashboardData)
    assert data.cluster_stats == {"sizes": {"0": 50, "1": 30}}
    assert len(data.temporal_validation_results) == 1
    assert data.temporal_validation_results[0]["label"] == "fraction=0.20"
    assert "fraction-0.20" in data.drift_tables
    assert len(data.plot_files) == 1
    assert len(data.plot_json_files) == 1
    assert data.plot_json_files[0].stem == "kaplan_meier_demo"


def test_loader_missing_dir_raises(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_results(missing)


def test_loader_returns_empty_optional_fields(tmp_path):
    (tmp_path / "results").mkdir()
    data = load_results(tmp_path)
    assert data.cluster_stats is None
    assert data.temporal_validation_results == []
    assert data.multisite_validation_results == []
    assert data.drift_tables == {}
    assert data.plot_files == []
    assert data.plot_json_files == []
