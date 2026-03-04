"""Report orchestrator: loads results and assembles HTML sections."""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .sections.classification_quality import generate_classification_quality_section
from .sections.clusters import generate_cluster_section
from .sections.data_quality import generate_data_quality_section
from .sections.external_validation import generate_external_validation_section
from .sections.feature_importance import generate_feature_importance_section
from .sections.header import generate_html_footer, generate_html_header, generate_nav_section
from .sections.methods import generate_methods_section
from .sections.model_selection import generate_model_selection_section
from .sections.multistate import generate_multistate_section
from .sections.outcomes import generate_outcomes_section
from .sections.stability import generate_stability_section
from .sections.summary import generate_summary_section
from .sections.survival import generate_survival_section
from .sections.validation import generate_validation_section


def generate_html_report(
    results_dir: Path,
    output_path: Optional[Path] = None,
    title: str = "PhenoCluster Analysis Report",
) -> Path:
    """
    Generate a comprehensive HTML report from analysis results.

    Parameters
    ----------
    results_dir : Path
        Directory containing analysis results
    output_path : Path, optional
        Output path for the report. Defaults to results_dir/report.html
    title : str
        Report title

    Returns
    -------
    Path
        Path to the generated report
    """
    results_dir = Path(results_dir)

    if output_path is None:
        output_path = results_dir / "analysis_report.html"

    data = _load_results(results_dir)

    html_parts = [
        generate_html_header(title),
        generate_nav_section(data),
        '<main class="container">',
        generate_summary_section(data),
        generate_methods_section(data),
        generate_data_quality_section(data),
        generate_model_selection_section(data, results_dir),
        generate_cluster_section(data, results_dir),
        generate_classification_quality_section(data),
        generate_stability_section(data, results_dir),
        generate_validation_section(data),
        generate_outcomes_section(data, results_dir),
        generate_survival_section(data, results_dir),
        generate_multistate_section(data, results_dir),
        generate_feature_importance_section(data),
        generate_external_validation_section(data, results_dir),
        "</main>",
        generate_html_footer(),
    ]

    html_content = "\n".join(html_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def _load_results(results_dir: Path) -> Dict:
    """Load all result files from the results directory."""
    data = {}

    json_files = {
        "config": "artifacts/config_used.yaml",
        "cluster_stats": "results/cluster_statistics.json",
        "model_selection": "results/model_selection_summary.json",
        "outcome_results": "results/outcome_results.json",
        "survival_results": "results/survival_results.json",
        "multistate_results": "results/multistate_results.json",
        "feature_importance": "results/feature_importance.json",
        "data_quality": "quality/data_quality_report.json",
        "split_info": "results/split_info.json",
        "stability_results": "results/stability_results.json",
        "validation_report": "results/validation_report.json",
        "classification_quality": "results/classification_quality.json",
        "classification_quality_test": "results/classification_quality_test.json",
        "external_validation_results": "results/external_validation_results.json",
    }

    for key, filename in json_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                if filename.endswith(".yaml"):
                    import yaml

                    with open(filepath, "r") as f:
                        data[key] = yaml.safe_load(f)
                else:
                    with open(filepath, "r") as f:
                        data[key] = json.load(f)
            except Exception:
                pass

    csv_files = {
        "model_selection_results": "data/model_selection_results.csv",
        "model_fit_metrics": "data/model_fit_metrics.csv",
    }

    for key, filename in csv_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            try:
                data[key] = pd.read_csv(filepath)
            except Exception:
                pass

    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        data["plots"] = {
            "html": list(plots_dir.glob("*.html")),
            "png": list(plots_dir.glob("*.png")),
        }
    else:
        data["plots"] = {"html": [], "png": []}

    return data
