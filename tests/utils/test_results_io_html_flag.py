"""Tests for the v0.3.0 generate_html_report config flag and CLI override."""

from unittest.mock import MagicMock

from phenocluster.config import PhenoClusterConfig
from phenocluster.utils.results_io import save_pipeline_results


def _config(**kwargs):
    cfg = PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "t", "random_state": 42},
            "data": {
                "continuous_columns": ["c1"],
                "categorical_columns": [],
                "split": {"test_size": 0.2},
            },
            "preprocessing": {
                "categorical_encoding": {"method": "label"},
                "imputation": {"enabled": False},
                "outlier": {"enabled": False},
            },
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _stub_results():
    import numpy as np
    import pandas as pd

    return {
        "data": pd.DataFrame({"x": [0, 1]}),
        "posterior_proba": np.array([[0.6, 0.4], [0.3, 0.7]]),
        "n_clusters": 2,
        "model_fit_metrics": {"BIC": 1.0},
        "split_info": {"train_size": 1, "test_size": 1},
    }


def test_generate_html_report_false_skips_report(tmp_path):
    cfg = _config(generate_html_report=False, output_dir=str(tmp_path))
    logger = MagicMock()
    save_pipeline_results(
        _stub_results(),
        cfg,
        preprocessor=None,
        feature_selector=None,
        reference_phenotype=0,
        logger=logger,
        output_dir=str(tmp_path),
    )
    assert not (tmp_path / "analysis_report.html").exists()
    skip_messages = [c.args[0] for c in logger.info.call_args_list if "disabled" in str(c.args[0])]
    assert any("HTML report generation disabled" in m for m in skip_messages)


def test_generate_html_report_true_attempts_report(tmp_path):
    cfg = _config(generate_html_report=True, output_dir=str(tmp_path))
    logger = MagicMock()
    save_pipeline_results(
        _stub_results(),
        cfg,
        preprocessor=None,
        feature_selector=None,
        reference_phenotype=0,
        logger=logger,
        output_dir=str(tmp_path),
    )
    skip_messages = [c.args[0] for c in logger.info.call_args_list if "disabled" in str(c.args[0])]
    assert not skip_messages
