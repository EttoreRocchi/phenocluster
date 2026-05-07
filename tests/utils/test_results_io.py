"""Tests for the pipeline results I/O module."""

import json
import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from phenocluster.config import PhenoClusterConfig
from phenocluster.utils import results_io
from phenocluster.utils.results_io import (
    _numpy_encoder,
    _save_artifacts,
    _save_data_files,
    _save_json_results,
    _save_model_selection,
    _save_plots,
    save_pipeline_results,
)


def _logger():
    """Create a stubbed logger that records calls without producing output."""
    log = MagicMock(spec=logging.Logger)
    log.info = MagicMock()
    log.warning = MagicMock()
    return log


def _make_config(tmp_path):
    """Build a minimal config pointing output_dir to tmp_path."""
    return PhenoClusterConfig.from_dict(
        {
            "global": {
                "project_name": "io_test",
                "output_dir": str(tmp_path),
                "random_state": 42,
            },
            "data": {"continuous_columns": ["x"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


class TestNumpyEncoder:
    def test_numpy_int32(self):
        assert _numpy_encoder(np.int32(3)) == 3

    def test_numpy_float_nan_to_none(self):
        assert _numpy_encoder(np.float64("nan")) is None

    def test_numpy_float_inf_to_none(self):
        assert _numpy_encoder(np.float64("inf")) is None


class TestSaveDataFiles:
    def test_writes_phenotypes_data(self, tmp_path):
        results = {
            "data": pd.DataFrame({"a": [1, 2]}),
            "n_clusters": 2,
        }
        _save_data_files(results, tmp_path, _logger())
        assert (tmp_path / "phenotypes_data.csv").exists()

    def test_writes_posterior_probabilities(self, tmp_path):
        results = {
            "n_clusters": 3,
            "posterior_proba": np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        }
        _save_data_files(results, tmp_path, _logger())
        df = pd.read_csv(tmp_path / "posterior_probabilities.csv")
        assert list(df.columns) == ["Cluster_0_Prob", "Cluster_1_Prob", "Cluster_2_Prob"]

    def test_writes_model_fit_metrics(self, tmp_path):
        results = {"model_fit_metrics": {"BIC": 100.0, "AIC": 90.0}}
        _save_data_files(results, tmp_path, _logger())
        assert (tmp_path / "model_fit_metrics.csv").exists()

    def test_no_files_when_keys_absent(self, tmp_path):
        _save_data_files({}, tmp_path, _logger())
        assert list(tmp_path.iterdir()) == []


class TestSaveJsonResults:
    def test_writes_each_json_pair(self, tmp_path):
        results = {
            "cluster_stats": {"k": 1},
            "outcome_results": {"a": 1},
            "survival_results": {"b": 2},
            "multistate_results": {"c": 3},
            "feature_importance": {"d": 4},
            "split_info": {"train": 80},
            "classification_quality": {"avepp": 0.9},
            "classification_quality_test": {"avepp": 0.85},
            "feature_selection": {"selected": ["a"]},
        }
        _save_json_results(results, tmp_path, reference_phenotype=0, logger=_logger())
        for fname in (
            "cluster_statistics.json",
            "outcome_results.json",
            "survival_results.json",
            "multistate_results.json",
            "feature_importance.json",
            "split_info.json",
            "classification_quality.json",
            "classification_quality_test.json",
            "feature_selection.json",
        ):
            assert (tmp_path / fname).exists()

    def test_skips_empty_results(self, tmp_path):
        _save_json_results({"cluster_stats": {}}, tmp_path, reference_phenotype=0, logger=_logger())
        assert not (tmp_path / "cluster_statistics.json").exists()

    def test_writes_stability_with_consensus_matrix(self, tmp_path):
        results = {
            "stability_results": {
                "ari": 0.8,
                "consensus_matrix": np.eye(3),
            }
        }
        _save_json_results(results, tmp_path, reference_phenotype=0, logger=_logger())
        assert (tmp_path / "stability_results.json").exists()
        assert (tmp_path / "consensus_matrix.npy").exists()
        loaded = np.load(tmp_path / "consensus_matrix.npy")
        assert loaded.shape == (3, 3)

    def test_validation_report_includes_reference(self, tmp_path):
        results = {"validation_metrics": {"ari": 0.7}}
        _save_json_results(results, tmp_path, reference_phenotype=2, logger=_logger())
        with open(tmp_path / "validation_report.json") as f:
            payload = json.load(f)
        assert payload["reference_phenotype"] == 2
        assert payload["ari"] == 0.7

    def test_external_validation_excludes_plots(self, tmp_path):
        results = {
            "external_validation_results": {
                "n_samples": 50,
                "plots": {"fig": object()},
            }
        }
        _save_json_results(results, tmp_path, reference_phenotype=0, logger=_logger())
        with open(tmp_path / "external_validation_results.json") as f:
            payload = json.load(f)
        assert "plots" not in payload
        assert payload["n_samples"] == 50


class TestSaveModelSelection:
    def test_no_op_when_missing(self, tmp_path):
        _save_model_selection({}, tmp_path, tmp_path, _logger())
        assert list(tmp_path.iterdir()) == []

    def test_writes_summary_only_when_no_table(self, tmp_path):
        results = {
            "model_selection": {
                "criterion_used": "BIC",
                "best_n_clusters": 3,
                "all_results": [{"n_clusters": 3, "BIC": 1234.5}],
            }
        }
        _save_model_selection(results, tmp_path, tmp_path, _logger())
        with open(tmp_path / "model_selection_summary.json") as f:
            payload = json.load(f)
        assert payload["best_n_clusters"] == 3
        assert payload["best_criterion_value"] == 1234.5

    def test_writes_comparison_table_with_sign_flip(self, tmp_path):
        comp = pd.DataFrame(
            {
                "n_clusters": [2, 3],
                "mean_score": [-100.0, -90.0],
                "rank": [2, 1],
            }
        )
        results = {
            "model_selection": {
                "criterion_used": "BIC",
                "best_n_clusters": 3,
                "all_results": [{"n_clusters": 3, "BIC": 90.0}],
                "comparison_table": comp,
                "cv_results": {"some": "value"},
            }
        }
        _save_model_selection(results, tmp_path, tmp_path, _logger())
        df = pd.read_csv(tmp_path / "model_selection_results.csv")
        assert df["mean_score"].tolist() == [100.0, 90.0]


class TestSaveArtifacts:
    def test_saves_config_yaml(self, tmp_path):
        cfg = _make_config(tmp_path)
        _save_artifacts(
            cfg, preprocessor=None, feature_selector=None, artifacts_dir=tmp_path, logger=_logger()
        )
        assert (tmp_path / "config_used.yaml").exists()

    def test_saves_feature_selector(self, tmp_path):
        cfg = _make_config(tmp_path)
        selector = MagicMock()
        _save_artifacts(
            cfg,
            preprocessor=None,
            feature_selector=selector,
            artifacts_dir=tmp_path,
            logger=_logger(),
        )
        assert (tmp_path / "feature_selector.joblib").exists()

    def test_saves_preprocessor_with_label_encoders(self, tmp_path):
        cfg = _make_config(tmp_path)
        prep = MagicMock()
        prep.label_encoders = {"cat": "encoder_obj"}
        _save_artifacts(
            cfg, preprocessor=prep, feature_selector=None, artifacts_dir=tmp_path, logger=_logger()
        )
        assert (tmp_path / "preprocessor.joblib").exists()
        assert (tmp_path / "label_encoders.joblib").exists()

    def test_warns_on_preprocessor_failure(self, tmp_path):
        cfg = _make_config(tmp_path)
        prep = MagicMock()
        prep.label_encoders = {}
        log = _logger()
        log_save_count = []

        def failing_save(*_args, **_kwargs):
            log_save_count.append(1)
            raise RuntimeError("cannot pickle")

        original_dump = results_io.joblib.dump
        results_io.joblib.dump = failing_save
        try:
            _save_artifacts(
                cfg, preprocessor=prep, feature_selector=None, artifacts_dir=tmp_path, logger=log
            )
        finally:
            results_io.joblib.dump = original_dump
        log.warning.assert_called()


class TestSavePlots:
    def test_writes_html_per_plot(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.visualization.save_plots = True
        fig = MagicMock()
        results = {"plots": {"summary": fig}}
        _save_plots(results, tmp_path, cfg, _logger())
        fig.write_html.assert_called_once()

    def test_skips_when_save_plots_disabled(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.visualization.save_plots = False
        fig = MagicMock()
        _save_plots({"plots": {"summary": fig}}, tmp_path, cfg, _logger())
        fig.write_html.assert_not_called()

    def test_skips_none_figures(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.visualization.save_plots = True
        log = _logger()
        _save_plots({"plots": {"empty": None}}, tmp_path, cfg, log)
        assert not list(tmp_path.iterdir())

    def test_external_validation_plots_saved(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.visualization.save_plots = True
        fig = MagicMock()
        results = {"external_validation_results": {"plots": {"ext": fig}}}
        _save_plots(results, tmp_path, cfg, _logger())
        fig.write_html.assert_called_once()


class TestSavePipelineResults:
    def test_warns_and_returns_for_empty_results(self, tmp_path):
        cfg = _make_config(tmp_path)
        log = _logger()
        save_pipeline_results({}, cfg, None, None, 0, log, output_dir=str(tmp_path))
        log.warning.assert_called()
        assert not (tmp_path / "data").exists()

    def test_full_run_creates_directory_layout(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.visualization.save_plots = False
        results = {
            "data": pd.DataFrame({"a": [1]}),
            "n_clusters": 2,
            "cluster_stats": {"k": 1},
        }
        save_pipeline_results(results, cfg, None, None, 0, _logger(), output_dir=str(tmp_path))
        for sub in ("data", "results", "plots", "artifacts"):
            assert (tmp_path / sub).is_dir()
        assert (tmp_path / "data" / "phenotypes_data.csv").exists()
        assert (tmp_path / "results" / "cluster_statistics.json").exists()
        assert (tmp_path / "artifacts" / "config_used.yaml").exists()

    def test_uses_config_output_dir_when_none(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.visualization.save_plots = False
        results = {"data": pd.DataFrame({"a": [1]}), "n_clusters": 2}
        save_pipeline_results(results, cfg, None, None, 0, _logger())
        assert (tmp_path / "data").is_dir()


class TestGenerateReport:
    def test_failure_logs_warning(self, tmp_path, monkeypatch):
        cfg = _make_config(tmp_path)

        def boom(*_args, **_kwargs):
            raise RuntimeError("template missing")

        monkeypatch.setattr("phenocluster.utils.report.generate_html_report", boom)
        log = _logger()
        results_io._generate_report(tmp_path, cfg, log)
        log.warning.assert_called()
