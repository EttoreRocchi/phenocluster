"""Smoke tests for data-quality figure generators."""

import plotly.graph_objects as go

from phenocluster.config import OutlierConfig, PhenoClusterConfig
from phenocluster.evaluation.data_quality.figures import QualityFigureGenerator


def _config(tmp_path):
    return PhenoClusterConfig(
        continuous_columns=["x"],
        output_dir=str(tmp_path),
        outlier=OutlierConfig(enabled=True),
    )


def _full_report():
    return {
        "missing_data": {
            "by_column": {
                "x1": {"percentage": 5.0, "count": 5},
                "x2": {"percentage": 25.0, "count": 25},
            }
        },
        "outliers": {
            "by_column": {
                "x1": {"percentage": 2.0, "count": 2},
                "x2": {"percentage": 8.0, "count": 8},
            }
        },
        "correlation": {
            "high_correlations": [{"variable1": "x1", "variable2": "x2", "correlation": 0.95}],
            "correlation_matrix": {
                "x1": {"x1": 1.0, "x2": 0.95},
                "x2": {"x1": 0.95, "x2": 1.0},
            },
        },
        "variance": {
            "by_column": {
                "x1": {"variance": 1.5},
                "x2": {"variance": 0.001},
            }
        },
    }


class TestQualityFigureGenerator:
    def test_missing_data_figure(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), _full_report())
        fig = gen.create_missing_data_figure()
        assert isinstance(fig, go.Figure)

    def test_missing_data_empty(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), {})
        assert gen.create_missing_data_figure() is None

    def test_missing_data_no_columns(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), {"missing_data": {"by_column": {}}})
        assert gen.create_missing_data_figure() is None

    def test_outlier_figure(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), _full_report())
        fig = gen.create_outlier_figure()
        assert isinstance(fig, go.Figure)

    def test_outlier_disabled(self, tmp_path):
        cfg = _config(tmp_path)
        cfg.outlier.enabled = False
        assert QualityFigureGenerator(cfg, _full_report()).create_outlier_figure() is None

    def test_outlier_no_data(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), {"outliers": {"by_column": {}}})
        assert gen.create_outlier_figure() is None

    def test_correlation_figure(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), _full_report())
        fig = gen.create_correlation_figure()
        assert isinstance(fig, go.Figure)

    def test_correlation_no_high_corr(self, tmp_path):
        gen = QualityFigureGenerator(
            _config(tmp_path), {"correlation": {"high_correlations": [], "correlation_matrix": {}}}
        )
        assert gen.create_correlation_figure() is None

    def test_variance_figure(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), _full_report())
        fig = gen.create_variance_figure()
        assert isinstance(fig, go.Figure)

    def test_variance_empty(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), {"variance": {"by_column": {}}})
        assert gen.create_variance_figure() is None

    def test_variance_missing(self, tmp_path):
        gen = QualityFigureGenerator(_config(tmp_path), {})
        assert gen.create_variance_figure() is None
