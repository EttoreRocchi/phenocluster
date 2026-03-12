"""Tests for HTML report generation."""

import json

from phenocluster.utils.report._helpers import (
    fmt_p,
    format_ci,
    format_value,
    safe,
)
from phenocluster.utils.report.generator import generate_html_report


def _setup_results_dir(tmp_path):
    """Create a minimal results directory with enough files for report generation."""
    results = tmp_path / "results"
    results.mkdir()
    (tmp_path / "quality").mkdir()
    (tmp_path / "plots").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "data").mkdir()

    # cluster_statistics.json
    (results / "cluster_statistics.json").write_text(
        json.dumps(
            {
                "n_clusters": 3,
                "cluster_sizes": {"0": 30, "1": 25, "2": 20},
            }
        )
    )

    # split_info.json
    (results / "split_info.json").write_text(
        json.dumps(
            {
                "train_size": 60,
                "test_size": 15,
                "n_features": 5,
            }
        )
    )

    # model_selection_summary.json
    (results / "model_selection_summary.json").write_text(
        json.dumps(
            {
                "best_n_clusters": 3,
                "criterion": "BIC",
                "best_score": -500.0,
            }
        )
    )

    return tmp_path


class TestGenerateHtmlReport:
    def test_creates_file(self, tmp_path):
        results_dir = _setup_results_dir(tmp_path)
        out = results_dir / "report.html"
        result_path = generate_html_report(results_dir, output_path=out)
        assert result_path.exists()

    def test_contains_sections(self, tmp_path):
        results_dir = _setup_results_dir(tmp_path)
        out = results_dir / "report.html"
        generate_html_report(results_dir, output_path=out)
        content = out.read_text()
        assert "<!DOCTYPE html>" in content or "<html" in content

    def test_default_output_path(self, tmp_path):
        results_dir = _setup_results_dir(tmp_path)
        result_path = generate_html_report(results_dir)
        assert result_path == results_dir / "analysis_report.html"
        assert result_path.exists()

    def test_empty_results_dir(self, tmp_path):
        # Empty directory — should still produce a report (with empty sections)
        results_dir = tmp_path / "empty"
        results_dir.mkdir()
        result_path = generate_html_report(results_dir)
        assert result_path.exists()


class TestHelperFunctions:
    def test_safe_escapes_html(self):
        assert safe("<script>") == "&lt;script&gt;"

    def test_format_value_none(self):
        assert format_value(None) == "N/A"

    def test_format_value_float(self):
        result = format_value(0.12345)
        assert result == "0.123"

    def test_format_value_small_float(self):
        result = format_value(0.0001)
        assert "e" in result  # scientific notation

    def test_format_ci_normal(self):
        result = format_ci(0.5, 1.5)
        assert "0.500" in result
        assert "1.500" in result

    def test_format_ci_none(self):
        result = format_ci(None, 1.5)
        assert result == "N/A"

    def test_fmt_p_small(self):
        assert fmt_p(0.00001) == "<0.0001"

    def test_fmt_p_normal(self):
        result = fmt_p(0.05)
        assert "0.0500" in result
