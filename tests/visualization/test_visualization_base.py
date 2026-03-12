"""Tests for visualization base module."""

import numpy as np
import plotly.graph_objects as go

from phenocluster.visualization._base import (
    BaseVisualizer,
    _p_value_stars,
    add_forest_decorations,
    add_forest_legend,
    apply_standard_layout,
    compute_forest_axis_range,
    style_significance_marker,
)


class TestApplyStandardLayout:
    def test_default_params(self):
        fig = go.Figure()
        result = apply_standard_layout(fig, title="Test")
        assert result.layout.height == 600
        assert result.layout.title.text == "Test"

    def test_custom_width(self):
        fig = go.Figure()
        result = apply_standard_layout(fig, width=800)
        assert result.layout.width == 800

    def test_no_width_by_default(self):
        fig = go.Figure()
        result = apply_standard_layout(fig)
        assert result.layout.width is None


class TestStyleSignificanceMarker:
    def test_p_lt_001_diamond(self):
        style = style_significance_marker(p_value=0.0005, effect=2.0)
        assert style["marker_symbol"] == "diamond"

    def test_p_lt_05_circle(self):
        style = style_significance_marker(p_value=0.03, effect=0.5)
        assert style["marker_symbol"] == "circle"
        assert style["marker_color"] == "#0072B2"  # lower risk color

    def test_ns_gray(self):
        style = style_significance_marker(p_value=0.1, effect=1.2)
        assert style["marker_color"] == "gray"

    def test_unreliable_x_marker(self):
        style = style_significance_marker(p_value=0.01, effect=2.0, unreliable=True)
        assert style["marker_symbol"] == "x"
        assert style["text_color"] == "orange"

    def test_none_pvalue_ns(self):
        style = style_significance_marker(p_value=None, effect=1.0)
        assert style["marker_color"] == "gray"


class TestComputeForestAxisRange:
    def test_normal_bounds(self):
        data = [
            {"CI_lower": 0.5, "CI_upper": 2.0},
            {"CI_lower": 0.3, "CI_upper": 5.0},
        ]
        x_min, x_max = compute_forest_axis_range(data)
        assert x_min < 0.5
        assert x_max > 5.0

    def test_empty_fallback(self):
        x_min, x_max = compute_forest_axis_range([])
        assert x_min == 0.1
        assert x_max == 10


class TestAddForestLegend:
    def test_adds_annotation(self):
        fig = go.Figure()
        add_forest_legend(fig)
        assert len(fig.layout.annotations) == 1


class TestAddForestDecorations:
    def test_adds_vrects_and_annotations(self):
        fig = go.Figure()
        add_forest_decorations(fig, 0.1, 10.0)
        # Should add 2 vrects (shapes) and 2 annotations
        assert len(fig.layout.shapes) == 2
        assert len(fig.layout.annotations) == 2

    def test_no_direction_labels(self):
        fig = go.Figure()
        add_forest_decorations(fig, 0.1, 10.0, show_direction_labels=False)
        assert len(fig.layout.annotations) == 0


class TestPValueStars:
    def test_none(self):
        assert _p_value_stars(None) == ""

    def test_three_stars(self):
        assert _p_value_stars(0.0005) == " ***"

    def test_two_stars(self):
        assert _p_value_stars(0.005) == " **"

    def test_one_star(self):
        assert _p_value_stars(0.03) == " *"

    def test_no_stars(self):
        assert _p_value_stars(0.1) == ""


class TestBaseVisualizer:
    def test_color_palette_small_k(self, minimal_config):
        vis = BaseVisualizer(minimal_config, n_clusters=3)
        assert len(vis.cluster_colors) == 3

    def test_color_palette_exact_8(self, minimal_config):
        vis = BaseVisualizer(minimal_config, n_clusters=8)
        assert len(vis.cluster_colors) == 8

    def test_color_palette_large_k(self, minimal_config):
        vis = BaseVisualizer(minimal_config, n_clusters=12)
        assert len(vis.cluster_colors) == 12

    def test_calculate_adaptive_height_min(self, minimal_config):
        vis = BaseVisualizer(minimal_config, n_clusters=3)
        h = vis._calculate_adaptive_height(n_features=1, min_height=400)
        assert h >= 400

    def test_calculate_adaptive_height_max(self, minimal_config):
        vis = BaseVisualizer(minimal_config, n_clusters=3)
        h = vis._calculate_adaptive_height(n_features=100, max_height=1200)
        assert h <= 1200

    def test_get_cluster_sizes(self, minimal_config):
        vis = BaseVisualizer(minimal_config, n_clusters=3)
        labels = np.array([0, 0, 1, 1, 1, 2])
        sizes = vis._get_cluster_sizes(labels)
        assert sizes == {0: 2, 1: 3, 2: 1}
