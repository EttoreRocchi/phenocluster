"""
Base Visualizer
===============

Shared configuration, styling constants, utility functions, and base class
for all domain-specific visualizer mixins.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger

# STYLE CONFIGURATION

STYLE = {
    "font_family": "Arial, sans-serif",
    "font_size": 12,
    "title_font_size": 16,
    "axis_title_font_size": 13,
    "annotation_font_size": 11,
    "at_risk_font_size": 9,
    "edge_label_font_size": 10,
    "plot_bgcolor": "white",
    "paper_bgcolor": "white",
    "gridcolor": "rgba(128, 128, 128, 0.2)",
    "line_width": 2,
    "marker_size": 8,
    # Colorblind-safe palette (Wong, 2011 - Nature Methods 8:441)
    "cluster_colors": [
        "#0072B2",  # Blue
        "#E69F00",  # Orange
        "#009E73",  # Bluish green
        "#CC79A7",  # Reddish purple
        "#56B4E9",  # Sky blue
        "#D55E00",  # Vermilion
        "#F0E442",  # Yellow
        "#999999",  # Gray
    ],
    "diverging_colorscale": "RdBu_r",
}

# Extended palette for K > 8 (Tol bright + muted)
_EXTENDED_COLORS = [
    "#332288",  # indigo
    "#88CCEE",  # cyan
    "#44AA99",  # teal
    "#117733",  # green
    "#999933",  # olive
    "#AA4499",  # purple
]


def apply_standard_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 600,
    width: Optional[int] = None,
    bottom_margin: int = 80,
    top_margin: int = 100,
    left_margin: int = 80,
    right_margin: int = 80,
) -> go.Figure:
    """Apply consistent styling to a Plotly figure."""
    layout_updates = {
        "title": dict(
            text=title,
            font=dict(size=STYLE["title_font_size"], family=STYLE["font_family"]),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        "font": dict(family=STYLE["font_family"], size=STYLE["font_size"]),
        "plot_bgcolor": STYLE["plot_bgcolor"],
        "paper_bgcolor": STYLE["paper_bgcolor"],
        "height": height,
        "margin": dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin, pad=10),
    }
    if width:
        layout_updates["width"] = width

    fig.update_layout(**layout_updates)

    fig.update_xaxes(
        showgrid=True,
        gridcolor=STYLE["gridcolor"],
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickfont=dict(family=STYLE["font_family"], size=STYLE["font_size"]),
        title_font=dict(family=STYLE["font_family"], size=STYLE["axis_title_font_size"]),
        automargin=True,
        title_standoff=15,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=STYLE["gridcolor"],
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickfont=dict(family=STYLE["font_family"], size=STYLE["font_size"]),
        title_font=dict(family=STYLE["font_family"], size=STYLE["axis_title_font_size"]),
        automargin=True,
        title_standoff=15,
    )

    return fig


# FOREST PLOT HELPERS


def style_significance_marker(
    p_value: Optional[float],
    effect: float,
    unreliable: bool = False,
    cluster_color: Optional[str] = None,
) -> Dict:
    """Return marker/line styling based on p-value significance.

    Returns dict with keys: marker_color, marker_symbol, line_width,
    line_dash, opacity, text_color.
    """
    if unreliable:
        return {
            "marker_color": cluster_color or "gray",
            "marker_symbol": "x",
            "line_width": 1.5,
            "line_dash": "dash",
            "opacity": 0.5,
            "text_color": "orange",
        }
    if p_value is not None and p_value < 0.001:
        color = "#D55E00" if effect > 1 else "#0072B2"
        return {
            "marker_color": color,
            "marker_symbol": "diamond",
            "line_width": 2.5,
            "line_dash": "solid",
            "opacity": 1.0,
            "text_color": color,
        }
    if p_value is not None and p_value < 0.05:
        color = "#D55E00" if effect > 1 else "#0072B2"
        return {
            "marker_color": color,
            "marker_symbol": "circle",
            "line_width": 2,
            "line_dash": "solid",
            "opacity": 1.0,
            "text_color": color,
        }
    # Non-significant
    return {
        "marker_color": "gray",
        "marker_symbol": "circle",
        "line_width": 1.5,
        "line_dash": "solid",
        "opacity": 1.0,
        "text_color": "gray",
    }


def compute_forest_axis_range(
    plot_data: List[Dict],
    ci_lower_key: str = "CI_lower",
    ci_upper_key: str = "CI_upper",
    exclude_unreliable: bool = False,
) -> Tuple[float, float]:
    """Compute log-scale x-axis range from CI bounds."""
    data = plot_data
    if exclude_unreliable:
        reliable = [d for d in plot_data if not d.get("unreliable", False)]
        if reliable:
            data = reliable

    lowers = [d[ci_lower_key] for d in data if d[ci_lower_key] > 0]
    uppers = [d[ci_upper_key] for d in data]

    if lowers and uppers:
        x_min = max(0.1, min(lowers) * 0.5)
        x_max = min(100, max(uppers) * 2)
    else:
        x_min, x_max = 0.1, 10
    return x_min, x_max


def add_forest_legend(fig: go.Figure, y_position: float = -0.22) -> None:
    """Add marker-shape significance legend below a forest plot."""
    legend_text = (
        "&#x25C6; p&lt;0.001 &nbsp;&nbsp; "
        "&#x25CF; p&lt;0.05 &nbsp;&nbsp; "
        '<span style="color:gray">&#x25CF;</span> NS &nbsp;&nbsp; '
        '<span style="color:orange">&#x2716;</span> Unreliable'
        "&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; "
        '<span style="color:#D55E00">&#x25A0;</span> Higher risk &nbsp;&nbsp; '
        '<span style="color:#0072B2">&#x25A0;</span> Lower risk'
    )
    fig.add_annotation(
        text=legend_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=y_position,
        showarrow=False,
        font=dict(size=STYLE["annotation_font_size"], family=STYLE["font_family"]),
        xanchor="center",
        yanchor="top",
    )


def add_forest_decorations(
    fig: go.Figure,
    x_min: float,
    x_max: float,
    show_direction_labels: bool = True,
) -> None:
    """Add risk-direction shading and annotation to a forest plot."""
    # Risk direction shading
    fig.add_vrect(x0=x_min, x1=1, fillcolor="green", opacity=0.05, layer="below", line_width=0)
    fig.add_vrect(x0=1, x1=x_max, fillcolor="red", opacity=0.05, layer="below", line_width=0)

    if show_direction_labels:
        fig.add_annotation(
            x=np.log10(x_min * 1.5),
            y=0.98,
            xref="x",
            yref="paper",
            text="<- Lower Risk",
            showarrow=False,
            font=dict(size=9, color="green"),
            yanchor="top",
        )
        fig.add_annotation(
            x=np.log10(x_max / 1.5),
            y=0.98,
            xref="x",
            yref="paper",
            text="Higher Risk ->",
            showarrow=False,
            font=dict(size=9, color="red"),
            yanchor="top",
        )


def _p_value_stars(p_value):
    """Return significance stars for a p-value."""
    if p_value is None:
        return ""
    if p_value < 0.001:
        return " ***"
    if p_value < 0.01:
        return " **"
    if p_value < 0.05:
        return " *"
    return ""


def _label_with_stars(p_value=None) -> str:
    """Return p-value label with significance stars."""
    if p_value is None:
        return ""
    return f"p={p_value:.3f}" + _p_value_stars(p_value)


def _hover_p_value(p_value) -> str:
    """Return a hover template snippet for p-value (empty if None)."""
    if p_value is None:
        return ""
    return f"<br>p-value: {p_value:.4f}"


class BaseVisualizer:
    """Base class providing shared state and utilities for all visualizer mixins."""

    def __init__(self, config: PhenoClusterConfig, n_clusters: int):
        self.config = config
        self.n_clusters = n_clusters
        self.logger = get_logger("visualization", config)

        base_palette = STYLE["cluster_colors"]
        if n_clusters <= len(base_palette):
            self.cluster_colors = base_palette[:n_clusters]
        else:
            extended = base_palette + _EXTENDED_COLORS
            if n_clusters > len(extended):
                self.logger.warning(
                    f"K={n_clusters} exceeds the extended palette "
                    f"({len(extended)} colours). Colours will cycle."
                )
                # Cycle colours to cover all clusters
                self.cluster_colors = [extended[i % len(extended)] for i in range(n_clusters)]
            else:
                self.cluster_colors = extended[:n_clusters]

    def _calculate_adaptive_height(
        self,
        n_features: int,
        base_height: int = 400,
        height_per_feature: int = 30,
        min_height: int = 400,
        max_height: int = 1200,
    ) -> int:
        """Calculate adaptive height based on number of features."""
        calculated_height = base_height + (n_features * height_per_feature)
        return max(min_height, min(calculated_height, max_height))

    def _get_cluster_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """Get size of each cluster."""
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
