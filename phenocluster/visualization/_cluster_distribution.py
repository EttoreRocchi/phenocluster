"""
Cluster Distribution Mixin
===========================

Cluster distribution bar chart and model selection criteria plots.
"""

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._base import STYLE, BaseVisualizer, apply_standard_layout


class ClusterDistributionVisualizer(BaseVisualizer):
    """Cluster distribution and model selection plots.

    Attributes inherited from BaseVisualizer:
        self.config
        self.n_clusters
        self.logger
        self.cluster_colors
    """

    # CLUSTER DISTRIBUTION PLOT

    def create_cluster_distribution(
        self, labels: np.ndarray, title: str = "Phenotype Distribution"
    ) -> go.Figure:
        """
        Create a bar chart showing the distribution of samples across clusters.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments
        title : str
            Plot title

        Returns
        -------
        go.Figure
            Plotly figure with cluster distribution
        """
        self.logger.info("Creating cluster distribution plot...")

        cluster_sizes = self._get_cluster_sizes(labels)
        total = len(labels)

        clusters = list(range(self.n_clusters))
        counts = [cluster_sizes.get(c, 0) for c in clusters]
        percentages = [c / total * 100 for c in counts]

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=[f"Phenotype {c}" for c in clusters],
                y=counts,
                marker_color=self.cluster_colors,
                text=[f"{c}<br>({p:.1f}%)" for c, p in zip(counts, percentages)],
                textposition="outside",
                textfont=dict(size=STYLE["font_size"]),
                hovertemplate=(
                    "<b>Phenotype %{x}</b><br>"
                    "Count: %{y}<br>"
                    "Percentage: %{customdata:.1f}%<extra></extra>"
                ),
                customdata=percentages,
            )
        )

        fig = apply_standard_layout(fig, title=title, height=450, width=500)
        max_count = max(counts) if counts else 1
        fig.update_layout(
            xaxis_title="Phenotype",
            yaxis_title="Number of Patients",
            yaxis=dict(range=[0, max_count * 1.2]),
            showlegend=False,
            bargap=0.3,
        )

        self.logger.info("Cluster distribution plot created")
        return fig

    # ENHANCED MODEL SELECTION PLOT

    def create_model_selection_plot(
        self, selection_results: Dict, title: str = "Model Selection"
    ) -> Optional[go.Figure]:
        """
        Create plot showing model selection criteria with elbow detection.

        Parameters
        ----------
        selection_results : Dict
            Results from model selection including 'all_results' and 'best_n_clusters'
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure object
        """
        if not selection_results or "all_results" not in selection_results:
            self.logger.info("No model selection results to plot.")
            return None

        self.logger.info("Creating model selection plot...")

        results = selection_results["all_results"]
        if not results:
            self.logger.warning("Empty results list")
            return None

        try:
            n_clusters_list = [r["n_clusters"] for r in results]
        except (KeyError, TypeError) as e:
            self.logger.error(f"Malformed results: {e}")
            return None

        best_k = selection_results.get("best_n_clusters")
        criterion = selection_results.get(
            "criterion_used", selection_results.get("criterion", "BIC")
        )

        # Detect available criteria in results
        known_criteria = ["BIC", "AIC", "ICL", "log_likelihood"]
        available = [c for c in known_criteria if c in results[0]]

        if not available:
            # Single-criterion data (from comparison_table): use the criterion key
            available = [
                k for k in results[0] if k not in ("n_clusters",) and not k.startswith("std_")
            ]

        if len(available) <= 1:
            # Single-panel plot
            criterion_key = available[0] if available else criterion
            values = [r.get(criterion_key, np.nan) for r in results]
            std_key = f"std_{criterion_key}"
            std_values = [r.get(std_key, 0) for r in results] if std_key in results[0] else None

            fig = go.Figure()
            trace_kwargs = {
                "x": n_clusters_list,
                "y": values,
                "mode": "markers",
                "name": criterion_key,
                "marker": dict(size=STYLE["marker_size"], color="#1f77b4"),
            }
            if std_values:
                trace_kwargs["error_y"] = dict(
                    type="data", array=std_values, visible=True, color="#1f77b4"
                )
            fig.add_trace(go.Scatter(**trace_kwargs))

            if best_k is not None:
                fig.add_vline(x=best_k, line_dash="dash", line_color="red", line_width=2)
                fig.add_annotation(
                    x=best_k,
                    y=max(v for v in values if not np.isnan(v)),
                    text=f"Best: k={best_k}",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=11, color="red"),
                )

            _ic_criteria = {"BIC", "AIC", "CAIC", "SABIC", "ICL"}
            direction = (
                "lower is better" if criterion_key.upper() in _ic_criteria else "higher is better"
            )
            fig.update_xaxes(title_text="Number of Clusters (k)", dtick=1)
            fig.update_yaxes(title_text=f"{criterion_key} ({direction})")

            fig = apply_standard_layout(
                fig,
                title=f"{title}<br><sup>Selected: {best_k} clusters by {criterion}</sup>",
                height=450,
            )
            fig.update_layout(showlegend=False)
        else:
            # Multi-criteria 2x2 plot
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    "BIC (lower is better)",
                    "AIC (lower is better)",
                    "ICL (lower is better)",
                    "Log-Likelihood (higher is better)",
                ],
                vertical_spacing=0.20,
                horizontal_spacing=0.15,
            )
            metrics = [
                ("BIC", 1, 1, "#1f77b4"),
                ("AIC", 1, 2, "#ff7f0e"),
                ("ICL", 2, 1, "#2ca02c"),
                ("log_likelihood", 2, 2, "#d62728"),
            ]
            for name, row, col, color in metrics:
                values = [r.get(name, r.get(f"mean_{name}", np.nan)) for r in results]
                std_key = f"std_{name}"
                trace_kwargs = {
                    "x": n_clusters_list,
                    "y": values,
                    "mode": "markers",
                    "name": name,
                    "marker": dict(size=STYLE["marker_size"], color=color),
                    "showlegend": False,
                }
                if std_key in results[0]:
                    std_values = [r.get(std_key, 0) for r in results]
                    trace_kwargs["error_y"] = dict(
                        type="data", array=std_values, visible=True, color=color
                    )
                fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)

            if best_k is not None:
                for row in range(1, 3):
                    for col in range(1, 3):
                        fig.add_vline(
                            x=best_k,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            row=row,
                            col=col,
                        )

            for i in range(1, 3):
                for j in range(1, 3):
                    fig.update_xaxes(title_text="Number of Clusters (k)", dtick=1, row=i, col=j)

            fig = apply_standard_layout(
                fig,
                title=f"{title}<br><sup>Selected: {best_k} clusters by {criterion}</sup>",
                height=800,
            )
            fig.update_layout(showlegend=False)

        self.logger.info("Model selection plot created")
        return fig
