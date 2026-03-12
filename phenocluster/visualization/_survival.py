"""
Survival Visualizer Mixin
=========================

Kaplan-Meier and Nelson-Aalen survival visualization methods.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from ._base import STYLE, BaseVisualizer, apply_standard_layout


class SurvivalVisualizer(BaseVisualizer):
    """Survival curve visualizations (Kaplan-Meier, Nelson-Aalen).

    Assumes the consuming class provides:
        self.config : PhenoClusterConfig
        self.n_clusters : int
        self.logger : logging.Logger
        self.cluster_colors : List[str]
    """

    # SHARED SURVIVAL CURVE HELPER

    def _plot_survival_curve(
        self,
        fig: go.Figure,
        cluster_data: Dict,
        cluster_id: int,
        y_key: str,
        extra_hover_lines: Optional[List[str]] = None,
    ) -> None:
        """Add a single cluster's survival/hazard curve with CI band to *fig*.

        Parameters
        ----------
        fig : go.Figure
            Target figure (modified in place).
        cluster_data : Dict
            Per-cluster dict with keys ``timeline``, *y_key*,
            ``confidence_interval_lower``, ``confidence_interval_upper``,
            ``n_events``, ``n_patients``.
        cluster_id : int
            Cluster index (used for colour look-up and legend label).
        y_key : str
            Key into *cluster_data* for the y-axis values, e.g.
            ``"survival_function"`` or ``"cumulative_hazard"``.
        extra_hover_lines : list of str, optional
            Additional hover-template lines inserted before ``<extra>``.
        """
        times = cluster_data.get("timeline", [])
        y_values = cluster_data.get(y_key, [])
        ci_lower = cluster_data.get("confidence_interval_lower", [])
        ci_upper = cluster_data.get("confidence_interval_upper", [])

        if len(times) == 0 or len(y_values) == 0:
            return

        color = self.cluster_colors[cluster_id]
        n_events = cluster_data.get("n_events", "N/A")
        n_patients = cluster_data.get("n_patients", "N/A")

        # Confidence interval (filled area)
        if len(ci_lower) > 0 and len(ci_upper) > 0:
            fig.add_trace(
                go.Scatter(
                    x=list(times) + list(times)[::-1],
                    y=list(ci_upper) + list(ci_lower)[::-1],
                    fill="toself",
                    fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Build hover template
        hover_parts = [
            f"<b>Phenotype {cluster_id}</b><br>",
            "Time: %{x:.1f}<br>",
        ]
        if extra_hover_lines:
            hover_parts.extend(extra_hover_lines)
        hover_parts.append(f"Events: {n_events}<extra></extra>")
        hover_template = "".join(hover_parts)

        # Main curve (step function)
        fig.add_trace(
            go.Scatter(
                x=times,
                y=y_values,
                mode="lines",
                line=dict(color=color, width=STYLE["line_width"], shape="hv"),
                name=f"Phenotype {cluster_id} (n={n_patients})",
                hovertemplate=hover_template,
            )
        )

    def _apply_survival_layout(
        self,
        fig: go.Figure,
        title: str,
        display_name: str,
        yaxis_title: str,
        yaxis_range: Optional[list] = None,
        time_unit: str = "days",
    ) -> go.Figure:
        """Apply standard layout tweaks shared by KM and NA plots."""
        fig = apply_standard_layout(
            fig,
            title=f"{title}<br><sup>{display_name}</sup>",
            height=500,
            width=750,
            top_margin=100,
            right_margin=160,
        )
        yaxis_kwargs: Dict = {}
        if yaxis_range is not None:
            yaxis_kwargs["range"] = yaxis_range
        fig.update_layout(
            xaxis_title=f"Time ({time_unit})",
            yaxis_title=yaxis_title,
            yaxis=yaxis_kwargs,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10),
            ),
        )
        return fig

    # NUMBER-AT-RISK TABLE FOR KAPLAN-MEIER PLOTS

    @staticmethod
    def _n_at_risk_at_time(
        timeline: np.ndarray,
        survival_fn: np.ndarray,
        n_patients: int,
        t: float,
    ) -> int:
        """Estimate the number at risk at time *t* from KM curve data.

        Uses the approximation ``n_at_risk(t) ~ ceil(n_patients * S(t-))``.
        This is the standard approach when only the survival function (not the
        full event table) is available in the visualisation layer.
        """
        if t <= 0:
            return n_patients
        mask = timeline <= t
        if not np.any(mask):
            return n_patients
        s_t = survival_fn[np.where(mask)[0][-1]]
        return max(0, math.ceil(n_patients * s_t))

    def _add_at_risk_table(
        self,
        fig: go.Figure,
        km_data: Dict,
        n_time_points: int = 6,
    ) -> go.Figure:
        """Append a number-at-risk table below an existing KM plot.

        Parameters
        ----------
        fig : go.Figure
            The Kaplan-Meier figure (modified in place and returned).
        km_data : dict
            Mapping ``cluster_id -> cluster_dict`` as stored in
            ``survival_result["survival_data"]``.
        n_time_points : int
            Number of evenly-spaced time points to display (default 6).

        Returns
        -------
        go.Figure
        """
        # Collect all cluster ids that are actually present.
        cluster_ids = sorted(km_data.keys())
        if not cluster_ids:
            return fig

        # Determine the global time range across all clusters.
        t_max = max(
            np.max(km_data[cid]["timeline"])
            for cid in cluster_ids
            if len(km_data[cid].get("timeline", [])) > 0
        )
        time_points = np.linspace(0, t_max, n_time_points)

        # Layout constants - increase both height and bottom margin so
        # the plot area stays the same size and the at-risk table gets
        # dedicated space below the x-axis labels.
        n_rows = len(cluster_ids) + 1  # +1 for header row
        extra_height = n_rows * 35 + 50  # pixels for the at-risk table
        current_height = fig.layout.height or 500
        current_margin_b = fig.layout.margin.b or 80
        fig.update_layout(
            height=current_height + extra_height,
            margin=dict(b=max(current_margin_b, extra_height + 30)),
        )

        row_height = 0.035  # fraction of the now-taller figure
        label_offset = 0.10  # gap to clear x-axis title + tick labels
        base_y = -label_offset

        # Header row - "No. at risk" label
        fig.add_annotation(
            x=-0.03,
            y=base_y,
            xref="paper",
            yref="paper",
            text="<b>No. at risk</b>",
            showarrow=False,
            font=dict(size=STYLE["at_risk_font_size"], family=STYLE["font_family"]),
            xanchor="right",
            yanchor="top",
        )

        for row_idx, cid in enumerate(cluster_ids):
            cluster_km = km_data[cid]
            timeline = np.asarray(cluster_km.get("timeline", []))
            surv_fn = np.asarray(cluster_km.get("survival_function", []))
            n_patients = cluster_km.get("n_patients", 0)
            color = self.cluster_colors[cid]

            y_pos = base_y - (row_idx + 1) * row_height

            # Cluster label on the left
            fig.add_annotation(
                x=-0.03,
                y=y_pos,
                xref="paper",
                yref="paper",
                text=f"<b>Phenotype {cid}</b>",
                showarrow=False,
                font=dict(
                    size=STYLE["at_risk_font_size"],
                    color=color,
                    family=STYLE["font_family"],
                ),
                xanchor="right",
                yanchor="top",
            )

            # At-risk counts at each time point
            for tp in time_points:
                n_risk = self._n_at_risk_at_time(timeline, surv_fn, n_patients, tp)
                fig.add_annotation(
                    x=tp,
                    y=y_pos,
                    xref="x",
                    yref="paper",
                    text=str(n_risk),
                    showarrow=False,
                    font=dict(
                        size=STYLE["at_risk_font_size"],
                        color=color,
                        family=STYLE["font_family"],
                    ),
                    xanchor="center",
                    yanchor="top",
                )

        return fig

    # KAPLAN-MEIER SURVIVAL CURVES

    def create_kaplan_meier_plot(
        self,
        survival_result: Dict,
        target_name: str = "survival",
        title: str = "Kaplan-Meier Survival Curves by Phenotype",
    ) -> Optional[go.Figure]:
        """
        Create Kaplan-Meier survival curves with confidence intervals.

        Parameters
        ----------
        survival_result : Dict
            Results from SurvivalAnalyzer.analyze_survival() for a single target.
            Expected keys: 'survival_data', 'comparison', 'median_survival'
        target_name : str
            Name of the survival target for the plot title
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure with KM curves
        """
        if not survival_result:
            self.logger.info("No survival results for KM plot.")
            return None

        self.logger.info(f"Creating Kaplan-Meier plot for {target_name}...")

        km_data = survival_result.get("survival_data", {})
        if not km_data:
            self.logger.info("No Kaplan-Meier data available.")
            return None

        fig = go.Figure()

        median_survivals = survival_result.get("median_survival", {})

        for cluster_id in range(self.n_clusters):
            if cluster_id not in km_data:
                continue

            cluster_km = km_data[cluster_id]

            # Extra hover info specific to KM
            median_surv = median_survivals.get(cluster_id, np.nan)
            median_str = f"{median_surv:.1f}" if not np.isnan(median_surv) else "N/A"

            self._plot_survival_curve(
                fig,
                cluster_km,
                cluster_id,
                y_key="survival_function",
                extra_hover_lines=[
                    "Survival: %{y:.3f}<br>",
                    f"Median: {median_str}<br>",
                ],
            )

        # Horizontal median reference line
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray", line_width=1)

        display_name = target_name.replace("_", " ").title()
        fig = self._apply_survival_layout(
            fig, title, display_name, yaxis_title="Survival Probability", yaxis_range=[0, 1.05]
        )

        # Append number-at-risk table below the KM curves.
        fig = self._add_at_risk_table(fig, km_data)

        self.logger.info(f"Kaplan-Meier plot created for {target_name}")
        return fig

    # NELSON-AALEN CUMULATIVE HAZARD PLOT

    def create_nelson_aalen_plot(
        self,
        survival_result: Dict,
        target_name: str = "survival",
        title: str = "Nelson-Aalen Cumulative Hazard by Phenotype",
    ) -> Optional[go.Figure]:
        """
        Create Nelson-Aalen cumulative hazard curves with confidence intervals.

        Parameters
        ----------
        survival_result : Dict
            Results from SurvivalAnalyzer.analyze_survival() for a single target.
            Expected keys: 'nelson_aalen_data', 'comparison'
        target_name : str
            Name of the survival target for the plot title
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure with Nelson-Aalen cumulative hazard curves
        """
        if not survival_result:
            self.logger.info("No survival results for Nelson-Aalen plot.")
            return None

        self.logger.info(f"Creating Nelson-Aalen plot for {target_name}...")

        na_data = survival_result.get("nelson_aalen_data", {})
        if not na_data:
            self.logger.info("No Nelson-Aalen data available.")
            return None

        fig = go.Figure()

        for cluster_id in range(self.n_clusters):
            if cluster_id not in na_data:
                continue

            self._plot_survival_curve(
                fig,
                na_data[cluster_id],
                cluster_id,
                y_key="cumulative_hazard",
                extra_hover_lines=["Cumulative Hazard: %{y:.3f}<br>"],
            )

        # Compute sensible y-axis upper bound to prevent one extreme cluster
        # from compressing all others
        all_hazards = []
        for cluster_id in range(self.n_clusters):
            if cluster_id in na_data:
                all_hazards.extend(na_data[cluster_id].get("cumulative_hazard", []))
        na_yrange = None
        if all_hazards:
            p95 = float(np.percentile(all_hazards, 95))
            y_max = min(p95 * 1.5, max(all_hazards) * 1.1)
            na_yrange = [0, y_max]

        display_name = target_name.replace("_", " ").title()
        fig = self._apply_survival_layout(
            fig,
            title,
            display_name,
            yaxis_title="Cumulative Hazard",
            yaxis_range=na_yrange,
        )

        self.logger.info(f"Nelson-Aalen plot created for {target_name}")
        return fig
