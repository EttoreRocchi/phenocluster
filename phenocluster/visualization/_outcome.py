"""
Outcome Visualizer Mixin
========================

Forest plot for odds ratio visualization of phenotype-outcome associations.
"""

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go

from ._base import (
    STYLE,
    BaseVisualizer,
    _hover_p_value,
    _label_with_stars,
    add_forest_decorations,
    add_forest_legend,
    apply_standard_layout,
    compute_forest_axis_range,
    style_significance_marker,
)


class OutcomeVisualizer(BaseVisualizer):
    """Outcome visualization (odds-ratio forest plots).

    Attributes inherited from BaseVisualizer:
        - self.config
        - self.n_clusters
        - self.logger
        - self.cluster_colors
    """

    def create_odds_ratio_forest_plot(
        self,
        outcome_results: Dict,
        title: str = "Outcome Associations by Phenotype",
        reference_label: str = "Phenotype 0 (Reference)",
        reference_phenotype: int = 0,
    ) -> Optional[go.Figure]:
        """
        Create a forest plot of odds ratios for phenotype-outcome associations.

        Parameters
        ----------
        outcome_results : Dict
            Results from ClusterEvaluator.analyze_outcomes()
        title : str
            Plot title
        reference_label : str
            Label for the reference cluster

        Returns
        -------
        go.Figure or None
            Plotly figure with forest plot
        """
        if not outcome_results:
            self.logger.info("No outcome results for forest plot.")
            return None

        self.logger.info("Creating odds ratio forest plot...")

        # Handle nested structure (full_cohort preferred)
        if isinstance(outcome_results, dict):
            if "full_cohort" in outcome_results:
                outcome_data = outcome_results["full_cohort"]
            elif "train" in outcome_results:
                outcome_data = outcome_results["train"]
            else:
                outcome_data = outcome_results
        else:
            outcome_data = outcome_results

        if not outcome_data:
            return None

        # Collect plot data
        plot_data = []
        for outcome, clusters in outcome_data.items():
            if not isinstance(clusters, dict):
                continue
            for cluster_id, result in clusters.items():
                if str(cluster_id).startswith("_"):
                    continue
                if not isinstance(result, dict):
                    continue
                cluster_int = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
                if cluster_int == reference_phenotype:
                    continue

                or_val = result.get("OR", np.nan)
                ci_lower = result.get("CI_lower", np.nan)
                ci_upper = result.get("CI_upper", np.nan)
                p_value = result.get("p_value")
                if np.isnan(or_val) or np.isnan(ci_lower) or np.isnan(ci_upper):
                    continue

                plot_data.append(
                    {
                        "outcome": outcome,
                        "cluster": cluster_int,
                        "effect": or_val,
                        "CI_lower": ci_lower,
                        "CI_upper": ci_upper,
                        "p_value": p_value,
                    }
                )

        if not plot_data:
            self.logger.warning("No valid OR data for forest plot.")
            return None

        plot_data.sort(key=lambda x: (x["outcome"], x["cluster"]))

        fig = go.Figure()

        # Reference line at OR=1
        fig.add_vline(x=1, line_dash="solid", line_color="gray", line_width=1.5)

        # Calculate y-positions with grouping
        y_positions = []
        y_labels = []
        current_y: float = 0.0
        current_outcome = None

        for d in plot_data:
            if current_outcome != d["outcome"]:
                if current_outcome is not None:
                    current_y += 1.5
                current_outcome = d["outcome"]

            y_positions.append(current_y)
            outcome_clean = d["outcome"].replace("_", " ").title()
            y_labels.append(f"{outcome_clean} (P{d['cluster']})")
            current_y += 1

        # Plot each point
        for i, d in enumerate(plot_data):
            y_pos = y_positions[i]
            sty = style_significance_marker(d["p_value"], d["effect"])

            # CI line
            fig.add_trace(
                go.Scatter(
                    x=[d["CI_lower"], d["CI_upper"]],
                    y=[y_pos, y_pos],
                    mode="lines",
                    line=dict(color=sty["marker_color"], width=sty["line_width"]),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Point estimate
            fig.add_trace(
                go.Scatter(
                    x=[d["effect"]],
                    y=[y_pos],
                    mode="markers+text",
                    marker=dict(
                        size=STYLE["marker_size"] + 2,
                        color=sty["marker_color"],
                        symbol=sty["marker_symbol"],
                        line=dict(width=1, color="black"),
                    ),
                    text=[_label_with_stars(d.get("p_value"))],
                    textposition="middle right",
                    textfont=dict(size=9, color=sty["text_color"]),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{d['outcome'].replace('_', ' ').title()}</b><br>"
                        f"Phenotype {d['cluster']} vs Reference<br>"
                        f"OR: {d['effect']:.2f}<br>"
                        f"95% CI: [{float(d['CI_lower']):.2f}, {float(d['CI_upper']):.2f}]"
                        + _hover_p_value(d.get("p_value"))
                        + "<extra></extra>"
                    ),
                )
            )

        # Axis range and layout
        x_min, x_max = compute_forest_axis_range(plot_data)

        # Dynamic left margin based on longest label
        max_label_len = max(len(lbl) for lbl in y_labels) if y_labels else 20
        left_margin = max(220, min(400, max_label_len * 7 + 40))

        fig = apply_standard_layout(
            fig,
            title=title,
            height=max(450, min(1800, 120 + len(plot_data) * 65)),
            bottom_margin=250,
            top_margin=100,
        )
        fig.update_layout(
            xaxis=dict(
                title="Odds Ratio (log scale)",
                type="log",
                range=[np.log10(x_min), np.log10(x_max)],
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=y_positions,
                ticktext=y_labels,
                showgrid=False,
                autorange="reversed",
                tickfont=dict(size=10),
            ),
            margin=dict(l=left_margin, r=60, t=100, b=250),
            showlegend=False,
        )

        # Decorations and significance legend
        add_forest_decorations(fig, x_min, x_max)
        add_forest_legend(fig)

        self.logger.info(f"Forest plot created with {len(plot_data)} comparisons")
        return fig
