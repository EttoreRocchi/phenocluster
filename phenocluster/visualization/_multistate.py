"""
Multistate Visualizer Mixin
============================

Visualization methods for multistate model results: pathway frequencies,
transition hazard forest plots, state occupation probabilities, and
state diagrams.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._base import (
    STYLE,
    BaseVisualizer,
    _hover_p_value,
    _label_with_stars,
    _p_value_stars,
    add_forest_decorations,
    add_forest_legend,
    apply_standard_layout,
    compute_forest_axis_range,
    style_significance_marker,
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color (#RRGGBB) to rgba string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _compute_state_positions(states) -> Dict[int, Tuple[float, float]]:
    """Compute (x, y) positions for each state in the diagram.

    States are laid out left-to-right: initial -> transient -> absorbing.
    Within each column states are spread vertically.

    Parameters
    ----------
    states : list
        List of state config objects with ``id`` and ``state_type`` attributes.

    Returns
    -------
    dict
        Mapping of state id to (x, y) position.
    """
    initial_states = [s for s in states if s.state_type == "initial"]
    transient_states = [s for s in states if s.state_type == "transient"]
    absorbing_states = [s for s in states if s.state_type == "absorbing"]

    positions: Dict[int, Tuple[float, float]] = {}
    x_offset = 0

    for s in initial_states:
        positions[s.id] = (x_offset, 0.5)
        x_offset += 1

    n_trans = len(transient_states)
    for i, s in enumerate(transient_states):
        y_pos = (i + 0.5) / max(n_trans, 1)
        positions[s.id] = (x_offset, y_pos)
    if transient_states:
        x_offset += 1

    n_abs = len(absorbing_states)
    for i, s in enumerate(absorbing_states):
        y_pos = (i + 0.5) / max(n_abs, 1)
        positions[s.id] = (x_offset, y_pos)

    return positions


def _bezier_curve(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    curve_offset: float,
    n_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) arrays for a quadratic Bezier curve between two points.

    Parameters
    ----------
    x0, y0 : float
        Start point.
    x1, y1 : float
        End point.
    curve_offset : float
        Perpendicular offset applied at the midpoint to create curvature.
    n_points : int
        Number of sample points along the curve.

    Returns
    -------
    tuple of np.ndarray
        (curve_x, curve_y) arrays.
    """
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2) or 1.0

    # Perpendicular unit vector
    perp_x = -dy / length
    perp_y = dx / length

    # Control point
    mid_x = (x0 + x1) / 2 + perp_x * curve_offset
    mid_y = (y0 + y1) / 2 + perp_y * curve_offset

    t_vals = np.linspace(0, 1, n_points)
    curve_x = (1 - t_vals) ** 2 * x0 + 2 * (1 - t_vals) * t_vals * mid_x + t_vals**2 * x1
    curve_y = (1 - t_vals) ** 2 * y0 + 2 * (1 - t_vals) * t_vals * mid_y + t_vals**2 * y1
    return curve_x, curve_y


def _edge_curve_offset(
    from_state: int,
    to_state: int,
    trans_idx: int,
    edge_counter: Dict,
) -> float:
    """Compute perpendicular curve offset for a transition edge.

    Handles overlapping parallel edges by assigning increasing offsets.

    Parameters
    ----------
    from_state, to_state : int
        Source and target state ids.
    trans_idx : int
        Sequential index of this transition among all transitions.
    edge_counter : dict
        Mutable counter tracking how many edges share the same state pair.
        Updated in place.

    Returns
    -------
    float
        The curve offset to pass to :func:`_bezier_curve`.
    """
    edge_key = (min(from_state, to_state), max(from_state, to_state))
    if edge_key not in edge_counter:
        edge_counter[edge_key] = 0
    offset = 0.08 * (edge_counter[edge_key] - 0.5)
    edge_counter[edge_key] += 1
    return offset


def _label_position_on_bezier(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    curve_offset: float,
    trans_idx: int,
) -> Tuple[float, float]:
    """Return (x, y) for placing an annotation along a Bezier edge.

    The parameter *t* along the curve is varied slightly based on
    *trans_idx* to reduce label collisions.
    """
    dx = x1 - x0
    dy = y1 - y0
    length = np.sqrt(dx**2 + dy**2) or 1.0
    perp_x = -dy / length
    perp_y = dx / length

    mid_x = (x0 + x1) / 2 + perp_x * curve_offset
    mid_y = (y0 + y1) / 2 + perp_y * curve_offset

    label_t = 0.4 + 0.2 * (trans_idx % 3 - 1)
    lx = (1 - label_t) ** 2 * x0 + 2 * (1 - label_t) * label_t * mid_x + label_t**2 * x1
    ly = (1 - label_t) ** 2 * y0 + 2 * (1 - label_t) * label_t * mid_y + label_t**2 * y1
    return float(lx), float(ly)


class MultistateVisualizer(BaseVisualizer):
    """Multistate visualization methods.

    Attributes inherited from BaseVisualizer:
    - self.config: PhenoClusterConfig
    - self.n_clusters: int
    - self.logger: Logger
    - self.cluster_colors: List[str]
    """

    def create_pathway_frequency_plot(
        self,
        pathway_results: List,
        top_n: int = 10,
        title: str = "Most Common State Pathways by Phenotype",
    ) -> Optional[go.Figure]:
        """
        Create a bar chart showing pathway frequencies by phenotype.

        Parameters
        ----------
        pathway_results : List[PathwayResult]
            Results from MultistateAnalyzer.analyze_pathway_frequencies()
        top_n : int
            Number of top pathways to show
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure with pathway frequency chart
        """
        if not pathway_results:
            self.logger.info("No pathway results for frequency plot.")
            return None

        self.logger.info("Creating pathway frequency plot...")

        # Take top N pathways
        top_pathways = pathway_results[:top_n]

        fig = go.Figure()

        # Build pathway labels, wrapping long names at arrows
        pathway_labels = []
        for pr in top_pathways:
            label = " -> ".join(pr["state_names"])
            if len(label) > 30:
                label = label.replace(" -> ", " -><br>")
            pathway_labels.append(label)

        # Calculate cluster sizes from ALL pathways (not just top N)
        cluster_totals = {c: 0 for c in range(self.n_clusters)}
        for pr in pathway_results:
            for c in range(self.n_clusters):
                cluster_totals[c] += pr["counts_by_phenotype"].get(str(c), 0)

        # Add bars for each phenotype (percentages within cluster)
        for cluster_id in range(self.n_clusters):
            percentages = []
            for pr in top_pathways:
                count = pr["counts_by_phenotype"].get(str(cluster_id), 0)
                total = cluster_totals[cluster_id]
                pct = (count / total * 100) if total > 0 else 0
                percentages.append(pct)

            fig.add_trace(
                go.Bar(
                    name=f"Phenotype {cluster_id}",
                    x=pathway_labels,
                    y=percentages,
                    marker_color=self.cluster_colors[cluster_id],
                    text=[f"{p:.1f}%" for p in percentages],
                    textposition="auto",
                )
            )

        # Dynamic bottom margin based on longest label
        max_label_len = (
            max(len(lbl.replace("<br>", "")) for lbl in pathway_labels) if pathway_labels else 20
        )
        bottom_margin = max(150, min(250, max_label_len * 4 + 50))

        fig = apply_standard_layout(fig, title=title, height=500)
        fig.update_layout(
            barmode="group",
            xaxis_title="Pathway",
            yaxis_title="Percentage of Patients (%)",
            xaxis=dict(tickangle=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(b=bottom_margin),
        )

        self.logger.info(f"Pathway frequency plot created ({len(top_pathways)} pathways)")
        return fig

    def create_transition_hazard_forest_plot(
        self,
        transition_results: Dict,
        title: str = "Transition Hazard Ratios by Phenotype",
        reference_phenotype: int = 0,
    ) -> Optional[go.Figure]:
        """
        Create a forest plot of hazard ratios for each transition by phenotype.

        Parameters
        ----------
        transition_results : Dict[str, TransitionResult]
            Results from MultistateAnalyzer.fit_cause_specific_hazards()
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure with forest plot
        """
        if not transition_results:
            self.logger.info("No transition results for forest plot.")
            return None

        self.logger.info("Creating transition hazard forest plot...")

        # Collect plot data (transition_results is always a dict from JSON)
        plot_data = []

        for trans_name, result in transition_results.items():
            for phenotype, effects in result["phenotype_effects"].items():
                phenotype_int = int(phenotype)  # Keys are strings from JSON
                if phenotype_int == reference_phenotype:  # Skip reference
                    continue

                hr = effects.get("HR", np.nan)
                ci_lower = effects.get("CI_lower", np.nan)
                ci_upper = effects.get("CI_upper", np.nan)
                p_value = effects.get("p_value")
                if np.isnan(hr) or np.isnan(ci_lower) or np.isnan(ci_upper):
                    continue

                plot_data.append(
                    {
                        "transition": trans_name.replace("_", " ").title(),
                        "phenotype": phenotype_int,
                        "HR": hr,
                        "CI_lower": ci_lower,
                        "CI_upper": ci_upper,
                        "unreliable": effects.get("unreliable", False),
                        "p_value": p_value,
                    }
                )

        if not plot_data:
            self.logger.warning("No valid HR data for forest plot.")
            return None

        # Sort by transition then phenotype
        plot_data.sort(key=lambda x: (x["transition"], x["phenotype"]))

        fig = go.Figure()

        # Reference line at HR=1
        fig.add_vline(x=1, line_dash="solid", line_color="gray", line_width=1.5)

        # Calculate y-positions with grouping by transition
        y_positions = []
        y_labels = []
        current_y: float = 0.0
        current_transition = None

        for d in plot_data:
            if current_transition != d["transition"]:
                if current_transition is not None:
                    current_y += 1.5  # Extra space between transition groups
                current_transition = d["transition"]

            y_positions.append(current_y)
            y_labels.append(f"{d['transition']} (P{d['phenotype']})")
            current_y += 1

        # Plot each point
        for i, d in enumerate(plot_data):
            y_pos = y_positions[i]
            color = self.cluster_colors[d["phenotype"]]
            sty = style_significance_marker(
                d["p_value"], d["HR"], d["unreliable"], cluster_color=color
            )

            # CI line (clamp to reasonable display range for unreliable estimates)
            ci_lower_display = max(0.01, d["CI_lower"])
            ci_upper_display = min(1000, d["CI_upper"])

            fig.add_trace(
                go.Scatter(
                    x=[ci_lower_display, ci_upper_display],
                    y=[y_pos, y_pos],
                    mode="lines",
                    line=dict(
                        color=sty["marker_color"],
                        width=sty["line_width"],
                        dash=sty["line_dash"],
                    ),
                    opacity=sty["opacity"],
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Point estimate (clamp HR to display range for unreliable estimates)
            hr_display = max(0.01, min(1000, d["HR"]))
            label = (
                ("unreliable" + _p_value_stars(d.get("p_value")))
                if d["unreliable"]
                else _label_with_stars(d.get("p_value"))
            )

            fig.add_trace(
                go.Scatter(
                    x=[hr_display],
                    y=[y_pos],
                    mode="markers+text",
                    marker=dict(
                        size=STYLE["marker_size"] + 2,
                        color=sty["marker_color"],
                        symbol=sty["marker_symbol"],
                        line=dict(width=1, color="black"),
                        opacity=sty["opacity"],
                    ),
                    text=[label],
                    textposition="middle right",
                    textfont=dict(size=9, color=sty["text_color"]),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{d['transition']}</b><br>"
                        f"Phenotype {d['phenotype']} vs Reference<br>"
                        f"HR: {d['HR']:.2f}" + (" (UNRELIABLE)" if d["unreliable"] else "") + "<br>"
                        f"95% CI: [{d['CI_lower']:.2f}, {d['CI_upper']:.2f}]"
                        + _hover_p_value(d.get("p_value"))
                        + "<extra></extra>"
                    ),
                )
            )

        # Axis range and layout
        x_min, x_max = compute_forest_axis_range(plot_data, exclude_unreliable=True)

        # Dynamic left margin based on longest label
        max_label_len = max(len(lbl) for lbl in y_labels) if y_labels else 20
        left_margin = max(220, min(400, max_label_len * 7 + 40))

        fig = apply_standard_layout(
            fig,
            title=title,
            height=max(500, min(2000, 120 + len(plot_data) * 85)),
            bottom_margin=250,
        )
        fig.update_layout(
            xaxis=dict(
                title="Hazard Ratio (log scale)",
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
            margin=dict(l=left_margin, r=50, t=100, b=250),
            showlegend=False,
        )

        # Decorations and significance legend
        add_forest_decorations(fig, x_min, x_max, show_direction_labels=False)
        add_forest_legend(fig)

        self.logger.info(f"Transition hazard forest plot created ({len(plot_data)} comparisons)")
        return fig

    def create_state_occupation_uncertainty_plot(
        self,
        mc_results: Dict,
        title: str = "State Occupation Probabilities",
    ) -> Optional[go.Figure]:
        """
        Create state-centric line plot of occupation probabilities.

        One subplot per active state, with one line per phenotype showing
        point estimates from CoxPH Monte Carlo simulation.

        Parameters
        ----------
        mc_results : Dict
            Monte Carlo results containing by_phenotype (and optionally
            by_phenotype_lower, by_phenotype_upper) and time_points.

        Returns
        -------
        go.Figure or None
        """
        if not mc_results:
            return None

        time_points = mc_results.get("time_points", [])
        by_phenotype = mc_results.get("by_phenotype", {})

        if not time_points or not by_phenotype:
            return None

        self.logger.info("Creating state occupation plot (state-centric)...")

        # State labels from config
        state_labels: Dict[str, str] = {}
        if hasattr(self.config, "multistate") and self.config.multistate.states:
            state_labels = {str(s.id): s.name for s in self.config.multistate.states}

        # Collect all active states (non-zero in at least one phenotype)
        sorted_phenotypes = sorted(by_phenotype.keys(), key=lambda x: int(x))
        all_state_ids: set = set()
        for phenotype in sorted_phenotypes:
            all_state_ids.update(by_phenotype[phenotype].keys())

        # Filter out always-zero states across all phenotypes
        active_states = []
        for state_id in sorted(all_state_ids, key=lambda x: int(x)):
            has_nonzero = False
            for phenotype in sorted_phenotypes:
                probs = by_phenotype[phenotype].get(state_id, [])
                if any(p > 0 for p in probs):
                    has_nonzero = True
                    break
            if has_nonzero:
                active_states.append(state_id)

        if not active_states:
            self.logger.warning("No active states for occupation plot.")
            return None

        # Layout: one subplot per active state
        n_states = len(active_states)
        n_cols = min(n_states, 2) if n_states > 2 else n_states
        n_rows = math.ceil(n_states / n_cols)

        subplot_titles = [state_labels.get(str(s), f"State {s}") for s in active_states]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_yaxes=True,
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )

        # Colors map to phenotypes (consistent with rest of report)
        phenotype_colors = STYLE["cluster_colors"]

        for s_idx, state_id in enumerate(active_states):
            row = s_idx // n_cols + 1
            col = s_idx % n_cols + 1

            for p_idx, phenotype in enumerate(sorted_phenotypes):
                color = phenotype_colors[p_idx % len(phenotype_colors)]
                probs = by_phenotype[phenotype].get(state_id, [])
                if not probs:
                    continue

                y_med = [p * 100 for p in probs]

                # Point estimate line
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=y_med,
                        name=f"Phenotype {phenotype}",
                        mode="lines",
                        line=dict(width=2, color=color),
                        showlegend=(s_idx == 0),
                        hovertemplate=(
                            f"Phenotype {phenotype}<br>"
                            "Time: %{x} days<br>"
                            "Probability: %{y:.1f}%<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )

        n_sims = mc_results.get("n_simulations", 0)
        subtitle = f"Point estimates from Cox PH simulation ({n_sims:,} trajectories)"

        fig = apply_standard_layout(
            fig, title=title, height=350 * n_rows, bottom_margin=70, top_margin=150
        )
        fig.update_layout(
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        )

        # Subtitle below title, above subplots
        fig.add_annotation(
            text=subtitle,
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.03,
            showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor="center",
            yanchor="bottom",
        )

        for r in range(n_rows):
            for c in range(n_cols):
                fig.update_xaxes(title_text="Time (days)", row=r + 1, col=c + 1)

        self.logger.info(
            f"State occupation plot created ({n_states} states, "
            f"{len(sorted_phenotypes)} phenotypes)"
        )
        return fig

    def create_state_diagram(
        self, transition_results: Dict, title: str = "State Transition Diagram"
    ) -> Optional[go.Figure]:
        """
        Create a state diagram showing transitions with hazard ratio annotations.

        Parameters
        ----------
        transition_results : Dict[str, TransitionResult]
            Results from MultistateAnalyzer
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure with state diagram
        """
        if not transition_results:
            self.logger.info("No transition results for state diagram.")
            return None

        self.logger.info("Creating state diagram...")

        # Get states from config
        if not hasattr(self.config, "multistate") or not self.config.multistate.states:
            self.logger.warning("No multistate config for state diagram.")
            return None

        states = self.config.multistate.states

        # Calculate node positions using extracted helper
        state_positions = _compute_state_positions(states)

        fig = go.Figure()

        # Track edge offsets to prevent overlap
        edge_counter: Dict = {}

        # Draw edges (transitions)
        for trans_idx, (trans_name, result) in enumerate(transition_results.items()):
            from_state = result["from_state"]
            to_state = result["to_state"]

            if from_state not in state_positions or to_state not in state_positions:
                continue

            x0, y0 = state_positions[from_state]
            x1, y1 = state_positions[to_state]

            # Compute curve offset using extracted helper
            dx = x1 - x0
            dy = y1 - y0
            curve_offset = _edge_curve_offset(from_state, to_state, trans_idx, edge_counter)

            # Override for parallel transitions at same x or y
            if abs(dx) < 0.1:  # Vertical transition
                curve_offset = 0.1 * (trans_idx % 3 - 1)
            elif abs(dy) < 0.1:  # Horizontal transition
                curve_offset = 0.08 * (trans_idx % 3 - 1)

            # Create curved path using extracted helper
            curve_x, curve_y = _bezier_curve(x0, y0, x1, y1, curve_offset)

            # Get HR for annotation (skip reference phenotype)
            hr_text = ""
            for p, effects in result["phenotype_effects"].items():
                if int(p) == 0:  # Keys are strings from JSON
                    continue
                hr = effects.get("HR", 1.0)
                unreliable = effects.get("unreliable", False)
                if unreliable:
                    continue  # Skip unreliable HRs in the diagram
                hr_text += f"P{p}:{hr:.1f} "

            # Draw curved arrow
            fig.add_trace(
                go.Scatter(
                    x=curve_x.tolist(),
                    y=curve_y.tolist(),
                    mode="lines",
                    line=dict(color="gray", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add HR annotation along the curve
            if hr_text.strip():
                label_x, label_y = _label_position_on_bezier(
                    x0, y0, x1, y1, curve_offset, trans_idx
                )
                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=hr_text.strip(),
                    showarrow=False,
                    font=dict(
                        size=STYLE["edge_label_font_size"],
                        family=STYLE["font_family"],
                    ),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1,
                )

        # Draw nodes (states)
        node_colors = {
            "initial": "#2ca02c",
            "transient": "#1f77b4",
            "absorbing": "#d62728",
        }

        for state in states:
            if state.id not in state_positions:
                continue

            x, y = state_positions[state.id]
            color = node_colors.get(state.state_type, "#7f7f7f")

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(size=55, color=color, line=dict(width=2, color="black")),
                    text=[state.name],
                    textposition="middle center",
                    textfont=dict(size=12, color="white"),
                    showlegend=False,
                    hovertemplate=(f"{state.name}<br>Type: {state.state_type}<extra></extra>"),
                )
            )

        fig = apply_standard_layout(fig, title=title, height=500, bottom_margin=100)
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        # Add state-type legend using dummy traces
        for stype, scolor, slabel in [
            ("initial", "#2ca02c", "Initial"),
            ("transient", "#1f77b4", "Transient"),
            ("absorbing", "#d62728", "Absorbing"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=12, color=scolor),
                    name=slabel,
                    showlegend=True,
                )
            )
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5,
                font=dict(size=STYLE["font_size"]),
            ),
        )

        self.logger.info(f"State diagram created ({len(states)} states)")
        return fig
