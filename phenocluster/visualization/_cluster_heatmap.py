"""
Cluster Heatmap Mixin
======================

Continuous variable heatmap and categorical variable heatmap/Sankey
visualizations by phenotype.
"""

import colorsys
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._base import STYLE, apply_standard_layout


class ClusterHeatmapMixin:
    """Mixin providing heatmap and categorical flow visualizations.

    Assumes the following attributes are provided by BaseVisualizer:
        self.config
        self.n_clusters
        self.logger
        self.cluster_colors
    """

    # ENHANCED HEATMAP WITH SIGNIFICANCE

    def create_heatmap(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        consensus_matrix: Optional[np.ndarray] = None,
        show_significance: bool = True,
        title: str = "Continuous Variables by Phenotype",
    ) -> Optional[go.Figure]:
        """
        Create heatmap of standardized continuous variables by cluster.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with continuous variables
        labels : np.ndarray
            Cluster assignments
        consensus_matrix : np.ndarray, optional
            Consensus matrix for cluster reordering
        show_significance : bool
            Whether to show significance markers (requires scipy)
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure object
        """
        if not self.config.continuous_columns:
            self.logger.info("No continuous columns for heatmap.")
            return None

        self.logger.info("Creating enhanced heatmap...")

        # Compute statistics per cluster
        cluster_stats = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            cluster_data = df.loc[cluster_mask, self.config.continuous_columns]
            stats = {
                "mean": cluster_data.mean().values,
                "std": cluster_data.std().values,
                "n": int(cluster_mask.sum()),
            }
            cluster_stats.append(stats)

        cluster_means = np.array([s["mean"] for s in cluster_stats])

        # Standardize means using individual-level statistics (not between-cluster SD,
        # which is unstable with only K=3-6 values and produces misleadingly extreme z-scores)
        overall_mean = df[self.config.continuous_columns].mean().values
        overall_std = df[self.config.continuous_columns].std().values
        overall_std[overall_std == 0] = 1  # Avoid division by zero
        z_scores = (cluster_means - overall_mean) / overall_std

        # Reorder clusters using hierarchical clustering if consensus matrix available
        cluster_order = list(range(self.n_clusters))
        if consensus_matrix is not None and self.n_clusters > 1:
            try:
                from scipy.cluster.hierarchy import leaves_list, linkage
                from scipy.spatial.distance import squareform

                cluster_consensus = np.zeros((self.n_clusters, self.n_clusters))
                for i in range(self.n_clusters):
                    mask_i = labels == i
                    for j in range(self.n_clusters):
                        mask_j = labels == j
                        if i == j:
                            cluster_consensus[i, j] = 1.0
                        else:
                            idx_i = np.where(mask_i)[0]
                            idx_j = np.where(mask_j)[0]
                            if len(idx_i) > 0 and len(idx_j) > 0:
                                consensus_subset = consensus_matrix[np.ix_(idx_i, idx_j)]
                                cluster_consensus[i, j] = np.mean(consensus_subset)

                distance_matrix = 1 - cluster_consensus
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)

                condensed_dist = squareform(distance_matrix, checks=False)
                linkage_matrix = linkage(condensed_dist, method="average")
                cluster_order = leaves_list(linkage_matrix).tolist()

                self.logger.info(f"Reordered clusters: {cluster_order}")

            except Exception as e:
                self.logger.warning(f"Could not reorder clusters: {e}")

        z_scores_ordered = z_scores[cluster_order]
        cluster_means_ordered = cluster_means[cluster_order]

        n_features = len(self.config.continuous_columns)
        adaptive_height = self._calculate_adaptive_height(
            n_features=n_features,
            base_height=250,
            height_per_feature=30,
            min_height=350,
            max_height=900,
        )

        # Create heatmap text with values and significance
        hover_text = []
        display_text = []
        for i in range(len(cluster_order)):
            hover_row = []
            display_row = []
            for j, col in enumerate(self.config.continuous_columns):
                raw_mean = cluster_means_ordered[i, j]
                z = z_scores_ordered[i, j]
                n = cluster_stats[cluster_order[i]]["n"]
                hover_row.append(f"{col}<br>Mean: {raw_mean:.2f}<br>Z-score: {z:.2f}<br>N: {n}")
                display_row.append(f"{z:.2f}")
            hover_text.append(hover_row)
            display_text.append(display_row)

        # Create figure
        fig = go.Figure(
            data=go.Heatmap(
                z=z_scores_ordered,
                x=self.config.continuous_columns,
                y=[
                    f"Phenotype {cluster_order[i]} (n={cluster_stats[cluster_order[i]]['n']})"
                    for i in range(self.n_clusters)
                ],
                colorscale=STYLE["diverging_colorscale"],
                zmid=0,
                zmin=-2.5,
                zmax=2.5,
                text=display_text,
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                colorbar=dict(
                    title=dict(text="Z-score", font=dict(size=11)),
                    tickfont=dict(size=10),
                ),
            )
        )

        # Dynamic bottom margin based on longest variable name at 45-deg angle
        max_var_len = max(len(c) for c in self.config.continuous_columns)
        bottom_margin = max(140, min(300, int(max_var_len * 5.5) + 40))

        fig = apply_standard_layout(
            fig, title=title, height=adaptive_height, bottom_margin=bottom_margin
        )
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Phenotype",
            xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        )

        self.logger.info(f"Enhanced heatmap created (height: {adaptive_height}px)")
        return fig

    # CATEGORICAL VARIABLE PLOTS

    def _group_variables_by_prefix(self) -> Dict[str, List[str]]:
        """Group categorical variables by prefix."""
        if not self.config.categorical_flow.group_by_prefix:
            return {"All Variables": self.config.categorical_columns}

        if self.config.categorical_flow.custom_groups:
            grouped = {}
            for (
                group_name,
                patterns,
            ) in self.config.categorical_flow.custom_groups.items():
                vars_in_group = []
                for pattern in patterns:
                    if pattern.endswith("*"):
                        prefix = pattern[:-1]
                        vars_in_group.extend(
                            [
                                col
                                for col in self.config.categorical_columns
                                if col.startswith(prefix)
                            ]
                        )
                    else:
                        if pattern in self.config.categorical_columns:
                            vars_in_group.append(pattern)

                if vars_in_group:
                    grouped[group_name] = vars_in_group

            all_grouped = [v for vars_list in grouped.values() for v in vars_list]
            ungrouped = [v for v in self.config.categorical_columns if v not in all_grouped]
            if ungrouped:
                grouped["Other"] = ungrouped

            return grouped

        separator = self.config.categorical_flow.prefix_separator
        grouped = {}

        for col in self.config.categorical_columns:
            if separator in col:
                prefix = col.split(separator)[0]
                if prefix not in grouped:
                    grouped[prefix] = []
                grouped[prefix].append(col)
            else:
                if "Other" not in grouped:
                    grouped["Other"] = []
                grouped["Other"].append(col)

        return grouped

    def _assign_gradient_category_colors(self, df: pd.DataFrame, variables: list) -> dict:
        """Assign gradient colors to categories within each variable."""
        color_map = {}
        if not variables:
            return color_map
        hues = np.linspace(0, 1, len(variables), endpoint=False)
        for hue, var in zip(hues, variables):
            if var not in df.columns:
                continue
            cats = sorted([str(c) for c in df[var].dropna().unique()])
            if not cats:
                continue
            shades = np.linspace(0.90, 0.35, len(cats))
            var_colors = {}
            for cat, shade in zip(cats, shades):
                r, g, b = colorsys.hsv_to_rgb(hue, 0.65, shade)
                var_colors[cat] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            color_map[var] = var_colors
        return color_map

    def _hex_to_rgba(self, hex_color: str, alpha: float = 0.65) -> str:
        """Convert hex color to rgba string."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    def _create_sankey_diagram(
        self, df: pd.DataFrame, labels: np.ndarray, variables: list, group_name: str
    ) -> go.Figure:
        """Create Sankey diagram for categorical variables to clusters."""
        variables = [v for v in variables if v in df.columns]
        if not variables:
            fig = go.Figure()
            fig.update_layout(title=f"Categorical -> Phenotype (No data): {group_name}")
            return fig

        color_map = self._assign_gradient_category_colors(df, variables)

        cat_nodes = []
        for var in variables:
            if var not in color_map:
                continue
            for cat in color_map[var].keys():
                cat_nodes.append(f"{var} - {cat}")

        cluster_nodes = [f"Phenotype {i}" for i in range(self.n_clusters)]
        node_labels = cat_nodes + cluster_nodes
        node_colors = ["#BBBBBB"] * len(cat_nodes) + self.cluster_colors

        node_index = {label: i for i, label in enumerate(node_labels)}

        source, target, value, link_colors = [], [], [], []

        for var in variables:
            if var not in color_map:
                continue
            sub = df[[var]].copy()
            sub["cluster"] = labels
            for cluster_id in range(self.n_clusters):
                tgt_idx = node_index[cluster_nodes[cluster_id]]
                for cat, hex_color in color_map[var].items():
                    src_idx = node_index[f"{var} - {cat}"]
                    count = ((sub[var].astype(str) == cat) & (sub["cluster"] == cluster_id)).sum()
                    if count > 0:
                        source.append(src_idx)
                        target.append(tgt_idx)
                        value.append(int(count))
                        link_colors.append(self._hex_to_rgba(hex_color, alpha=0.65))

        if not value:
            fig = go.Figure()
            fig.update_layout(title=f"Categorical -> Phenotype (No links): {group_name}")
            return fig

        # Truncate long node labels (full name available in hover)
        display_labels = [(lbl[:22] + "...") if len(lbl) > 25 else lbl for lbl in node_labels]

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=25,
                        thickness=20,
                        line=dict(color="black", width=0.6),
                        label=display_labels,
                        customdata=node_labels,
                        hovertemplate="%{customdata}<extra></extra>",
                        color=node_colors,
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=link_colors,
                    ),
                )
            ]
        )

        fig = apply_standard_layout(
            fig,
            title=f"Categorical Variables -> Phenotype: {group_name}",
            height=max(800, len(variables) * 35),
        )

        return fig

    def create_categorical_flow_plots(
        self, df: pd.DataFrame, labels: np.ndarray
    ) -> Dict[str, go.Figure]:
        """
        Create categorical flow visualizations.

        Based on config settings, creates either:
        - Proportion heatmaps (cleaner, recommended)
        - Sankey diagrams (optional, can be cluttered)
        """
        if not self.config.categorical_columns:
            self.logger.info("No categorical columns for flow plots.")
            return {}

        figures = {}

        # Create proportion heatmap (recommended, cleaner)
        if self.config.categorical_flow.show_proportion_heatmap:
            self.logger.info("Creating categorical proportion heatmap...")
            heatmap = self.create_categorical_heatmap(df, labels)
            if heatmap:
                figures["categorical_heatmap"] = heatmap

        # Create Sankey diagrams (optional, can be hard to interpret)
        if self.config.categorical_flow.show_sankey:
            self.logger.info("Creating Sankey diagrams...")
            var_groups = self._group_variables_by_prefix()
            for group_name, vars_list in var_groups.items():
                safe_name = re.sub(r"[^0-9a-zA-Z_]+", "_", group_name).strip("_") or "group"
                sankey_fig = self._create_sankey_diagram(df, labels, vars_list, group_name)
                figures[f"{safe_name}_sankey"] = sankey_fig
            self.logger.info(f"Created {len(var_groups)} Sankey diagram(s)")

        return figures

    def create_categorical_heatmap(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        title: str = "Categorical Variables by Phenotype",
    ) -> Optional[go.Figure]:
        """
        Create a heatmap showing proportions of categorical variables by cluster.

        This provides a cleaner visualization than Sankey diagrams, showing
        the proportion of each category within each phenotype.

        Parameters
        ----------
        df : pd.DataFrame
            Data with categorical variables
        labels : np.ndarray
            Cluster assignments
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure with categorical heatmap
        """
        if not self.config.categorical_columns:
            return None

        self.logger.info("Creating categorical proportion heatmap...")

        # Get minimum category percentage threshold
        min_pct = self.config.categorical_flow.min_category_pct

        # Build proportion data for each variable and category
        heatmap_data = []
        y_labels = []
        hover_texts = []

        for var in self.config.categorical_columns:
            if var not in df.columns:
                continue

            # Get categories, grouping small ones as "Other"
            value_counts = df[var].value_counts(normalize=True)
            main_cats = value_counts[value_counts >= min_pct].index.tolist()
            other_cats = value_counts[value_counts < min_pct].index.tolist()

            categories = main_cats
            if other_cats:
                categories = main_cats + ["Other"]

            for cat in categories:
                row_data = []
                hover_row = []

                for cluster_id in range(self.n_clusters):
                    cluster_mask = labels == cluster_id
                    cluster_data = df.loc[cluster_mask, var]
                    # Use non-NaN count as denominator to avoid underestimating proportions
                    total_valid = cluster_data.notna().sum()

                    if cat == "Other" and other_cats:
                        count = cluster_data.isin(other_cats).sum()
                    else:
                        count = (cluster_data == cat).sum()

                    proportion = count / total_valid if total_valid > 0 else 0
                    row_data.append(proportion * 100)  # Convert to percentage

                    hover_row.append(
                        f"{var}={cat}<br>"
                        f"Phenotype {cluster_id}<br>"
                        f"Count: {count}/{total_valid}<br>"
                        f"Proportion: {proportion:.1%}"
                    )

                heatmap_data.append(row_data)
                var_display = var.replace("_", " ")
                y_labels.append(f"{var_display}: {cat}")
                hover_texts.append(hover_row)

        if not heatmap_data:
            return None

        heatmap_array = np.array(heatmap_data)

        n_rows = len(y_labels)
        adaptive_height = self._calculate_adaptive_height(
            n_features=n_rows,
            base_height=200,
            height_per_feature=25,
            min_height=400,
            max_height=1500,
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_array,
                x=[f"Phenotype {i}" for i in range(self.n_clusters)],
                y=y_labels,
                colorscale="Blues",
                zmin=0,
                zmax=100,
                text=[[f"{v:.1f}%" for v in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertext=hover_texts,
                hovertemplate="%{hovertext}<extra></extra>",
                colorbar=dict(
                    title=dict(text="%", font=dict(size=11)),
                    tickfont=dict(size=10),
                    ticksuffix="%",
                ),
            )
        )

        # Dynamic left margin based on longest y-label
        max_label_len = max(len(lbl) for lbl in y_labels) if y_labels else 20
        left_margin = max(200, min(400, max_label_len * 6 + 30))
        tick_size = 8 if n_rows > 40 else 9

        fig = apply_standard_layout(fig, title=title, height=adaptive_height)
        fig.update_layout(
            xaxis_title="Phenotype",
            yaxis_title="Variable: Category",
            yaxis=dict(tickfont=dict(size=tick_size)),
            margin=dict(l=left_margin),
        )

        # Add variable group separators
        current_var = None
        separator_positions = []
        for i, label in enumerate(y_labels):
            var_name = label.split(":")[0].strip()
            if current_var is not None and var_name != current_var:
                separator_positions.append(i - 0.5)
            current_var = var_name

        for pos in separator_positions:
            fig.add_hline(y=pos, line_color="white", line_width=2)

        self.logger.info(f"Categorical heatmap created ({n_rows} rows)")
        return fig
