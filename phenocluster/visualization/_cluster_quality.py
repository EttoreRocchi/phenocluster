"""
Cluster Quality Mixin
======================

Consensus matrix heatmap and entropy-based classification quality plots.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._base import STYLE, BaseVisualizer, apply_standard_layout


class ClusterQualityVisualizer(BaseVisualizer):
    """Classification quality and consensus matrix plots.

    Attributes inherited from BaseVisualizer:
        self.config
        self.n_clusters
        self.logger
        self.cluster_colors
    """

    # CLASSIFICATION QUALITY PLOT (Entropy-based)

    def create_classification_quality_plot(
        self,
        posterior_probs: np.ndarray,
        labels: np.ndarray,
        title: str = "Classification Quality",
    ) -> Optional[go.Figure]:
        """
        Create a plot showing classification quality based on posterior probabilities.

        Uses entropy-based metrics appropriate for latent class / profile analysis (LCA/LPA),
        not geometric distance-based measures like silhouette score.

        Parameters
        ----------
        posterior_probs : np.ndarray
            N x K matrix of posterior probabilities from the model
        labels : np.ndarray
            Modal cluster assignments
        title : str
            Plot title

        Returns
        -------
        go.Figure or None
            Plotly figure with classification quality analysis
        """
        if posterior_probs is None or len(posterior_probs) == 0:
            self.logger.info("No posterior probabilities for classification quality plot.")
            return None

        self.logger.info("Creating classification quality plot...")

        n_samples, n_classes = posterior_probs.shape

        # 1. Per-sample entropy: -sum(p * log(p))
        eps = 1e-10  # Avoid log(0)
        sample_entropy = -np.sum(posterior_probs * np.log(posterior_probs + eps), axis=1)
        max_entropy = np.log(n_classes)  # Maximum possible entropy

        # 2. Relative entropy (normalized: 0 = perfect, 1 = random)
        relative_entropy = sample_entropy / max_entropy

        # 3. Average posterior probability (AvePP) per cluster
        max_posteriors = np.max(posterior_probs, axis=1)

        # 4. Overall metrics
        overall_entropy = np.mean(relative_entropy)
        overall_avepp = np.mean(max_posteriors)

        near_perfect = overall_avepp > 0.99 and overall_entropy < 0.01

        if near_perfect:
            # Single-column visualization for near-perfect classification (AvePP only)
            # Sample size is already shown in cluster_distribution plot
            fig = go.Figure()

            # Bar chart of AvePP per cluster
            cluster_avepps = []
            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                cluster_max_post = max_posteriors[mask]
                cluster_avepps.append(np.mean(cluster_max_post))

            fig.add_trace(
                go.Bar(
                    x=[f"Phenotype {i}" for i in range(self.n_clusters)],
                    y=cluster_avepps,
                    marker_color=[self.cluster_colors[i] for i in range(self.n_clusters)],
                    text=[f"{v:.4f}" for v in cluster_avepps],
                    textposition="outside",
                )
            )

            fig = apply_standard_layout(fig, title=title, height=400, width=500, bottom_margin=60)

            fig.update_yaxes(title_text="Average Posterior Probability (AvePP)", range=[0, 1.05])
            fig.update_xaxes(title_text="Phenotype")

            # Add 0.7 threshold reference line
            fig.add_hline(
                y=0.7,
                line_dash="dash",
                line_color="gray",
                line_width=1,
                annotation_text="Acceptable threshold (0.7)",
                annotation_position="top right",
                annotation_font=dict(size=9, color="gray"),
            )

            # Annotate actual min AvePP
            min_avepp = min(cluster_avepps)
            fig.add_annotation(
                text=f"All phenotypes AvePP > {min_avepp:.4f}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.95,
                showarrow=False,
                font=dict(size=STYLE["annotation_font_size"], color="green"),
            )

            fig.update_layout(showlegend=False)
        else:
            # Standard visualization
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=[
                    "Distribution of Maximum Posterior Probability",
                    "Relative Entropy by Phenotype",
                ],
                horizontal_spacing=0.12,
            )

            # Plot 1: Histogram of max posterior probabilities by cluster
            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                cluster_max_post = max_posteriors[mask]
                cluster_avepp = np.mean(cluster_max_post)

                fig.add_trace(
                    go.Histogram(
                        x=cluster_max_post,
                        name=f"P{cluster_id} (AvePP: {cluster_avepp:.3f})",
                        marker_color=self.cluster_colors[cluster_id],
                        opacity=0.7,
                        nbinsx=20,
                    ),
                    row=1,
                    col=1,
                )

            # Add vertical line at 0.7 (common threshold for acceptable classification)
            fig.add_vline(x=0.7, line_dash="dash", line_color="red", line_width=1.5, row=1, col=1)

            # Plot 2: Box plot of relative entropy by cluster
            for cluster_id in range(self.n_clusters):
                mask = labels == cluster_id
                cluster_entropy = relative_entropy[mask]

                fig.add_trace(
                    go.Box(
                        y=cluster_entropy,
                        name=f"Phenotype {cluster_id}",
                        marker_color=self.cluster_colors[cluster_id],
                        boxpoints="outliers",
                    ),
                    row=1,
                    col=2,
                )

            fig = apply_standard_layout(fig, title=title, height=450, width=800, bottom_margin=60)

            # Update axes
            fig.update_xaxes(title_text="Maximum Posterior Probability", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_xaxes(title_text="Phenotype", row=1, col=2)
            fig.update_yaxes(title_text="Relative Entropy", range=[0, 1], row=1, col=2)

            fig.update_layout(
                barmode="overlay",
                legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
            )

        self.logger.info(
            f"Classification quality plot created "
            f"(AvePP: {overall_avepp:.3f}, Entropy: {overall_entropy:.3f})"
        )
        return fig

    # CONSENSUS MATRIX HEATMAP

    def create_consensus_matrix_plot(
        self,
        consensus_matrix: np.ndarray,
        labels: np.ndarray,
        title: str = "Consensus Matrix (Cluster Stability)",
    ) -> go.Figure:
        """
        Create a heatmap of the consensus matrix from stability analysis.

        Parameters
        ----------
        consensus_matrix : np.ndarray
            N x N matrix where entry (i,j) is the proportion of times
            samples i and j were assigned to the same cluster
        labels : np.ndarray
            Cluster assignments for ordering samples
        title : str
            Plot title

        Returns
        -------
        go.Figure
            Plotly figure with consensus matrix heatmap
        """
        self.logger.info("Creating consensus matrix plot...")

        # Sort samples by cluster assignment for better visualization
        sorted_indices = np.argsort(labels)
        sorted_matrix = consensus_matrix[np.ix_(sorted_indices, sorted_indices)]
        sorted_labels = labels[sorted_indices]

        # Find cluster boundaries
        boundaries = []
        current_cluster = sorted_labels[0]
        for i, label in enumerate(sorted_labels):
            if label != current_cluster:
                boundaries.append(i)
                current_cluster = label

        # Downsample large matrices for readability
        n_samples = sorted_matrix.shape[0]
        averaging_note = ""
        if n_samples > 80:
            block_size = max(2, n_samples // 80)
            n_blocks = n_samples // block_size
            display_matrix = np.zeros((n_blocks, n_blocks))
            for i in range(n_blocks):
                for j in range(n_blocks):
                    block = sorted_matrix[
                        i * block_size : (i + 1) * block_size,
                        j * block_size : (j + 1) * block_size,
                    ]
                    display_matrix[i, j] = np.mean(block)
            # Scale boundaries to block coordinates
            boundaries = [b // block_size for b in boundaries]
            boundaries = sorted(set(boundaries))  # deduplicate
            averaging_note = f" (averaged over {block_size}x{block_size} blocks)"
        else:
            display_matrix = sorted_matrix

        fig = go.Figure()

        # Main heatmap
        fig.add_trace(
            go.Heatmap(
                z=display_matrix,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                colorbar=dict(
                    title=dict(text="Co-clustering<br>Probability", font=dict(size=11)),
                    tickfont=dict(size=10),
                ),
                hovertemplate=("Co-clustering: %{z:.3f}" + averaging_note + "<extra></extra>"),
            )
        )

        # Add cluster boundary lines
        for boundary in boundaries:
            fig.add_hline(y=boundary - 0.5, line_color="white", line_width=2)
            fig.add_vline(x=boundary - 0.5, line_color="white", line_width=2)

        fig = apply_standard_layout(fig, title=title, height=600, width=700, bottom_margin=100)
        fig.update_layout(
            xaxis_title="Samples (sorted by phenotype)",
            yaxis_title="Samples (sorted by phenotype)",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False, autorange="reversed"),
        )

        # Calculate and display average consensus within clusters
        within_cluster_consensus = []
        for c in range(self.n_clusters):
            mask = sorted_labels == c
            if np.sum(mask) > 1:
                cluster_matrix = sorted_matrix[np.ix_(mask, mask)]
                # Exclude diagonal
                n = cluster_matrix.shape[0]
                off_diag = cluster_matrix[~np.eye(n, dtype=bool)]
                within_cluster_consensus.append(np.mean(off_diag))

        avg_consensus = np.mean(within_cluster_consensus) if within_cluster_consensus else 0

        self.logger.info(f"Consensus matrix plot created (avg consensus: {avg_consensus:.3f})")
        return fig
