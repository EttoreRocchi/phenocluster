"""Cluster stability section."""

from pathlib import Path
from typing import Dict

from .._helpers import embed_plot, format_value, safe


def generate_stability_section(data: Dict, results_dir: Path) -> str:
    """Generate stability analysis section."""
    stability_results = data.get("stability_results", {})

    if not stability_results:
        return """
    <section id="stability">
        <h2>Cluster Stability</h2>
        <p>Stability analysis was not enabled or results not available.</p>
    </section>
"""

    mean_consensus = stability_results.get("mean_consensus", None)
    n_runs = stability_results.get("n_runs", "N/A")
    silhouette_mean = stability_results.get("silhouette_mean", None)
    silhouette_std = stability_results.get("silhouette_std", None)

    cluster_stability = stability_results.get("cluster_stability", {})
    cluster_rows = []
    for cluster_id, cluster_data in sorted(
        cluster_stability.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0
    ):
        if isinstance(cluster_data, dict):
            stability = cluster_data.get("mean_consistency", "N/A")
            size = cluster_data.get("size", cluster_data.get("n_samples", "N/A"))
            cluster_rows.append(f"""
                <tr>
                    <td>Phenotype {safe(cluster_id)}</td>
                    <td>{format_value(stability)}</td>
                    <td>{size}</td>
                </tr>
            """)

    cluster_table = ""
    if cluster_rows:
        cluster_table = f"""
        <h3>Per-Cluster Stability</h3>
        <table>
            <tr><th>Phenotype</th><th>Stability</th><th>Size</th></tr>
            {"".join(cluster_rows)}
        </table>
        """

    consensus_plot = embed_plot(results_dir, "consensus_matrix")

    silhouette_section = ""
    if silhouette_mean is not None:
        silhouette_section = f"""
        <div class="metric-card">
            <div class="value">{format_value(silhouette_mean)}</div>
            <div class="label">Silhouette Score
            (&plusmn;{format_value(silhouette_std) if silhouette_std else "N/A"})</div>
        </div>
        """

    return f"""
    <section id="stability">
        <h2>Cluster Stability</h2>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="value">
                    {format_value(mean_consensus) if mean_consensus else "N/A"}
                </div>
                <div class="label">Mean Consensus</div>
            </div>
            <div class="metric-card">
                <div class="value">{n_runs}</div>
                <div class="label">Subsampling Runs</div>
            </div>
            {silhouette_section}
        </div>

        {cluster_table}
        {consensus_plot}
    </section>
"""
