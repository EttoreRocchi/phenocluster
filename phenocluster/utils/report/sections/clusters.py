"""Phenotype characteristics section."""

from pathlib import Path
from typing import Dict

from .._helpers import embed_plot, embed_plots_matching, safe


def generate_cluster_section(data: Dict, results_dir: Path) -> str:
    """Generate cluster/phenotype characteristics section."""
    cluster_stats = data.get("cluster_stats", {})

    if not cluster_stats:
        return """
    <section id="clusters">
        <h2>Phenotype Characteristics</h2>
        <p>Cluster statistics not available.</p>
    </section>
"""

    dist_rows = []
    total_samples = sum(
        s.get("n_samples", s.get("size", s.get("n", 0)))
        for s in cluster_stats.values()
        if isinstance(s, dict)
    )

    for cluster_id, stats in sorted(
        cluster_stats.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0
    ):
        if not isinstance(stats, dict):
            continue
        n = stats.get("n_samples", stats.get("size", stats.get("n", "N/A")))
        pct = stats.get(
            "percentage",
            (n / total_samples * 100) if total_samples and isinstance(n, (int, float)) else 0,
        )
        dist_rows.append(
            f"<tr><td>Phenotype {safe(cluster_id)}</td><td>{safe(n)}</td><td>{pct:.1f}%</td></tr>"
        )

    dist_plot = embed_plot(results_dir, "cluster_distribution")
    heatmap_plot = embed_plot(results_dir, "heatmap")
    categorical_heatmap_plot = embed_plot(results_dir, "categorical_heatmap")
    sankey_plots = embed_plots_matching(results_dir, "sankey")

    return f"""
    <section id="clusters">
        <h2>Phenotype Characteristics</h2>

        <h3>Distribution</h3>
        <table>
            <tr><th>Phenotype</th><th>N</th><th>Percentage</th></tr>
            {"".join(dist_rows)}
        </table>

        {dist_plot}

        <h3>Continuous Variable Profiles</h3>
        {heatmap_plot}

        <h3>Categorical Variable Profiles</h3>
        {categorical_heatmap_plot}
        {sankey_plots}
    </section>
"""
