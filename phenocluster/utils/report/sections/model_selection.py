"""Model selection section."""

from pathlib import Path
from typing import Dict

from .._helpers import embed_plot, format_value, safe


def generate_model_selection_section(data: Dict, results_dir: Path) -> str:
    """Generate model selection section."""
    # Skip entirely when model selection is disabled
    config = data.get("config", {})
    selection_cfg = config.get("model", {}).get("selection", {})
    if not selection_cfg.get("enabled", True):
        return ""

    model_sel = data.get("model_selection", {})
    model_results = data.get("model_selection_results")

    results_table = ""
    if model_results is not None and len(model_results) > 0:
        rows = []
        for _, row in model_results.iterrows():
            k = int(row.get("n_clusters", row.get("k", 0)))
            score = row.get("mean_score", row.get("score", None))
            rank = row.get("rank", "")
            badge = ' <span class="badge badge-primary">Selected</span>' if rank == 1 else ""
            rows.append(
                f"<tr><td>{safe(k)}{badge}</td><td>{format_value(score)}</td><td>{safe(rank)}</td></tr>"
            )
        results_table = f"""
        <table>
            <tr><th>Clusters</th>
            <th>Score ({
            safe(model_sel.get("criterion_used", model_sel.get("criterion", "BIC")))
        })</th>
            <th>Rank</th></tr>
            {"".join(rows)}
        </table>
"""

    plot_html = embed_plot(results_dir, "model_selection")
    return f"""
    <section id="model">
        <h2>Model Selection</h2>

        {results_table}
        {plot_html}
    </section>
"""
