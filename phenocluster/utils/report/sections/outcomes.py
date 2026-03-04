"""Outcome associations section."""

from pathlib import Path
from typing import Dict

from .._helpers import embed_plot, empty_section, fmt_p, format_ci, format_value, safe


def generate_outcomes_section(data: Dict, results_dir: Path) -> str:
    """Generate outcomes association section."""
    outcome_results = data.get("outcome_results", {})

    if not outcome_results:
        return empty_section(
            "outcomes",
            "Outcome Associations",
            "Outcome analysis results not available.",
        )

    combined = outcome_results.get("full_cohort", outcome_results.get("train", {}))
    rows = _build_outcome_rows(combined)

    outcomes_table = (
        f"""
        <table>
            <tr><th>Outcome</th><th>Phenotype</th>
            <th>Prevalence</th><th>OR</th><th>95% CI</th>
            <th>p-value</th><th>q-value</th></tr>
            {"".join(rows)}
        </table>
    """
        if rows
        else "<p>No outcome comparisons available.</p>"
    )

    forest_plot = embed_plot(results_dir, "forest_plot")

    return f"""
    <section id="outcomes">
        <h2>Outcome Associations</h2>

        {outcomes_table}
        {forest_plot}
    </section>
"""


def _build_outcome_rows(combined: dict) -> list:
    """Build outcome table rows from combined results."""
    rows = []
    for outcome_name, outcome_data in combined.items():
        if not isinstance(outcome_data, dict):
            continue

        comparisons = outcome_data.get("comparisons", None)
        if comparisons is None:
            for cluster_id, cluster_data in outcome_data.items():
                if str(cluster_id).startswith("_"):
                    continue
                if not isinstance(cluster_data, dict):
                    continue
                or_val = cluster_data.get("OR")
                ci_lower = cluster_data.get("CI_lower", cluster_data.get("ci_lower"))
                ci_upper = cluster_data.get("CI_upper", cluster_data.get("ci_upper"))
                if or_val == 1.0 and ci_lower == 1.0:
                    continue

                rows.append(
                    _format_outcome_row(
                        outcome_name,
                        cluster_id,
                        or_val,
                        ci_lower,
                        ci_upper,
                        cluster_data,
                    )
                )
        else:
            for comp in comparisons:
                cluster = comp.get("cluster", comp.get("phenotype", ""))
                if "reference" in str(cluster).lower():
                    continue
                or_val = comp.get("OR")
                ci_lower = comp.get("CI_lower", comp.get("ci_lower", None))
                ci_upper = comp.get("CI_upper", comp.get("ci_upper", None))

                rows.append(
                    _format_outcome_row(
                        outcome_name,
                        cluster,
                        or_val,
                        ci_lower,
                        ci_upper,
                        comp,
                    )
                )

    return rows


def _format_outcome_row(outcome_name, cluster_id, or_val, ci_lower, ci_upper, data_dict) -> str:
    """Format a single outcome row."""
    prevalence = data_dict.get("prevalence")
    p_val = data_dict.get("p_value")
    q_val = data_dict.get("p_value_fdr", data_dict.get("q_value"))

    ci_str = format_ci(ci_lower, ci_upper, bracket="()")
    prev_str = f"{prevalence:.1%}" if isinstance(prevalence, (int, float)) else "N/A"
    p_str = fmt_p(p_val) if isinstance(p_val, (int, float)) else "N/A"
    q_str = fmt_p(q_val) if isinstance(q_val, (int, float)) else "N/A"

    return f"""
                    <tr>
                        <td>{safe(outcome_name.replace("_", " ").title())}</td>
                        <td>Phenotype {safe(cluster_id)}</td>
                        <td>{prev_str}</td>
                        <td>{format_value(or_val)}</td>
                        <td>{ci_str}</td>
                        <td>{p_str}</td>
                        <td>{q_str}</td>
                    </tr>
                """
