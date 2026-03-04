"""Survival analysis section."""

from pathlib import Path
from typing import Dict

from .._helpers import embed_plots_matching, empty_section, fmt_p, format_ci, format_value, safe


def generate_survival_section(data: Dict, results_dir: Path) -> str:
    """Generate survival analysis section."""
    survival_results = data.get("survival_results", {})

    if not survival_results:
        return empty_section(
            "survival",
            "Survival Analysis",
            "Survival analysis was not performed or results not available.",
        )

    survival_tables = []
    for target_name, target_data in survival_results.items():
        if target_name.startswith("_") or target_name.endswith("_weighted"):
            continue
        if not isinstance(target_data, dict):
            continue
        survival_tables.append(_build_survival_target(target_name, target_data))

    km_plots = embed_plots_matching(results_dir, "kaplan_meier")
    na_plots = embed_plots_matching(results_dir, "nelson_aalen")

    return f"""
    <section id="survival">
        <h2>Survival Analysis</h2>

        {"".join(survival_tables)}

        <h3>Kaplan-Meier Survival Curves</h3>
        {km_plots if km_plots else "<p>No Kaplan-Meier plots available.</p>"}

        <h3>Nelson-Aalen Cumulative Hazard</h3>
        {na_plots if na_plots else "<p>No Nelson-Aalen plots available.</p>"}
    </section>
"""


def _build_survival_target(target_name: str, target_data: dict) -> str:
    """Build HTML for a single survival target."""
    median_surv = target_data.get("median_survival", {})

    rows = []
    for cluster_id, median in median_surv.items():
        median_str = f"{median:.1f}" if median and median != float("inf") else "Not reached"
        survival_data = target_data.get("survival_data", {})
        cluster_data = survival_data.get(int(cluster_id), survival_data.get(cluster_id, {}))
        n_events = cluster_data.get("n_events", "N/A")
        n_patients = cluster_data.get("n_patients", "N/A")
        rows.append(f"""
                <tr>
                    <td>Phenotype {safe(cluster_id)}</td>
                    <td>{safe(n_patients)}</td>
                    <td>{safe(n_events)}</td>
                    <td>{median_str}</td>
                </tr>
            """)

    display_name = target_name.replace("_", " ").title()
    comparison_table = _build_comparison_table(target_data)

    return f"""
            <h3>{safe(display_name)}</h3>
            <table>
                <tr><th>Phenotype</th><th>N</th><th>Events</th><th>Median Time</th></tr>
                {"".join(rows)}
            </table>
            {comparison_table}
        """


def _build_comparison_table(target_data: dict) -> str:
    """Build Cox PH comparison table for a survival target."""
    comparison = target_data.get("comparison", {})
    comparison_rows = []
    for pheno_id, pheno_data in comparison.items():
        if not isinstance(pheno_data, dict):
            continue
        hr_val = pheno_data.get("HR", "N/A")
        ci_lower = pheno_data.get("CI_lower", pheno_data.get("ci_lower", "N/A"))
        ci_upper = pheno_data.get("CI_upper", pheno_data.get("ci_upper", "N/A"))
        p_val = pheno_data.get("p_value")
        q_val = pheno_data.get("p_value_fdr", pheno_data.get("q_value"))

        hr_str = format_value(hr_val) if isinstance(hr_val, (int, float)) else hr_val
        ci_str = format_ci(ci_lower, ci_upper)
        p_str = fmt_p(p_val) if isinstance(p_val, (int, float)) else "N/A"
        q_str = fmt_p(q_val) if isinstance(q_val, (int, float)) else "N/A"
        pheno_label = str(pheno_id).replace("_vs_", " vs ")

        comparison_rows.append(f"""
                <tr>
                    <td>Phenotype {safe(pheno_label)}</td>
                    <td>{hr_str}</td>
                    <td>{ci_str}</td>
                    <td>{p_str}</td>
                    <td>{q_str}</td>
                </tr>
            """)

    if not comparison_rows:
        return ""

    logrank_p = target_data.get("logrank_p_value")
    logrank_note = ""
    if logrank_p is not None:
        logrank_note = f"<p><strong>Overall log-rank test:</strong> p = {fmt_p(logrank_p)}</p>"

    return f"""
            {logrank_note}
            <table>
                <tr><th>Comparison</th><th>HR</th>
                <th>95% CI</th>
                <th>p-value</th>
                <th>q-value</th></tr>
                {"".join(comparison_rows)}
            </table>
            """
