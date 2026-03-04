"""External validation section."""

from pathlib import Path
from typing import Dict

from .._helpers import embed_plot, embed_plots_matching, fmt_p, format_ci, format_value, safe
from .classification_quality import _render_classification_quality_block


def generate_external_validation_section(data: Dict, results_dir: Path) -> str:
    """Generate external validation section comparing derivation and external cohorts."""
    ext_results = data.get("external_validation_results", {})

    if not ext_results:
        return ""

    metric_cards = _build_metric_cards(ext_results)
    ext_cq_section = _build_ext_cq_section(ext_results)
    distribution_table = _build_distribution_table(ext_results)
    outcome_section = _build_outcome_section(ext_results)
    survival_section = _build_survival_section(ext_results, data, results_dir)
    multistate_section = _build_multistate_section(ext_results, data, results_dir)

    return f"""
    <section id="external_validation">
        <h2>External Validation</h2>
        {metric_cards}
        {ext_cq_section}
        {distribution_table}
        {outcome_section}
        {survival_section}
        {multistate_section}
    </section>
"""


def _build_metric_cards(ext_results: dict) -> str:
    """Build metric cards for external validation."""
    n_samples = ext_results.get("n_samples", "N/A")
    log_likelihood = ext_results.get("log_likelihood")

    metric_cards = f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="value">{safe(n_samples)}</div>
                <div class="label">External Cohort Size</div>
            </div>
    """
    if log_likelihood is not None:
        metric_cards += f"""
            <div class="metric-card">
                <div class="value">{format_value(log_likelihood)}</div>
                <div class="label">External LL/sample</div>
            </div>
        """
    ext_cq = ext_results.get("classification_quality", {})
    if ext_cq:
        metric_cards += f"""
            <div class="metric-card">
                <div class="value">{format_value(ext_cq.get("overall_avepp"))}</div>
                <div class="label">External AvePP</div>
            </div>
        """
    metric_cards += "</div>"
    return metric_cards


def _build_ext_cq_section(ext_results: dict) -> str:
    """Build external classification quality section."""
    ext_cq = ext_results.get("classification_quality", {})
    if not ext_cq:
        return ""
    return _render_classification_quality_block(ext_cq, "Classification Quality (External Cohort)")


def _build_distribution_table(ext_results: dict) -> str:
    """Build cluster distribution comparison table."""
    ext_dist = ext_results.get("cluster_distribution", {})
    der_dist = ext_results.get("derivation_distribution") or {}

    if not ext_dist:
        return ""

    rows = []
    for k in sorted(ext_dist.keys(), key=lambda x: int(x)):
        ext_info = ext_dist[k]
        der_info = der_dist.get(int(k), der_dist.get(str(k), {}))

        ext_pct = ext_info.get("percentage", 0)
        ext_n = ext_info.get("count", 0)
        der_pct = der_info.get("percentage", 0) if der_info else 0
        der_n = der_info.get("count", 0) if der_info else 0
        diff = ext_pct - der_pct if der_info else None

        if diff is not None:
            diff_class = "val-positive" if abs(diff) < 5 else "val-warning"
            diff_cell = f'<td class="{diff_class}">{diff:+.1f}%</td>'
        else:
            diff_cell = "<td>N/A</td>"

        rows.append(
            f"<tr><td>Phenotype {safe(k)}</td>"
            f"<td>{der_n}</td><td>{der_pct:.1f}%</td>"
            f"<td>{ext_n}</td><td>{ext_pct:.1f}%</td>"
            f"{diff_cell}</tr>"
        )

    return f"""
        <h3>Cluster Distribution Comparison</h3>
        <table>
            <thead>
                <tr>
                    <th>Phenotype</th>
                    <th>Derivation n</th><th>Derivation %</th>
                    <th>External n</th><th>External %</th>
                    <th>Difference</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        """


def _build_outcome_section(ext_results: dict) -> str:
    """Build outcome comparison section."""
    outcome_comparison = ext_results.get("outcome_comparison", {})
    if not outcome_comparison:
        return ""

    outcome_rows = []
    for outcome, comp in outcome_comparison.items():
        deriv_prev = comp.get("derivation_prevalence", {})
        ext_sizes = comp.get("external_cluster_sizes", {})

        for cluster_id in sorted(ext_sizes.keys(), key=lambda x: int(x)):
            d_prev = deriv_prev.get(str(cluster_id), deriv_prev.get(int(cluster_id)))
            e_n = ext_sizes.get(cluster_id, ext_sizes.get(str(cluster_id), 0))
            d_prev_str = f"{float(d_prev) * 100:.1f}%" if d_prev is not None else "N/A"

            outcome_rows.append(
                f"<tr><td>{safe(outcome)}</td><td>Phenotype {safe(cluster_id)}</td>"
                f"<td>{d_prev_str}</td><td>{e_n}</td></tr>"
            )

    if not outcome_rows:
        return ""

    return f"""
            <h3>Outcome Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Outcome</th><th>Phenotype</th>
                        <th>Derivation Prevalence</th><th>External n</th>
                    </tr>
                </thead>
                <tbody>{"".join(outcome_rows)}</tbody>
            </table>
            """


def _build_survival_section(ext_results: dict, data: dict, results_dir: Path) -> str:
    """Build external survival analysis section."""
    ext_survival = ext_results.get("survival_results", {})
    if not ext_survival:
        return ""

    survival_tables = []
    for target_name, target_data in ext_survival.items():
        if not isinstance(target_data, dict):
            continue
        survival_tables.append(_build_ext_survival_target(target_name, target_data))

    ext_km_plots = embed_plots_matching(results_dir, "ext_kaplan_meier")
    ext_na_plots = embed_plots_matching(results_dir, "ext_nelson_aalen")

    return f"""
        <h3>Survival Analysis (External Cohort)</h3>
        {"".join(survival_tables)}

        {f"<h4>Kaplan-Meier Curves</h4>{ext_km_plots}" if ext_km_plots else ""}
        {f"<h4>Nelson-Aalen Cumulative Hazard</h4>{ext_na_plots}" if ext_na_plots else ""}
        """


def _build_ext_survival_target(target_name: str, target_data: dict) -> str:
    """Build HTML for a single external survival target."""
    median_surv = target_data.get("median_survival", {})
    rows = []
    for cluster_id, median in median_surv.items():
        median_str = f"{median:.1f}" if median and median != float("inf") else "Not reached"
        survival_data = target_data.get("survival_data", {})
        cluster_data = survival_data.get(int(cluster_id), survival_data.get(cluster_id, {}))
        n_events = cluster_data.get("n_events", "N/A")
        n_patients = cluster_data.get("n_patients", "N/A")
        rows.append(
            f"<tr><td>Phenotype {safe(cluster_id)}</td>"
            f"<td>{safe(n_patients)}</td><td>{safe(n_events)}</td>"
            f"<td>{median_str}</td></tr>"
        )

    display_name = target_name.replace("_", " ").title()

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

        comparison_rows.append(
            f"<tr><td>Phenotype {safe(pheno_label)}</td>"
            f"<td>{hr_str}</td><td>{ci_str}</td>"
            f"<td>{p_str}</td><td>{q_str}</td></tr>"
        )

    logrank_p = target_data.get("logrank_p_value")
    logrank_note = ""
    if logrank_p is not None:
        logrank_note = f"<p><strong>Overall log-rank test:</strong> p = {fmt_p(logrank_p)}</p>"

    comparison_table = ""
    if comparison_rows:
        comparison_table = f"""
                {logrank_note}
                <table>
                    <tr><th>Comparison</th><th>HR</th>
                    <th>95% CI</th><th>p-value</th><th>q-value</th></tr>
                    {"".join(comparison_rows)}
                </table>
                """

    return f"""
                <h4>{safe(display_name)}</h4>
                <table>
                    <tr><th>Phenotype</th><th>N</th>
                    <th>Events</th><th>Median Time</th></tr>
                    {"".join(rows)}
                </table>
                {comparison_table}
            """


def _build_multistate_section(ext_results: dict, data: dict, results_dir: Path) -> str:
    """Build external multistate analysis section."""
    ext_multistate = ext_results.get("multistate_results", {})
    if not ext_multistate:
        return ""

    transition_results = ext_multistate.get("transition_results", {})
    pathway_results = ext_multistate.get("pathway_results", [])

    transition_rows = []
    for trans_name, trans_data in transition_results.items():
        n_events = trans_data.get("n_events", "N/A")
        n_at_risk = trans_data.get("n_at_risk", "N/A")
        phenotype_effects = trans_data.get("phenotype_effects", {})

        for pheno_id, effects in phenotype_effects.items():
            ref = data.get("reference_phenotype", 0)
            if str(pheno_id) == str(ref):
                continue
            hr = effects.get("HR", "N/A")
            ci_lower = effects.get("CI_lower", effects.get("ci_lower", "N/A"))
            ci_upper = effects.get("CI_upper", effects.get("ci_upper", "N/A"))
            p_val = effects.get("p_value")
            q_val = effects.get("p_value_fdr", effects.get("q_value"))

            hr_str = format_value(hr) if isinstance(hr, (int, float)) else hr
            ci_str = format_ci(ci_lower, ci_upper)
            p_str = fmt_p(p_val) if isinstance(p_val, (int, float)) else "N/A"
            q_str = fmt_p(q_val) if isinstance(q_val, (int, float)) else "N/A"

            transition_rows.append(
                f"<tr><td>{safe(trans_name.replace('_', ' ').title())}</td>"
                f"<td>Phenotype {safe(pheno_id)}</td>"
                f"<td>{hr_str}</td><td>{ci_str}</td>"
                f"<td>{p_str}</td><td>{q_str}</td>"
                f"<td>{n_events}/{n_at_risk}</td></tr>"
            )

    transition_table = ""
    if transition_rows:
        transition_table = f"""
            <table>
                <tr><th>Transition</th><th>Comparison</th>
                <th>HR</th><th>95% CI</th>
                <th>p-value</th><th>q-value</th>
                <th>Events/At Risk</th></tr>
                {"".join(transition_rows)}
            </table>
            """

    pathway_table = ""
    if pathway_results:
        pathway_rows = []
        for pr in pathway_results[:10]:
            pathway_str = " -> ".join(pr.get("state_names", []))
            total = pr.get("total_count", 0)
            pathway_rows.append(f"<tr><td>{safe(pathway_str)}</td><td>{safe(total)}</td></tr>")
        if pathway_rows:
            pathway_table = f"""
                <h4>Top Pathway Frequencies</h4>
                <table>
                    <tr><th>Pathway</th><th>Total</th></tr>
                    {"".join(pathway_rows)}
                </table>
                """

    ext_hazard_plot = embed_plot(results_dir, "ext_multistate_transition_hazards")
    ext_state_occ_plot = embed_plot(results_dir, "ext_multistate_state_occupation_uncertainty")

    return f"""
        <h3>Multistate Analysis (External Cohort)</h3>
        {f"<h4>Transition Hazard Ratios</h4>{transition_table}" if transition_table else ""}
        {ext_hazard_plot}
        {ext_state_occ_plot}
        {pathway_table}
        """
