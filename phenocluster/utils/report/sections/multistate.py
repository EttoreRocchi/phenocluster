"""Multistate analysis section."""

from pathlib import Path
from typing import Dict

from .._helpers import embed_plot, empty_section, fmt_p, format_ci, format_value, safe


def generate_multistate_section(data: Dict, results_dir: Path) -> str:
    """Generate multistate analysis section."""
    multistate_results = data.get("multistate_results", {})

    if not multistate_results:
        return empty_section(
            "multistate",
            "Multistate Analysis",
            "Multistate analysis was not enabled or results not available.",
        )

    model_summary = multistate_results.get("model_summary", {})
    transition_results = multistate_results.get("transition_results", {})
    pathway_results = multistate_results.get("pathway_results", [])

    transition_table = _build_transition_table(transition_results, data)
    pathway_table = _build_pathway_table(pathway_results)

    pathway_plot = embed_plot(results_dir, "multistate_pathways")
    hazard_plot = embed_plot(results_dir, "multistate_transition_hazards")
    state_occ_plot = embed_plot(results_dir, "multistate_state_occupation_uncertainty")
    state_diagram_plot = embed_plot(results_dir, "multistate_state_diagram")

    n_patients = model_summary.get("n_patients", "N/A")
    n_transitions = len(transition_results)
    n_pathways = len(pathway_results)
    state_occ_section = _build_state_occ_section(
        multistate_results,
        state_occ_plot,
    )
    state_diagram_section = ""
    if state_diagram_plot:
        state_diagram_section = f"""
        <h3>State Diagram</h3>
        {state_diagram_plot}
        """

    return f"""
    <section id="multistate">
        <h2>Multistate Analysis</h2>

        <h3>Summary</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="value">{safe(n_patients)}</div>
                <div class="label">Patients</div>
            </div>
            <div class="metric-card">
                <div class="value">{n_transitions}</div>
                <div class="label">Transition Types</div>
            </div>
            <div class="metric-card">
                <div class="value">{n_pathways}</div>
                <div class="label">Unique Pathways</div>
            </div>
        </div>

        {state_diagram_section}

        <h3>Transition Hazard Ratios</h3>
        {transition_table}
        {hazard_plot}

        {state_occ_section}

        <h3>Pathway Frequencies</h3>
        {pathway_table}
        {pathway_plot}
    </section>
"""


def _build_transition_table(transition_results: dict, data: dict) -> str:
    """Build transition hazard ratio table."""
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

            transition_rows.append(f"""
                <tr>
                    <td>{safe(trans_name.replace("_", " ").title())}</td>
                    <td>Phenotype {safe(pheno_id)}</td>
                    <td>{hr_str}</td>
                    <td>{ci_str}</td>
                    <td>{p_str}</td>
                    <td>{q_str}</td>
                    <td>{n_events}/{n_at_risk}</td>
                </tr>
            """)

    return f"""
        <table>
            <tr>
                <th>Transition</th>
                <th>Comparison</th>
                <th>HR</th>
                <th>95% CI</th>
                <th>p-value</th>
                <th>q-value</th>
                <th>Events/At Risk</th>
            </tr>
            {
        "".join(transition_rows)
        if transition_rows
        else ('<tr><td colspan="7">No transition results available</td></tr>')
    }
        </table>
    """


def _build_pathway_table(pathway_results: list) -> str:
    """Build pathway frequency table."""
    pathway_rows = []
    for pr in pathway_results[:10]:
        pathway_str = " -> ".join(pr.get("state_names", []))
        total = pr.get("total_count", 0)
        counts = pr.get("counts_by_phenotype", {})

        count_cells = " ".join(
            [f"<td>{counts.get(str(i), counts.get(i, 0))}</td>" for i in range(len(counts))]
        )

        pathway_rows.append(f"""
            <tr>
                <td>{safe(pathway_str)}</td>
                <td>{total}</td>
                {count_cells}
            </tr>
        """)

    n_phenotypes = len(pathway_results[0].get("counts_by_phenotype", {})) if pathway_results else 0
    pheno_headers = " ".join([f"<th>P{i}</th>" for i in range(n_phenotypes)])

    return f"""
        <table>
            <tr>
                <th>Pathway</th>
                <th>Total</th>
                {pheno_headers}
            </tr>
            {
        "".join(pathway_rows)
        if pathway_rows
        else ('<tr><td colspan="100%">No pathway results available</td></tr>')
    }
        </table>
    """


def _build_state_occ_section(multistate_results: dict, state_occ_plot: str) -> str:
    """Build state occupation probabilities section."""
    state_occ = multistate_results.get("state_occupation_probabilities")
    if not state_occ:
        return ""

    return f"""
        <h3>State Occupation Probabilities</h3>
        {state_occ_plot}
        """
