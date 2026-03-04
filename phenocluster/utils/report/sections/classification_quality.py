"""Classification quality section."""

from typing import Dict

from .._helpers import empty_section, format_value, safe


def _render_classification_quality_block(cq: dict, cohort_label: str) -> str:
    """Render classification quality metrics block for a single cohort."""
    if not cq:
        return ""

    overall_avepp = cq.get("overall_avepp")
    overall_rel_entropy = cq.get("overall_relative_entropy")
    confidence = cq.get("assignment_confidence", {})
    per_phenotype = cq.get("per_phenotype", {})
    n_samples = cq.get("n_samples", "")
    n_label = f" (N={safe(n_samples)})" if n_samples else ""

    pheno_rows = []
    for k in sorted(per_phenotype.keys(), key=lambda x: int(x)):
        p = per_phenotype[k]
        avepp = p.get("avepp", 0)
        pheno_rows.append(
            f"<tr><td>Phenotype {safe(k)}</td>"
            f"<td>{p.get('n', 'N/A')}</td>"
            f"<td>{format_value(avepp)}</td>"
            f"<td>{format_value(p.get('min_pp'))}</td>"
            f"<td>{format_value(p.get('median_pp'))}</td>"
            f"<td>{format_value(p.get('mean_entropy'))}</td></tr>"
        )

    pheno_table = (
        f"""
        <table>
            <tr><th>Phenotype</th><th>N</th><th>AvePP</th>
            <th>Min PP</th><th>Median PP</th><th>Mean Entropy</th></tr>
            {"".join(pheno_rows)}
        </table>
    """
        if pheno_rows
        else ""
    )

    return f"""
        <h3>{safe(cohort_label)}{n_label}</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="value">{format_value(overall_avepp)}</div>
                <div class="label">Overall AvePP</div>
            </div>
            <div class="metric-card">
                <div class="value">{format_value(overall_rel_entropy)}</div>
                <div class="label">Mean Relative Entropy</div>
            </div>
            <div class="metric-card">
                <div class="value">{format_value(confidence.get("above_90"))}%</div>
                <div class="label">Assigned with &gt;90% confidence</div>
            </div>
            <div class="metric-card">
                <div class="value">{format_value(confidence.get("below_70"))}%</div>
                <div class="label">Assigned with &le;70% confidence</div>
            </div>
        </div>

        <h4>Per-Phenotype Classification Quality</h4>
        {pheno_table}

        <h4>Assignment Confidence Breakdown</h4>
        <table>
            <tr><th>Confidence Threshold</th><th>% of Patients</th></tr>
            <tr><td>&gt; 90%</td><td>{format_value(confidence.get("above_90"))}%</td></tr>
            <tr><td>&gt; 80%</td><td>{format_value(confidence.get("above_80"))}%</td></tr>
            <tr><td>&gt; 70%</td><td>{format_value(confidence.get("above_70"))}%</td></tr>
            <tr><td>&le; 70%</td><td>{format_value(confidence.get("below_70"))}%</td></tr>
        </table>
    """


def generate_classification_quality_section(data: Dict) -> str:
    """Generate classification quality section showing assignment confidence."""
    cq_full = data.get("classification_quality", {})
    cq_test = data.get("classification_quality_test", {})

    if not cq_full and not cq_test:
        return empty_section(
            "classification",
            "Classification Quality",
            "Classification quality metrics not available.",
        )

    full_block = _render_classification_quality_block(cq_full, "Full Cohort")
    test_block = _render_classification_quality_block(cq_test, "Test Set")

    return f"""
    <section id="classification">
        <h2>Classification Quality</h2>

        {full_block}
        {test_block}
    </section>
"""
