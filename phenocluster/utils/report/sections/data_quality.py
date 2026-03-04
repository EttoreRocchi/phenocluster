"""Data quality section."""

from typing import Dict

from .._helpers import format_value, safe


def generate_data_quality_section(data: Dict) -> str:
    """Generate data quality section."""
    quality = data.get("data_quality", {})

    if not quality:
        return """
    <section id="quality">
        <h2>Data Quality</h2>
        <p>Data quality report not available.</p>
    </section>
"""

    summary = quality.get("summary", {})
    issues = summary.get("issues", [])

    issues_html = ""
    if issues:
        issues_html = "<ul>" + "".join([f"<li>{safe(issue)}</li>" for issue in issues]) + "</ul>"
    else:
        issues_html = (
            '<p><span class="badge badge-success">No critical issues identified</span></p>'
        )

    return f"""
    <section id="quality">
        <h2>Data Quality Assessment</h2>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="value">{
        format_value(
            summary.get("overall_missing_rate", 0) * 100
            if summary.get("overall_missing_rate")
            else quality.get("missing_data", {}).get("overall_rate", 0),
            1,
        )
    }%</div>
                <div class="label">Overall Missing Rate</div>
            </div>
            <div class="metric-card">
                <div class="value">{len(issues)}</div>
                <div class="label">Issues Identified</div>
            </div>
        </div>

        <h3>Issues</h3>
        {issues_html}
    </section>
"""
