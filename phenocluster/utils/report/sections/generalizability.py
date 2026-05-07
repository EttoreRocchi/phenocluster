"""Generalizability section of the static HTML report (v0.3.0)."""

from pathlib import Path
from typing import Any, Dict, List

from .._helpers import embed_plots_matching


def generate_generalizability_section(data: Dict, results_dir: Path) -> str:
    """Render the temporal and multi-site generalizability blocks.

    Reads the JSON files written by ``_save_generalizability_outputs`` and
    embeds the corresponding plotly figures from the ``plots/`` directory.
    """
    temporal = data.get("temporal_validation_results") or []
    multisite = data.get("multisite_validation_results") or []
    external = data.get("external_cohorts_results") or []
    summary = data.get("generalizability_summary") or {}

    if not temporal and not multisite and not external:
        return ""

    blocks: List[str] = []
    if summary:
        blocks.append(_render_summary(summary))
    if temporal:
        blocks.append(_render_cohort_group("Temporal cohorts", temporal, results_dir, "temporal"))
    if multisite:
        blocks.append(
            _render_cohort_group("Multi-site cohorts", multisite, results_dir, "multisite")
        )
    if external:
        blocks.append(_render_cohort_group("External cohorts", external, results_dir, "external"))

    body = "\n".join(blocks)
    return f"""
    <section id="generalizability">
        <h2>Generalizability</h2>
        {body}
    </section>
    """


def _render_summary(summary: Dict[str, Any]) -> str:
    rows: List[str] = []
    for kind in ("temporal", "multisite"):
        block = summary.get(kind) or {}
        if not block:
            continue
        rows.append(
            f"<tr><td>{kind}</td>"
            f"<td>{block.get('n_cohorts', '')}</td>"
            f"<td>{_fmt(block.get('mean_ari'))}</td>"
            f"<td>{_fmt(block.get('mean_psi'))}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
    <h3>Summary</h3>
    <table class="results-table">
        <thead><tr><th>Kind</th><th># cohorts</th><th>Mean ARI</th><th>Mean PSI</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """


def _render_cohort_group(
    heading: str, cohorts: List[Dict[str, Any]], results_dir: Path, prefix: str
) -> str:
    cards: List[str] = []
    for cohort in cohorts:
        cards.append(_render_cohort_card(cohort, results_dir, prefix))
    return f"<h3>{heading}</h3>" + "\n".join(cards)


def _render_cohort_card(cohort: Dict[str, Any], results_dir: Path, prefix: str) -> str:
    label = cohort.get("label", "?")
    n_samples = cohort.get("n_samples", "?")
    log_likelihood = cohort.get("log_likelihood")
    refit = cohort.get("refit") or {}
    drift = cohort.get("drift") or []
    calibration = cohort.get("calibration") or {}

    metrics = []
    if log_likelihood is not None:
        metrics.append(f"<li>Log-likelihood: {_fmt(log_likelihood)}</li>")
    if "ari" in refit:
        metrics.append(f"<li>Refit ARI: {_fmt(refit.get('ari'))}</li>")
    if "nmi" in refit:
        metrics.append(f"<li>Refit NMI: {_fmt(refit.get('nmi'))}</li>")
    if "matched_accuracy" in refit:
        metrics.append(
            f"<li>Hungarian-matched accuracy: {_fmt(refit.get('matched_accuracy'))}</li>"
        )
    if "ece" in calibration:
        metrics.append(f"<li>ECE: {_fmt(calibration.get('ece'))}</li>")

    drift_table = _render_drift_table(drift)
    plot_html = embed_plots_matching(results_dir, _safe(label))

    warnings = cohort.get("warnings") or []
    warn_html = ""
    if warnings:
        warn_html = (
            "<details><summary>Warnings</summary><ul>"
            + "".join(f"<li>{w}</li>" for w in warnings)
            + "</ul></details>"
        )

    return f"""
    <details class="cohort-card" open>
        <summary><strong>{label}</strong> (n={n_samples})</summary>
        <ul>{"".join(metrics)}</ul>
        {drift_table}
        {warn_html}
        {plot_html}
    </details>
    """


def _render_drift_table(drift: List[Dict[str, Any]], top_k: int = 10) -> str:
    if not drift:
        return ""
    sorted_rows = sorted(
        (d for d in drift if d.get("psi") is not None),
        key=lambda r: abs(r.get("psi") or 0.0),
        reverse=True,
    )[:top_k]
    if not sorted_rows:
        return ""
    body_rows = []
    for r in sorted_rows:
        body_rows.append(
            f"<tr><td>{r.get('feature')}</td>"
            f"<td>{r.get('kind')}</td>"
            f"<td>{_fmt(r.get('psi'))}</td>"
            f"<td>{_fmt(r.get('ks_p'))}</td>"
            f"<td>{_fmt(r.get('chi2_p'))}</td></tr>"
        )
    header = "<tr><th>Feature</th><th>Kind</th><th>PSI</th><th>KS p</th><th>Chi2 p</th></tr>"
    return f"""
    <h4>Top drifted features</h4>
    <table class="results-table">
        <thead>{header}</thead>
        <tbody>{"".join(body_rows)}</tbody>
    </table>
    """


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _safe(label: str) -> str:
    return (
        str(label)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("=", "-")
        .replace(",", "_")
    )
