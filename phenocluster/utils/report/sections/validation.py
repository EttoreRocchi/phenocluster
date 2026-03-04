"""Internal validation section."""

from typing import Dict

from .._helpers import empty_section, format_ci, format_value, safe


def generate_validation_section(data: Dict) -> str:
    """Generate internal validation section comparing train and test results."""
    validation = data.get("validation_report", {})
    outcome_results = data.get("outcome_results", {})
    model_fit = data.get("model_fit_metrics")

    if not validation and model_fit is None:
        return empty_section(
            "validation",
            "Internal Validation",
            "Validation metrics not available. Re-run the pipeline to generate.",
        )

    ll_section = _build_ll_section(validation, model_fit)
    proportion_section = _build_proportion_section(validation)
    or_section = _build_or_section(validation, outcome_results)

    return f"""
    <section id="validation">
        <h2>Internal Validation</h2>
        {ll_section}
        {proportion_section}
        {or_section}
    </section>
"""


def _build_ll_section(validation: dict, model_fit) -> str:
    """Build log-likelihood comparison section."""
    train_ll = validation.get("train_log_likelihood")
    test_ll = validation.get("test_log_likelihood")
    n_train = validation.get("n_train")
    n_test = validation.get("n_test")

    if train_ll is None and model_fit is not None and len(model_fit) > 0:
        row = model_fit.iloc[0]
        train_ll = row.get("train_log_likelihood")
        test_ll = row.get("test_log_likelihood")
        n_train = int(row.get("n_train", 0)) or None
        n_test = int(row.get("n_test", 0)) or None

    if train_ll is None or test_ll is None or not n_train or not n_test:
        return ""

    train_ll_per = train_ll
    test_ll_per = test_ll
    ll_diff = test_ll_per - train_ll_per

    return f"""
        <h3>Model Generalization</h3>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="value">{format_value(train_ll_per)}</div>
                <div class="label">Train LL/sample (n={n_train})</div>
            </div>
            <div class="metric-card">
                <div class="value">{format_value(test_ll_per)}</div>
                <div class="label">Test LL/sample (n={n_test})</div>
            </div>
            <div class="metric-card">
                <div class="value">{format_value(ll_diff)}</div>
                <div class="label">Test &minus; Train LL</div>
            </div>
        </div>
        """


def _build_proportion_section(validation: dict) -> str:
    """Build cluster proportion stability section."""
    train_pcts = validation.get("train_cluster_pcts", {})
    test_pcts = validation.get("test_cluster_pcts", {})
    train_sizes = validation.get("train_cluster_sizes", {})
    test_sizes = validation.get("test_cluster_sizes", {})

    if not train_pcts or not test_pcts:
        return ""

    rows = []
    for k in sorted(train_pcts.keys(), key=lambda x: int(x)):
        t_pct = train_pcts[k]
        s_pct = test_pcts.get(k, test_pcts.get(str(k), 0))
        diff = s_pct - t_pct
        t_n = train_sizes.get(k, train_sizes.get(str(k), ""))
        s_n = test_sizes.get(k, test_sizes.get(str(k), ""))
        diff_class = "val-positive" if abs(diff) < 5 else "val-warning"
        rows.append(
            f"<tr><td>Phenotype {safe(k)}</td>"
            f"<td>{safe(t_n)}</td><td>{t_pct:.1f}%</td>"
            f"<td>{safe(s_n)}</td><td>{s_pct:.1f}%</td>"
            f'<td class="{diff_class}">'
            f"{diff:+.1f}%</td></tr>"
        )

    return f"""
        <h3>Cluster Proportion Stability</h3>
        <table>
            <thead>
                <tr><th>Phenotype</th><th>Train N</th><th>Train %</th>
                <th>Test N</th><th>Test %</th><th>Difference</th></tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """


def _build_or_section(validation: dict, outcome_results: dict) -> str:
    """Build outcome association consistency section."""
    train_outcomes = outcome_results.get("train", {})
    test_outcomes = outcome_results.get("test", {})
    full_outcomes = outcome_results.get("full_cohort", {})

    if not train_outcomes or not test_outcomes or not full_outcomes:
        return ""

    or_rows = []
    for outcome_name in full_outcomes:
        if outcome_name.startswith("_"):
            continue
        train_data = train_outcomes.get(outcome_name, {})
        test_data = test_outcomes.get(outcome_name, {})
        full_data = full_outcomes.get(outcome_name, {})

        for cluster_id in sorted((k for k in full_data if not str(k).startswith("_")), key=str):
            fc = full_data.get(cluster_id, {})
            if not isinstance(fc, dict):
                continue
            fc_or = fc.get("OR")
            fc_ci_lo = fc.get("CI_lower", fc.get("ci_lower"))
            if fc_or == 1.0 and fc_ci_lo == 1.0:
                continue

            tr = train_data.get(cluster_id, train_data.get(str(cluster_id), {}))
            te = test_data.get(cluster_id, test_data.get(str(cluster_id), {}))
            if not isinstance(tr, dict):
                tr = {}
            if not isinstance(te, dict):
                te = {}

            def _or_cell(d: dict) -> str:
                or_v = d.get("OR")
                lo = d.get("CI_lower", d.get("ci_lower"))
                hi = d.get("CI_upper", d.get("ci_upper"))
                if or_v is None:
                    return "N/A"
                return f"{format_value(or_v)} {format_ci(lo, hi, bracket='()')}"

            tr_or = tr.get("OR")
            te_or = te.get("OR")
            if (
                isinstance(tr_or, (int, float))
                and isinstance(te_or, (int, float))
                and tr_or != 1.0
                and te_or != 1.0
            ):
                same_dir = (tr_or > 1 and te_or > 1) or (tr_or < 1 and te_or < 1)
                badge_txt = "Consistent" if same_dir else "Inconsistent"
                badge_class = "val-match" if same_dir else "val-mismatch"
                consistency = f'<span class="{badge_class}">{badge_txt}</span>'
            else:
                consistency = "N/A"

            display_name = outcome_name.replace("_", " ").title()
            or_rows.append(
                f"<tr><td>{safe(display_name)}</td>"
                f"<td>Phenotype {safe(cluster_id)}</td>"
                f"<td>{_or_cell(tr)}</td>"
                f"<td>{_or_cell(te)}</td>"
                f"<td>{_or_cell(fc)}</td>"
                f"<td>{consistency}</td></tr>"
            )

    if not or_rows:
        return ""

    return f"""
            <h3>Outcome Association Consistency</h3>
            <div style="overflow-x:auto">
            <table>
                <thead>
                    <tr><th>Outcome</th><th>Phenotype</th>
                    <th>Train OR (95% CI)</th><th>Test OR (95% CI)</th>
                    <th>Full OR (95% CI)</th><th>Consistency</th></tr>
                </thead>
                <tbody>
                    {"".join(or_rows)}
                </tbody>
            </table>
            </div>
            """
