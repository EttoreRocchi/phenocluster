"""
PhenoCluster Dashboard - Streamlit app
======================================

Streamlit script entry point. Launched indirectly via the CLI:

    phenocluster dashboard <results_dir>

Tabs:
- Overview: run summary, project metadata, and headline metrics
- Phenotypes: cluster sizes, classification quality, per-phenotype detail
- Outcomes: per-phenotype OR forest plot with FDR filter
- Survival: Cox HR table and embedded Kaplan-Meier curves
- Multistate: transition hazard ratios and saved multistate plots
- Generalizability: temporal, multi-site, and external cohort comparison
- Drift explorer: per-cohort drift table with PSI threshold
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phenocluster.dashboard._imports import require_streamlit  # noqa: E402
from phenocluster.dashboard.loader import DashboardData, load_results  # noqa: E402

PALETTE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#999999",
]


def _resolve_results_dir() -> Path:
    args = [a for a in sys.argv[1:] if a and not a.startswith("--")]
    if args:
        return Path(args[0]).expanduser().resolve()
    return Path("results").resolve()


def _last_modified(results_dir: Path) -> Optional[datetime]:
    candidates = [
        results_dir / "results" / "cluster_statistics.json",
        results_dir / "analysis_report.html",
    ]
    for path in candidates:
        if path.exists():
            return datetime.fromtimestamp(path.stat().st_mtime)
    return None


def _plot_stems(data: DashboardData) -> set:
    """Return the set of plot names available as JSON or HTML."""
    return {p.stem for p in data.plot_json_files} | {p.stem for p in data.plot_files}


def _figure_from_html(html_content: str):
    """Parse a Plotly figure out of a saved write_html() document.

    Plotly's HTML export embeds the figure as a single
    ``Plotly.newPlot("UUID", data, layout, config)`` call near the end
    of the document. We locate the call, then use ``json.JSONDecoder``
    in raw-decode mode to walk the JSON arguments (data array, layout
    object, optional config object) regardless of nesting depth.
    Returns a ``go.Figure`` or ``None`` on parse failure.
    """
    import json
    import re

    import plotly.graph_objects as go

    matches = list(
        re.finditer(
            r'Plotly\.newPlot\(\s*"[0-9a-f-]{8,}"\s*,\s*',
            html_content,
        )
    )
    if not matches:
        return None
    pos = matches[-1].end()
    decoder = json.JSONDecoder()
    try:
        data, idx = decoder.raw_decode(html_content[pos:])
        pos += idx
        while pos < len(html_content) and html_content[pos] in ", \n\t":
            pos += 1
        layout, _ = decoder.raw_decode(html_content[pos:])
    except (json.JSONDecodeError, ValueError):
        return None
    return go.Figure(data=data, layout=layout)


def _embed_plot(
    st,
    data: DashboardData,
    plot_name: str,
    height: int = 520,
    square: bool = False,
) -> None:
    """Render a saved Plotly figure inside the dashboard.

    Three resolution paths, in order:

    1. ``plot_json_files``: read the JSON sidecar saved by the pipeline
       (v0.3.0+) via ``plotly.io.from_json``. Fastest, smallest payload.
    2. ``plot_files``: extract the figure JSON from the saved HTML
       export with :func:`_figure_from_html`. Lets older runs work
       without re-running the pipeline.
    3. Neither found: print a clear "not found" info banner.

    All paths render via ``st.plotly_chart`` - no iframe, no
    ``st.components.v1.html``, no deprecation warnings, no data-URL
    size limits.

    If the figure already declares its own ``layout.height`` (e.g. the
    Kaplan-Meier curves expand their height to make room for the
    number-at-risk table) we honour it instead of forcing the caller's
    default. Otherwise, ``height`` is the fallback used when the figure
    leaves height unset.
    """
    fig = None
    json_path = next((p for p in data.plot_json_files if p.stem == plot_name), None)
    if json_path is not None:
        try:
            import plotly.io as pio

            fig = pio.from_json(json_path.read_text())
        except (ValueError, OSError) as exc:
            st.warning(f"Could not parse `{plot_name}.json` ({type(exc).__name__}): {exc}")

    if fig is None:
        html_path = next((p for p in data.plot_files if p.stem == plot_name), None)
        if html_path is not None:
            fig = _figure_from_html(html_path.read_text())
            if fig is None:
                st.warning(
                    f"Could not extract figure from `{plot_name}.html`. "
                    "Re-run the pipeline to regenerate the JSON sidecar."
                )

    if fig is None:
        st.info(f"Plot `{plot_name}` not found.")
        return

    fig_height = None
    try:
        fig_height = fig.layout.height
    except AttributeError:
        fig_height = None
    render_height = int(fig_height) if fig_height else height
    try:
        title_text = None
        try:
            title_text = fig.layout.title.text
        except AttributeError:
            title_text = None
        if title_text in (None, "", "undefined", "None"):
            fig.update_layout(title=dict(text=""))

        layout_updates: Dict[str, Any] = dict(
            width=None,
            autosize=True,
            modebar=dict(orientation="v"),
        )
        if square:
            layout_updates["yaxis"] = dict(scaleanchor="x", scaleratio=1, constrain="domain")
            layout_updates["xaxis"] = dict(constrain="domain")
        fig.update_layout(**layout_updates)
    except (AttributeError, ValueError):
        pass
    if square:
        outer = st.columns([1, 3, 1])
        with outer[1]:
            st.plotly_chart(fig, width="stretch", height=render_height)
    else:
        st.plotly_chart(fig, width="stretch", height=render_height)


def _phenotype_sizes(cluster_stats: Optional[Dict[str, Any]]) -> Dict[int, int]:
    """Extract ``{phenotype_id: n_samples}`` from a cluster_statistics.json dict.

    The JSON saved by ``ClusterStatistics`` is keyed by phenotype id (as a
    string) with each entry carrying its own ``n_samples`` count plus
    feature summaries. There is no top-level ``cluster_sizes`` field.
    """
    if not cluster_stats:
        return {}
    out: Dict[int, int] = {}
    for k, v in cluster_stats.items():
        if not isinstance(v, dict):
            continue
        if "n_samples" not in v:
            continue
        try:
            out[int(k)] = int(v["n_samples"])
        except (TypeError, ValueError):
            continue
    return out


def _format_num(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v) or math.isinf(v):
        return "n/a"
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
        return f"{v:.{decimals}e}"
    return f"{v:.{decimals}f}"


def _inject_style(st) -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {min-width: 280px;}
        .stMetric {background: rgba(0,114,178,0.04); border-radius: 8px; padding: 8px 12px;}
        div[data-testid="stMetricValue"] {font-size: 1.5rem;}
        h2 {margin-top: 1.4rem;}
        h3 {margin-top: 1.2rem; color: #0072B2;}
        .pheno-pill {
            display: inline-block; padding: 2px 10px; margin: 2px;
            border-radius: 999px; background: #0072B2; color: white;
            font-weight: 600; font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_SETTINGS_DEFAULTS = {
    "settings_decimals": 3,
    "settings_height": 520,
    "settings_fdr": 0.05,
    "settings_psi": 0.10,
    "settings_warnings": True,
    "settings_phenotype_highlight": None,  # None = all phenotypes
}


def _reset_settings() -> None:
    """Restore every sidebar widget to its default value.

    Removing the key from ``session_state`` causes Streamlit to fall
    back to the widget's declared default on the next render, so we do
    not need to re-set the values here.
    """
    import streamlit as st

    for key in _SETTINGS_DEFAULTS:
        st.session_state.pop(key, None)


def _settings_sidebar(st, data: DashboardData) -> Dict[str, Any]:
    st.sidebar.title("PhenoCluster")
    st.sidebar.caption(f"Results: {data.results_dir}")
    last = _last_modified(data.results_dir)
    if last:
        st.sidebar.caption(f"Updated: {last:%Y-%m-%d %H:%M}")

    st.sidebar.divider()
    st.sidebar.subheader("Display settings")
    decimals = st.sidebar.slider(
        "Numeric precision",
        2,
        6,
        _SETTINGS_DEFAULTS["settings_decimals"],
        1,
        key="settings_decimals",
    )
    plot_height = st.sidebar.slider(
        "Plot height (px)",
        300,
        1000,
        _SETTINGS_DEFAULTS["settings_height"],
        20,
        key="settings_height",
    )
    fdr = st.sidebar.slider(
        "FDR threshold",
        0.0,
        1.0,
        _SETTINGS_DEFAULTS["settings_fdr"],
        0.01,
        key="settings_fdr",
        help="Applied across the Outcomes tab and any cross-cohort concordance views.",
    )
    psi = st.sidebar.slider(
        "PSI threshold",
        0.0,
        1.0,
        _SETTINGS_DEFAULTS["settings_psi"],
        0.01,
        key="settings_psi",
        help="Applied to drift bar charts and feature filters across all tabs.",
    )
    show_warnings = st.sidebar.toggle(
        "Show cohort warnings",
        value=_SETTINGS_DEFAULTS["settings_warnings"],
        key="settings_warnings",
    )

    available_phenos = _phenotype_sizes(data.cluster_stats)
    pheno_options = sorted(available_phenos.keys()) if available_phenos else []
    selected_phenos: List[int] = []
    if pheno_options:
        selected_phenos = st.sidebar.multiselect(
            "Highlight phenotypes",
            pheno_options,
            default=pheno_options,
            key="settings_phenotype_highlight",
            help="Filter tables and plots across all tabs to only the selected phenotypes.",
        )

    st.sidebar.button(
        "Reset settings",
        on_click=_reset_settings,
        key="settings_reset",
        use_container_width=True,
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"[User guide]({DOCS_URL}) &nbsp;&middot;&nbsp; "
        f"[Repository](https://github.com/EttoreRocchi/phenocluster)"
    )
    return {
        "decimals": decimals,
        "plot_height": plot_height,
        "fdr": fdr,
        "psi": psi,
        "show_warnings": show_warnings,
        "phenotype_highlight": selected_phenos or pheno_options,
        "all_phenotypes": pheno_options,
    }


DOCS_URL = "https://ettorerocchi.github.io/phenocluster"

_GLOSSARY_GROUPS: List[Tuple[str, List[Tuple[str, str]]]] = [
    (
        "Phenotype discovery",
        [
            (
                "Phenotype",
                "A latent patient subgroup discovered by the LCA/LPA model. "
                "Phenotypes are numbered by size (largest = Phenotype 0).",
            ),
            (
                "AvePP",
                "Average Posterior Probability. The mean of the assigned-class "
                "posterior across patients in a phenotype. Higher values mean "
                "tighter, more confident assignments.",
            ),
            (
                "Relative entropy",
                "Mean per-sample posterior entropy normalised by log K. "
                "Lower means assignments are more decisive.",
            ),
            (
                "BIC / AIC / ICL",
                "Information criteria used to pick the cluster count. Lower is "
                "better; ICL also penalises overlapping clusters.",
            ),
        ],
    ),
    (
        "Inference",
        [
            (
                "OR (odds ratio)",
                "Phenotype-vs-reference odds ratio for a binary outcome. "
                "OR=1 = no association. CIs are 95% Wald.",
            ),
            (
                "HR (hazard ratio)",
                "Cox proportional-hazards ratio for time-to-event outcomes. "
                "HR=1 = same hazard as reference; HR<1 protective.",
            ),
            (
                "FDR / q-value",
                "Benjamini-Hochberg-adjusted p-value controlling the expected "
                "false-discovery rate at the chosen threshold.",
            ),
            (
                "Schoenfeld / PH diagnostic",
                "Test of the proportional-hazards assumption. Violation means "
                "the HR is not constant over time.",
            ),
        ],
    ),
    (
        "Generalizability",
        [
            (
                "training_scope",
                "`per_split` (default) refits a fresh model on each "
                "derivation cohort; `global` reuses the pipeline's full-cohort "
                "model. External CSVs always use the global model.",
            ),
            (
                "fit_mode",
                "Which model produced a cohort's metrics: `per_split` for "
                "in-CSV splits under the default scope, `global` for external "
                "CSVs and the legacy permissive path.",
            ),
            (
                "feature_selector_mode",
                "Whether the feature selector was refit on this split "
                "(`per_split_refit`), reused (`global_reused`), or reused with "
                "a warning when the LASSO target column collides with an "
                "outcome (`global_reused_with_warning`).",
            ),
            (
                "derivation_only_ari",
                "Sanity check: ARI between the per-split fresh derivation fit "
                "and the global model's labels on the same rows.",
            ),
            (
                "ARI / NMI (refit)",
                "Adjusted Rand Index and Normalized Mutual Information "
                "between the derivation model's labels on the validation rows "
                "and a fresh refit on those same rows.",
            ),
            (
                "PSI",
                "Population Stability Index. PSI < 0.10 = no meaningful drift; "
                "0.10-0.25 = moderate; >= 0.25 = substantial.",
            ),
            (
                "ECE / Brier",
                "Expected Calibration Error and Brier score for posterior "
                "probabilities. Computed only in refit-and-match mode.",
            ),
            (
                "Lin's CCC",
                "Concordance correlation coefficient comparing derivation OR "
                "vectors to validation OR vectors. CCC <= Pearson r.",
            ),
        ],
    ),
]


def _render_glossary(st) -> None:
    st.subheader("Glossary and quick reference")
    st.markdown(
        f"For full documentation see [the user guide]({DOCS_URL}). The terms "
        "below appear throughout the tabs and saved JSON files."
    )
    cols = st.columns(min(3, len(_GLOSSARY_GROUPS)))
    for i, (group, items) in enumerate(_GLOSSARY_GROUPS):
        col = cols[i % len(cols)]
        with col:
            with st.expander(group, expanded=False):
                for term, definition in items:
                    st.markdown(f"**{term}** &nbsp; {definition}")


def _project_header(st, data: DashboardData) -> None:
    project = "PhenoCluster"
    n_clusters: Optional[int] = None
    n_samples: Optional[int] = None
    seed: Optional[int] = None
    training_scope: Optional[str] = None
    if data.config:
        project = (data.config.get("global") or {}).get("project_name", project)
        seed = (data.config.get("global") or {}).get("random_state")
        gen = data.config.get("generalizability") or {}
        training_scope = gen.get("training_scope")
    if data.cluster_stats:
        sizes = _phenotype_sizes(data.cluster_stats)
        if sizes:
            n_clusters = len(sizes)
            n_samples = int(sum(sizes.values()))

    st.title(f"{project}")
    st.caption("Pipeline dashboard")

    cols = st.columns(4)
    cols[0].metric("Phenotypes", n_clusters if n_clusters else "n/a")
    cols[1].metric("Samples", f"{n_samples:,}" if n_samples else "n/a")
    cols[2].metric("Random seed", seed if seed is not None else "n/a")
    cols[3].metric("training_scope", training_scope or "n/a")
    st.divider()


def _render_overview_config_block(st, config: Dict[str, Any]) -> None:
    data_cfg = config.get("data") or {}
    outcome = config.get("outcome") or {}
    survival = config.get("survival") or {}
    multistate = config.get("multistate") or {}
    gen = config.get("generalizability") or {}
    n_cont = len(data_cfg.get("continuous_columns", []))
    n_cat = len(data_cfg.get("categorical_columns", []))

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Pipeline configuration**")
        st.markdown(
            f"- **Continuous features:** {n_cont}\n"
            f"- **Categorical features:** {n_cat}\n"
            f"- **Outcome analysis:** {'enabled' if outcome.get('enabled') else 'disabled'}\n"
            f"- **Survival analysis:** {'enabled' if survival.get('enabled') else 'disabled'}\n"
            f"- **Multistate modelling:** "
            f"{'enabled' if multistate.get('enabled') else 'disabled'}\n"
        )
    with cols[1]:
        st.markdown("**Generalizability**")
        if gen.get("enabled"):
            st.markdown(
                f"- **training_scope:** `{gen.get('training_scope', 'per_split')}`\n"
                f"- **feature_selector_scope:** "
                f"`{gen.get('feature_selector_scope', 'auto')}`\n"
                f"- **temporal:** {'configured' if gen.get('temporal') else 'off'}\n"
                f"- **multisite:** {'configured' if gen.get('multisite') else 'off'}\n"
                f"- **external_cohorts:** {len(gen.get('external_cohorts') or [])}\n"
            )
        else:
            st.info("Generalizability assessment not enabled in this run.")


def _has_plot(data: DashboardData, stem: str) -> bool:
    return any(p.stem == stem for p in data.plot_files) or any(
        p.stem == stem for p in data.plot_json_files
    )


def _render_overview_tab(st, data: DashboardData) -> None:
    st.subheader("Run overview")
    if not data.config:
        st.info("Pipeline configuration not found in the results directory.")
    else:
        _render_overview_config_block(st, data.config)

    if _has_plot(data, "model_selection"):
        st.subheader("Model selection")
        _embed_plot(st, data, "model_selection", height=520)

    _render_glossary(st)


def _phenotype_size_chart(sizes: Dict[Any, Any], height: int):
    import plotly.graph_objects as go

    keys = sorted(sizes.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
    counts = [int(sizes[k]) for k in keys]
    fig = go.Figure(
        data=go.Bar(
            x=[f"Phenotype {k}" for k in keys],
            y=counts,
            marker_color=[PALETTE[i % len(PALETTE)] for i, _ in enumerate(keys)],
            text=counts,
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>n=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(text="Phenotype size distribution", x=0.5, xanchor="center", y=0.97),
        yaxis=dict(title="Patients", automargin=True),
        xaxis=dict(title=None, automargin=True, tickangle=0),
        margin=dict(l=60, r=20, t=80, b=60),
        font=dict(family="Arial, sans-serif", size=13),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return fig


def _render_phenotype_sizes_section(
    st, sizes: Dict[int, int], highlight: set, plot_height: int
) -> None:
    if not sizes:
        return
    sizes_view = {k: v for k, v in sizes.items() if k in highlight} or sizes
    fig = _phenotype_size_chart(sizes_view, plot_height)
    st.plotly_chart(fig, width="stretch")


def _per_phenotype_quality_rows(
    per: Dict[Any, Any], decimals: int, highlight: set
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k, v in per.items():
        if not isinstance(v, dict):
            continue
        rows.append(
            {
                "Phenotype": int(k) if str(k).isdigit() else k,
                "n": v.get("n"),
                "AvePP": _format_num(v.get("avepp"), decimals),
                "Median PP": _format_num(v.get("median_pp"), decimals),
                "Min PP": _format_num(v.get("min_pp"), decimals),
                "Mean entropy": _format_num(v.get("mean_entropy"), decimals),
            }
        )
    if highlight:
        rows = [r for r in rows if r["Phenotype"] in highlight]
    return rows


def _render_classification_quality_section(
    st, cq: Dict[str, Any], settings: Dict[str, Any], highlight: set
) -> None:
    st.markdown("### Classification quality")
    cols = st.columns(4)
    cols[0].metric("Overall AvePP", _format_num(cq.get("overall_avepp"), settings["decimals"]))
    cols[1].metric(
        "Overall relative entropy",
        _format_num(cq.get("overall_relative_entropy"), settings["decimals"]),
    )
    ac = cq.get("assignment_confidence") or {}
    cols[2].metric("Conf. > 90%", f"{ac.get('above_90', 0):.1f}%" if ac else "n/a")
    cols[3].metric("Conf. > 80%", f"{ac.get('above_80', 0):.1f}%" if ac else "n/a")

    rows = _per_phenotype_quality_rows(
        cq.get("per_phenotype") or {},
        settings["decimals"],
        highlight if settings.get("phenotype_highlight") else set(),
    )
    if rows:
        with st.expander("Per-phenotype quality", expanded=False):
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_feature_profiles_section(st, data: DashboardData, plot_height: int) -> None:
    plot_stems = _plot_stems(data)
    has_heat_cont = "heatmap_continuous" in plot_stems
    has_heat_cat = "categorical_heatmap" in plot_stems
    has_consensus = "consensus_matrix" in plot_stems
    if has_heat_cont or has_heat_cat:
        st.markdown("### Feature profiles")
        if has_heat_cont:
            _embed_plot(st, data, "heatmap_continuous", height=plot_height)
        if has_heat_cat:
            _embed_plot(st, data, "categorical_heatmap", height=plot_height)
    if has_consensus:
        st.markdown("### Consensus matrix (stability)")
        _embed_plot(st, data, "consensus_matrix", height=plot_height, square=True)


def _render_phenotypes_tab(st, data: DashboardData, settings: Dict[str, Any]) -> None:
    st.subheader("Phenotype distribution and quality")
    if not data.cluster_stats:
        st.info("No cluster statistics found in `results/cluster_statistics.json`.")
        return

    sizes = _phenotype_sizes(data.cluster_stats)
    highlight = set(settings.get("phenotype_highlight") or sizes.keys())
    _render_phenotype_sizes_section(st, sizes, highlight, settings["plot_height"])

    if data.classification_quality:
        _render_classification_quality_section(st, data.classification_quality, settings, highlight)

    _render_feature_profiles_section(st, data, settings["plot_height"])


def _outcome_forest_plot(rows: List[Dict[str, Any]], height: int, decimals: int):
    import plotly.graph_objects as go

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["OR"]).copy()
    if df.empty:
        return None
    df["log_or"] = np.log(df["OR"].astype(float))
    df["log_lo"] = np.log(df["CI_lower"].astype(float))
    df["log_hi"] = np.log(df["CI_upper"].astype(float))
    df = df.sort_values(["outcome", "phenotype"])
    df["row_label"] = df.apply(lambda r: f"{r['outcome']}  -  P{r['phenotype']}", axis=1)

    significant = df["p_value_fdr"].fillna(1.0) <= 0.05
    colors = np.where(significant, "#0072B2", "#999999")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["log_or"],
            y=df["row_label"],
            error_x=dict(
                type="data",
                symmetric=False,
                array=df["log_hi"] - df["log_or"],
                arrayminus=df["log_or"] - df["log_lo"],
            ),
            mode="markers",
            marker=dict(size=10, color=colors, line=dict(color="black", width=0.5)),
            customdata=np.stack([df["OR"], df["p_value_fdr"].fillna(np.nan)], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>OR=%{customdata[0]:."
                + str(decimals)
                + "f}<br>q=%{customdata[1]:.3g}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=max(height, 80 + 26 * len(df)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(text="Per-phenotype log(OR) forest", x=0.5, xanchor="center", y=0.97),
        xaxis=dict(title="log(OR), 95% CI", automargin=True, zeroline=True),
        yaxis=dict(title=None, automargin=True),
        margin=dict(l=180, r=40, t=80, b=60),
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def _outcome_rows(full_cohort: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for outcome, per_pheno in full_cohort.items():
        for cluster_id, entry in per_pheno.items():
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "phenotype": cluster_id,
                    "OR": entry.get("OR"),
                    "CI_lower": entry.get("CI_lower"),
                    "CI_upper": entry.get("CI_upper"),
                    "p_value": entry.get("p_value"),
                    "p_value_fdr": entry.get("p_value_fdr"),
                    "test_method": entry.get("test_method"),
                }
            )
    return rows


def _filter_outcome_phenotypes(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    highlight = settings.get("phenotype_highlight")
    all_phenos = settings.get("all_phenotypes", [])
    if highlight and len(highlight) != len(all_phenos):
        return df[df["phenotype"].isin(highlight)]
    return df


def _render_outcomes_table(st, df_view: pd.DataFrame, decimals: int) -> None:
    with st.expander("Tabular results", expanded=False):
        display = df_view.copy()
        for col in ["OR", "CI_lower", "CI_upper", "p_value", "p_value_fdr"]:
            if col in display:
                display[col] = display[col].map(
                    lambda v: _format_num(v, decimals) if pd.notna(v) else "n/a"
                )
        st.dataframe(display, width="stretch", hide_index=True)


def _render_outcomes_tab(st, data: DashboardData, settings: Dict[str, Any]) -> None:
    st.subheader("Outcome associations")
    if not data.outcome_results:
        st.info("Outcome analysis was not run for this pipeline.")
        return
    full_cohort = data.outcome_results.get("full_cohort", {})
    if not full_cohort:
        st.info("Outcome results have no `full_cohort` block.")
        return

    rows = _outcome_rows(full_cohort)
    if not rows:
        st.info("No per-phenotype outcome results to display.")
        return

    fdr = float(settings["fdr"])
    df = _filter_outcome_phenotypes(pd.DataFrame(rows), settings)
    cols = st.columns(3)
    cols[0].metric("Phenotype-outcome pairs", f"{len(df)}")
    cols[1].metric(f"Below FDR={fdr:.2f}", f"{int((df['p_value_fdr'].fillna(1.0) <= fdr).sum())}")
    cols[2].metric("Outcomes analyzed", df["outcome"].nunique())

    df_view = df if fdr >= 1.0 else df[df["p_value_fdr"].fillna(1.0) <= fdr]
    if df_view.empty:
        st.info("No phenotype-outcome pairs pass the threshold.")
        return

    fig = _outcome_forest_plot(
        df_view.to_dict("records"), settings["plot_height"], settings["decimals"]
    )
    if fig is not None:
        st.plotly_chart(fig, width="stretch")

    _render_outcomes_table(st, df_view, settings["decimals"])


def _render_survival_tab(st, data: DashboardData, settings: Dict[str, Any]) -> None:
    if not data.survival_results:
        st.info("No survival results were saved.")
        return
    targets = [t for t in data.survival_results.keys() if not t.endswith("_weighted")]
    if not targets:
        st.info("No survival targets found.")
        return
    target = st.selectbox("Survival target", targets, key="surv_target")

    stems = _plot_stems(data)
    km_name = f"kaplan_meier_{target}" if f"kaplan_meier_{target}" in stems else None
    na_name = f"nelson_aalen_{target}" if f"nelson_aalen_{target}" in stems else None
    cols = st.columns(2)
    with cols[0]:
        if km_name is not None:
            _embed_plot(st, data, km_name, height=settings["plot_height"])
    with cols[1]:
        if na_name is not None:
            _embed_plot(st, data, na_name, height=settings["plot_height"])


def _cohort_summary_rows(
    cohorts: List[Dict[str, Any]], kind: str, decimals: int
) -> List[Dict[str, Any]]:
    rows = []
    for c in cohorts:
        refit = c.get("refit") or {}
        drift = c.get("drift") or []
        psi_vals = [r.get("psi") for r in drift if isinstance(r, dict) and r.get("psi") is not None]
        rows.append(
            {
                "Kind": kind,
                "Label": c.get("label"),
                "n": c.get("n_samples"),
                "fit_mode": c.get("fit_mode") or "n/a",
                "feature_selector_mode": c.get("feature_selector_mode") or "n/a",
                "logL": _format_num(c.get("log_likelihood"), decimals),
                "ARI (refit)": _format_num(refit.get("ari"), decimals),
                "NMI (refit)": _format_num(refit.get("nmi"), decimals),
                "Mean PSI": _format_num(
                    sum(psi_vals) / len(psi_vals) if psi_vals else None, decimals
                ),
                "Max PSI": _format_num(max(psi_vals) if psi_vals else None, decimals),
                "Warnings": len(c.get("warnings") or []),
            }
        )
    return rows


def _drift_bar_plot(drift: List[Dict[str, Any]], top_k: int, threshold: float, height: int):
    import plotly.graph_objects as go

    if not drift:
        return None
    df = pd.DataFrame(drift)
    if "psi" not in df.columns:
        return None
    df = df.dropna(subset=["psi"]).copy()
    df["abs_psi"] = df["psi"].abs()
    if threshold > 0:
        df = df[df["abs_psi"] >= threshold]
    if df.empty:
        return None
    df = df.sort_values("abs_psi", ascending=False).head(top_k).iloc[::-1]
    colors = np.where(
        df["abs_psi"] >= 0.25,
        "#D55E00",
        np.where(df["abs_psi"] >= 0.10, "#E69F00", "#0072B2"),
    )
    fig = go.Figure(
        data=go.Bar(
            x=df["psi"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>PSI=%{x:.3f}<extra></extra>",
        )
    )
    for ref, label, color in [
        (0.10, "moderate (0.10)", "#E69F00"),
        (0.25, "substantial (0.25)", "#D55E00"),
        (-0.10, None, "#E69F00"),
        (-0.25, None, "#D55E00"),
    ]:
        fig.add_vline(
            x=ref,
            line_dash="dot",
            line_color=color,
            opacity=0.5,
            annotation_text=label if label else "",
            annotation_position="top",
        )
    fig.update_layout(
        height=max(height, 100 + 26 * len(df)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(text="Top features by PSI", x=0.5, xanchor="center", y=0.97),
        xaxis=dict(title="Population Stability Index (signed)", automargin=True),
        yaxis=dict(title=None, automargin=True),
        margin=dict(l=180, r=40, t=80, b=60),
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def _render_cohort_header(st, cohort: Dict[str, Any], settings: Dict[str, Any]) -> None:
    head_cols = st.columns(4)
    head_cols[0].metric("n", cohort.get("n_samples"))
    head_cols[1].metric("fit_mode", cohort.get("fit_mode") or "n/a")
    head_cols[2].metric("logL", _format_num(cohort.get("log_likelihood"), settings["decimals"]))
    head_cols[3].metric(
        "deriv-only ARI",
        _format_num(cohort.get("derivation_only_ari"), settings["decimals"]),
    )
    fs_mode = cohort.get("feature_selector_mode")
    if fs_mode:
        st.caption(f"feature_selector_mode: `{fs_mode}`")


def _render_phenotype_prevalence_section(
    st, cluster_dist: Dict[Any, Any], deriv_dist: Dict[Any, Any], plot_height: int
) -> None:
    if not cluster_dist:
        return
    import plotly.graph_objects as go

    keys = sorted(cluster_dist.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Validation",
            x=[f"P{k}" for k in keys],
            y=[float((cluster_dist.get(k) or {}).get("percentage", 0)) for k in keys],
            marker_color="#0072B2",
        )
    )
    if deriv_dist:
        fig.add_trace(
            go.Bar(
                name="Derivation",
                x=[f"P{k}" for k in keys],
                y=[float((deriv_dist.get(k) or {}).get("percentage", 0)) for k in keys],
                marker_color="#999999",
            )
        )
    fig.update_layout(
        barmode="group",
        height=plot_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(
            text="Phenotype prevalence (% of cohort)",
            x=0.5,
            xanchor="center",
            y=0.97,
        ),
        yaxis=dict(title="%", automargin=True),
        xaxis=dict(title=None, automargin=True),
        margin=dict(l=60, r=20, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")


def _render_refit_section(st, refit: Dict[str, Any], decimals: int) -> None:
    if not refit:
        return
    st.markdown("### Refit-and-match")
    cols = st.columns(3)
    cols[0].metric("ARI", _format_num(refit.get("ari"), decimals))
    cols[1].metric("NMI", _format_num(refit.get("nmi"), decimals))
    cols[2].metric("Matched accuracy", _format_num(refit.get("matched_accuracy"), decimals))
    with st.expander("Cluster mapping and unmatched", expanded=False):
        st.json(
            {
                "mapping": refit.get("mapping"),
                "unmatched_derivation_clusters": refit.get("unmatched_derivation_clusters"),
                "unmatched_validation_clusters": refit.get("unmatched_validation_clusters"),
            }
        )


def _render_drift_section(
    st, drift: List[Dict[str, Any]], cohort_label: Any, settings: Dict[str, Any]
) -> None:
    if not drift:
        return
    st.markdown("### Drift")
    threshold = float(settings["psi"])
    top_k = st.slider("Top features", 5, 50, 20, 1, key=f"detail_topk_{cohort_label}")
    fig = _drift_bar_plot(drift, top_k, threshold, settings["plot_height"])
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No features pass the PSI threshold.")
    with st.expander("Full drift table", expanded=False):
        st.dataframe(pd.DataFrame(drift), width="stretch", hide_index=True)


def _render_warnings_section(st, warnings_list: List[str]) -> None:
    if not warnings_list:
        return
    with st.expander(f"Warnings ({len(warnings_list)})", expanded=True):
        for w in warnings_list:
            st.warning(w)


def _render_cohort_detail(st, cohort: Dict[str, Any], settings: Dict[str, Any]) -> None:
    _render_cohort_header(st, cohort, settings)
    _render_phenotype_prevalence_section(
        st,
        cohort.get("cluster_distribution") or {},
        cohort.get("derivation_distribution") or {},
        settings["plot_height"],
    )
    _render_refit_section(st, cohort.get("refit") or {}, settings["decimals"])
    _render_drift_section(st, cohort.get("drift") or [], cohort.get("label"), settings)
    if settings["show_warnings"]:
        _render_warnings_section(st, cohort.get("warnings") or [])


def _render_multistate_tab(st, data: DashboardData, settings: Dict[str, Any]) -> None:
    stems = _plot_stems(data)
    if not data.multistate_results and not any(s.startswith("multistate_") for s in stems):
        st.info("No multistate analysis was run for this pipeline.")
        return

    multistate_plots = [
        ("Pathway frequency", "multistate_pathways"),
        ("Transition hazards", "multistate_transition_hazards"),
        ("State occupation uncertainty", "multistate_state_occupation_uncertainty"),
        ("State diagram", "multistate_state_diagram"),
    ]
    rendered = False
    for label, stem in multistate_plots:
        if stem not in stems:
            continue
        st.markdown(f"### {label}")
        _embed_plot(st, data, stem, height=settings["plot_height"])
        rendered = True

    if not rendered:
        st.info("Multistate analysis ran but no plots were saved.")
        return

    if data.multistate_results:
        with st.expander("Raw multistate results", expanded=False):
            transition_results = data.multistate_results.get("transition_results", {})
            if transition_results:
                rows = []
                for transition_name, entry in transition_results.items():
                    if not isinstance(entry, dict):
                        continue
                    pheno_effects = entry.get("phenotype_effects") or {}
                    for pheno_id, eff in pheno_effects.items():
                        if not isinstance(eff, dict):
                            continue
                        rows.append(
                            {
                                "Transition": transition_name,
                                "Phenotype": pheno_id,
                                "HR": _format_num(eff.get("HR"), settings["decimals"]),
                                "CI_lower": _format_num(eff.get("CI_lower"), settings["decimals"]),
                                "CI_upper": _format_num(eff.get("CI_upper"), settings["decimals"]),
                                "p": _format_num(eff.get("p_value"), settings["decimals"]),
                                "q": _format_num(eff.get("q_value"), settings["decimals"]),
                                "n_events": entry.get("n_events"),
                                "n_at_risk": entry.get("n_at_risk"),
                            }
                        )
                if rows:
                    st.markdown("**Transition hazard ratios (per phenotype)**")
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_summary_block(st, summary: Dict[str, Any], decimals: int) -> None:
    if not summary:
        return
    scope = summary.get("training_scope")
    if scope:
        tag = "per-split refit" if scope == "per_split" else "global model"
        st.caption(f"training_scope: `{scope}` ({tag})")
    cols = st.columns(3)
    for i, kind in enumerate(("temporal", "multisite", "external")):
        block = summary.get(kind) or {}
        with cols[i]:
            st.markdown(f"**{kind.title()}**")
            st.metric("Cohorts", block.get("n_cohorts", 0))
            if block.get("mean_ari") is not None:
                st.metric("Mean ARI", _format_num(block.get("mean_ari"), decimals))
            if block.get("mean_psi") is not None:
                st.metric("Mean PSI", _format_num(block.get("mean_psi"), decimals))


def _gen_cohort_summary_rows(data: DashboardData, decimals: int) -> List[Dict[str, Any]]:
    return (
        _cohort_summary_rows(data.temporal_validation_results, "temporal", decimals)
        + _cohort_summary_rows(data.multisite_validation_results, "site", decimals)
        + _cohort_summary_rows(data.external_cohorts_results, "external", decimals)
    )


def _render_cohort_inspector(
    st, data: DashboardData, rows: List[Dict[str, Any]], settings: Dict[str, Any]
) -> None:
    cohort_options: Dict[str, Tuple[str, str]] = {
        f"{r['Kind']}: {r['Label']}": (r["Kind"], r["Label"]) for r in rows if r["Label"]
    }
    if not cohort_options:
        return
    st.markdown("### Cohort detail")
    choice = st.selectbox("Inspect cohort", list(cohort_options.keys()), key="gen_cohort_choice")
    kind, label = cohort_options[choice]
    if kind == "temporal":
        cohorts = data.temporal_validation_results
    elif kind == "site":
        cohorts = data.multisite_validation_results
    else:
        cohorts = data.external_cohorts_results
    cohort = next((c for c in cohorts if c.get("label") == label), None)
    if cohort is not None:
        _render_cohort_detail(st, cohort, settings)


def _render_generalizability_tab(st, data: DashboardData, settings: Dict[str, Any]) -> None:
    st.subheader("Generalizability assessment")
    has_any = (
        data.temporal_validation_results
        or data.multisite_validation_results
        or data.external_cohorts_results
    )
    if not has_any:
        st.info(
            "No generalizability results found. Enable `generalizability` in the "
            "config and re-run the pipeline."
        )
        return

    _render_summary_block(st, data.generalizability_summary or {}, settings["decimals"])

    rows = _gen_cohort_summary_rows(data, settings["decimals"])
    if rows:
        st.markdown("### Cohort summary")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    _render_cohort_inspector(st, data, rows, settings)


def _render_drift_explorer_tab(st, data: DashboardData, settings: Dict[str, Any]) -> None:
    st.subheader("Drift explorer")
    if not data.drift_tables:
        st.info("No per-cohort drift tables saved.")
        return
    label = st.selectbox("Cohort", sorted(data.drift_tables.keys()), key="drift_explorer_cohort")
    df = data.drift_tables[label]
    if df is None or df.empty:
        st.info(f"Cohort `{label}` has an empty drift table.")
        return

    threshold = float(settings["psi"])
    top_k = st.slider("Top features", 5, 100, 25, 1, key=f"explorer_topk_{label}")

    feature_kind = st.multiselect(
        "Filter by kind",
        sorted(df["kind"].dropna().unique().tolist()) if "kind" in df.columns else [],
        default=sorted(df["kind"].dropna().unique().tolist()) if "kind" in df.columns else [],
        key=f"explorer_kind_{label}",
    )
    df_view = df.copy()
    if feature_kind and "kind" in df_view.columns:
        df_view = df_view[df_view["kind"].isin(feature_kind)]
    fig = _drift_bar_plot(df_view.to_dict("records"), top_k, threshold, settings["plot_height"])
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No features pass the threshold.")
    with st.expander("Full drift table", expanded=False):
        st.dataframe(df_view, width="stretch", hide_index=True)


def main() -> None:
    """Streamlit script entry."""
    st = require_streamlit()
    st.set_page_config(
        page_title="PhenoCluster Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_style(st)

    results_dir = _resolve_results_dir()

    try:
        data = load_results(results_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    settings = _settings_sidebar(st, data)
    _project_header(st, data)

    tabs = st.tabs(
        [
            "Overview",
            "Phenotypes",
            "Outcomes",
            "Survival",
            "Multistate",
            "Generalizability",
            "Drift explorer",
        ]
    )
    with tabs[0]:
        _render_overview_tab(st, data)
    with tabs[1]:
        _render_phenotypes_tab(st, data, settings)
    with tabs[2]:
        _render_outcomes_tab(st, data, settings)
    with tabs[3]:
        _render_survival_tab(st, data, settings)
    with tabs[4]:
        _render_multistate_tab(st, data, settings)
    with tabs[5]:
        _render_generalizability_tab(st, data, settings)
    with tabs[6]:
        _render_drift_explorer_tab(st, data, settings)


if __name__ == "__main__":
    main()
