"""
PhenoCluster Generalizability Visualizations
============================================

Plotly visualizations for the v0.3.0 generalizability outputs:
phenotype prevalence comparison, top-K feature drift, OR/HR concordance
scatter, and a LOGO/window summary forest plot.

These plots consume the *plain dicts* produced by
:meth:`GeneralizabilityReport.to_dict()` (so they work both during the
pipeline and from JSON-loaded artifacts).
"""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._base import STYLE


def _default_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            y=0.97,
            yanchor="top",
            font=dict(size=STYLE["title_font_size"]),
        ),
        plot_bgcolor=STYLE["plot_bgcolor"],
        paper_bgcolor=STYLE["paper_bgcolor"],
        font=dict(family=STYLE["font_family"], size=STYLE["font_size"]),
        margin=dict(l=80, r=40, t=80, b=60),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def create_cohort_prevalence_heatmap(
    cohorts: Iterable[Dict[str, Any]],
    title: str = "Phenotype prevalence by cohort",
) -> Optional[go.Figure]:
    """Heatmap of phenotype prevalence (percentage) across validation cohorts.

    Each row is a cohort and each column is a phenotype.
    """
    cohort_list = list(cohorts)
    if not cohort_list:
        return None

    phenotype_set = set()
    for c in cohort_list:
        phenotype_set.update((c.get("cluster_distribution") or {}).keys())
    if not phenotype_set:
        return None
    phenotypes = sorted(phenotype_set, key=lambda v: int(v) if str(v).isdigit() else str(v))
    labels = [str(c.get("label", "?")) for c in cohort_list]

    matrix = np.full((len(cohort_list), len(phenotypes)), np.nan)
    for i, c in enumerate(cohort_list):
        dist = c.get("cluster_distribution") or {}
        for j, p in enumerate(phenotypes):
            entry = dist.get(p)
            if entry is None:
                continue
            matrix[i, j] = float(entry.get("percentage", float("nan")))

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[f"Phenotype {p}" for p in phenotypes],
            y=labels,
            colorscale="Viridis",
            colorbar=dict(title="%"),
            hovertemplate="Cohort=%{y}<br>%{x}<br>%{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(320, 36 * len(labels) + 160),
        xaxis=dict(tickangle=-30, automargin=True),
        yaxis=dict(automargin=True),
        margin=dict(l=160, r=40, t=80, b=120),
    )
    return _default_layout(fig, title)


def create_drift_bar_chart(
    drift: List[Dict[str, Any]],
    top_k: int = 20,
    title: str = "Top features by PSI",
) -> Optional[go.Figure]:
    """Horizontal bar of the top-K features by absolute PSI."""
    if not drift:
        return None
    df = pd.DataFrame(drift)
    if df.empty or "psi" not in df.columns:
        return None
    df = df.dropna(subset=["psi"])
    if df.empty:
        return None
    df = df.assign(_abs=df["psi"].abs()).sort_values("_abs", ascending=False).head(top_k)
    df = df.iloc[::-1]
    fig = go.Figure(
        data=go.Bar(
            x=df["psi"],
            y=df["feature"],
            orientation="h",
            marker=dict(
                color=df["psi"], colorscale="RdBu_r", cmin=-df["_abs"].max(), cmax=df["_abs"].max()
            ),
            hovertemplate=("Feature=%{y}<br>PSI=%{x:.3f}<extra></extra>"),
        )
    )
    fig.update_layout(
        xaxis_title="Population Stability Index (PSI)",
        yaxis_title="Feature",
        height=max(360, 28 * len(df) + 140),
    )
    fig.update_layout(margin=dict(l=180, r=40, t=80, b=60))
    return _default_layout(fig, title)


def create_or_concordance_scatter(
    concordance: Dict[str, Any],
    title: str = "Outcome concordance",
) -> Optional[go.Figure]:
    """Scatter of derivation log(OR) vs validation log(OR) per phenotype."""
    if not concordance:
        return None
    rows = []
    for outcome, payload in (concordance.get("outcomes") or {}).items():
        for entry in payload.get("per_phenotype", []) or []:
            rows.append(
                {
                    "outcome": outcome,
                    "phenotype": entry.get("phenotype"),
                    "log_d": entry.get("log_effect_derivation"),
                    "log_v": entry.get("log_effect_validation"),
                    "fdr": entry.get("p_value_fdr"),
                }
            )
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["log_d", "log_v"])
    if df.empty:
        return None

    fig = go.Figure()
    palette = STYLE["cluster_colors"]
    for i, outcome in enumerate(sorted(df["outcome"].unique())):
        sub = df[df["outcome"] == outcome]
        fig.add_trace(
            go.Scatter(
                x=sub["log_d"],
                y=sub["log_v"],
                mode="markers+text",
                text=[str(p) for p in sub["phenotype"]],
                textposition="top center",
                name=outcome,
                marker=dict(size=10, color=palette[i % len(palette)]),
                hovertemplate=(
                    "Phenotype=%{text}<br>log(OR)_d=%{x:.2f}<br>log(OR)_v=%{y:.2f}<extra>"
                    + outcome
                    + "</extra>"
                ),
            )
        )
    lo = float(min(df["log_d"].min(), df["log_v"].min()))
    hi = float(max(df["log_d"].max(), df["log_v"].max()))
    pad = 0.1 * max(abs(lo), abs(hi), 0.5)
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="identity",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis=dict(
            title="log(OR) derivation",
            zeroline=True,
            zerolinecolor="lightgray",
            automargin=True,
        ),
        yaxis=dict(
            title="log(OR) validation",
            zeroline=True,
            zerolinecolor="lightgray",
            automargin=True,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=520,
    )
    fig.update_traces(textposition="top center", cliponaxis=False)
    return _default_layout(fig, title)


def create_logo_forest_plot(
    cohorts: Iterable[Dict[str, Any]],
    metric: str = "ari",
    title: Optional[str] = None,
) -> Optional[go.Figure]:
    """Per-cohort dot plot of refit-and-match metric (ARI by default)."""
    cohort_list = list(cohorts)
    if not cohort_list:
        return None
    rows = []
    for c in cohort_list:
        refit = c.get("refit") or {}
        if metric not in refit:
            continue
        rows.append({"label": c.get("label", "?"), "value": float(refit[metric])})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("value")
    fig = go.Figure(
        data=go.Scatter(
            x=df["value"],
            y=df["label"],
            mode="markers",
            marker=dict(size=12, color=STYLE["cluster_colors"][0]),
            hovertemplate="Cohort=%{y}<br>" + metric.upper() + "=%{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(title=metric.upper(), zeroline=True, automargin=True),
        yaxis=dict(title="Cohort", automargin=True),
        height=max(320, 32 * len(df) + 140),
        margin=dict(l=180, r=40, t=80, b=60),
    )
    return _default_layout(fig, title or f"Per-cohort {metric.upper()}")


def create_all_generalizability_plots(
    generalizability_results: Optional[Dict[str, Any]],
    drift_top_k: int = 20,
) -> Dict[str, go.Figure]:
    """Build a dict of plotly figures from a generalizability results dict."""
    plots: Dict[str, go.Figure] = {}
    if not generalizability_results:
        return plots

    temporal = generalizability_results.get("temporal") or []
    multisite = generalizability_results.get("multisite") or []
    external = generalizability_results.get("external") or []

    if temporal:
        fig = create_cohort_prevalence_heatmap(
            temporal, title="Phenotype prevalence by temporal cohort"
        )
        if fig is not None:
            plots["gen_prevalence_temporal"] = fig
        forest = create_logo_forest_plot(temporal, metric="ari")
        if forest is not None:
            plots["gen_temporal_ari"] = forest
    if multisite:
        fig = create_cohort_prevalence_heatmap(multisite, title="Phenotype prevalence by site")
        if fig is not None:
            plots["gen_prevalence_multisite"] = fig
        forest = create_logo_forest_plot(multisite, metric="ari")
        if forest is not None:
            plots["gen_multisite_ari"] = forest
    if external:
        fig = create_cohort_prevalence_heatmap(
            external, title="Phenotype prevalence by external cohort"
        )
        if fig is not None:
            plots["gen_prevalence_external"] = fig
        forest = create_logo_forest_plot(external, metric="ari")
        if forest is not None:
            plots["gen_external_ari"] = forest

    for cohort in temporal + multisite + external:
        label = str(cohort.get("label", "cohort")).replace(" ", "_")
        drift = cohort.get("drift")
        drift_records = drift if isinstance(drift, list) else None
        if isinstance(drift, pd.DataFrame):
            drift_records = drift.to_dict(orient="records")
        if drift_records:
            fig = create_drift_bar_chart(
                drift_records, top_k=drift_top_k, title=f"Top drifted features - {label}"
            )
            if fig is not None:
                plots[f"gen_drift_{label}"] = fig
        concordance = cohort.get("outcome_concordance")
        if concordance:
            fig = create_or_concordance_scatter(concordance, title=f"OR concordance - {label}")
            if fig is not None:
                plots[f"gen_concordance_{label}"] = fig
    return plots
