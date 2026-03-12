"""Data quality visualization figures."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class QualityFigureGenerator:
    """Generates Plotly figures for data quality reports."""

    def __init__(self, config, quality_report: Dict):
        self.config = config
        self.quality_report = quality_report

    def create_missing_data_figure(self) -> Optional[go.Figure]:
        """Create missing data visualization."""
        if not self.quality_report or "missing_data" not in self.quality_report:
            return None

        missing_data = self.quality_report["missing_data"]["by_column"]
        if not missing_data:
            return None

        cols = list(missing_data.keys())
        missing_pcts = [missing_data[col]["percentage"] for col in cols]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=cols,
                    y=missing_pcts,
                    marker_color=[
                        "red"
                        if p > self.config.data_quality.missing_threshold * 100
                        else "lightblue"
                        for p in missing_pcts
                    ],
                    text=[f"{p:.1f}%" for p in missing_pcts],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Missing Data by Column",
            xaxis_title="Column",
            yaxis_title="Missing Data (%)",
            height=500,
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-45)
        return fig

    def create_outlier_figure(self) -> Optional[go.Figure]:
        """Create outlier distribution visualization."""
        if not self.config.outlier.enabled:
            return None
        if not self.quality_report or "outliers" not in self.quality_report:
            return None

        outlier_data = self.quality_report["outliers"]["by_column"]
        if not outlier_data:
            return None

        cols = list(outlier_data.keys())
        outlier_pcts = [outlier_data[col]["percentage"] for col in cols]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=cols,
                    y=outlier_pcts,
                    marker_color="orange",
                    text=[f"{p:.1f}%" for p in outlier_pcts],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Outlier Distribution by Column",
            xaxis_title="Column",
            yaxis_title="Outliers (%)",
            height=500,
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-45)
        return fig

    def create_correlation_figure(self) -> Optional[go.Figure]:
        """Create correlation heatmap visualization (lower triangular)."""
        if not self.quality_report or "correlation" not in self.quality_report:
            return None
        if not self.quality_report["correlation"]["high_correlations"]:
            return None

        high_corrs = self.quality_report["correlation"]["high_correlations"]
        vars_with_high_corr = set()
        for corr in high_corrs:
            vars_with_high_corr.add(corr["variable1"])
            vars_with_high_corr.add(corr["variable2"])

        if not vars_with_high_corr:
            return None

        corr_matrix_full = pd.DataFrame(self.quality_report["correlation"]["correlation_matrix"])
        vars_list = sorted(list(vars_with_high_corr))
        corr_matrix_subset = corr_matrix_full.loc[vars_list, vars_list]

        mask = np.triu(np.ones_like(corr_matrix_subset, dtype=bool), k=1)
        corr_matrix_masked = corr_matrix_subset.copy()
        corr_matrix_masked.values[mask] = np.nan

        text_matrix = corr_matrix_masked.copy()
        text_matrix = text_matrix.map(lambda x: f"{x:.2f}" if not pd.isna(x) else "")

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=corr_matrix_masked.values,
                    x=corr_matrix_masked.columns,
                    y=corr_matrix_masked.index,
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=text_matrix.values,
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorbar=dict(title="Correlation"),
                )
            ]
        )
        fig.update_layout(
            title="Correlation Heatmap (High Correlations Only)",
            xaxis_title="",
            yaxis_title="",
            height=max(400, len(vars_list) * 40),
            width=max(500, len(vars_list) * 40),
        )
        return fig

    def create_variance_figure(self) -> Optional[go.Figure]:
        """Create variance distribution visualization."""
        if not self.quality_report or "variance" not in self.quality_report:
            return None

        variance_data = self.quality_report["variance"]["by_column"]
        if not variance_data:
            return None

        cols = list(variance_data.keys())
        variances = [variance_data[col]["variance"] for col in cols]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=cols,
                    y=variances,
                    marker_color="lightgreen",
                    text=[f"{v:.4f}" for v in variances],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            title="Variance Distribution by Column",
            xaxis_title="Column",
            yaxis_title="Variance",
            height=500,
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-45)
        return fig
