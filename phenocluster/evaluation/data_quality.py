"""
PhenoCluster Data Quality Module
=================================

Comprehensive data quality assessment including missing data analysis
and outlier profiling.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


def littles_mcar_test(data: pd.DataFrame) -> Dict:
    """
    Perform Little's MCAR (Missing Completely at Random) test.

    Little's MCAR test (Little, 1988) tests the null hypothesis that the
    missing data mechanism is MCAR. The test compares observed variable means
    for each missing data pattern against expected means if data were MCAR.

    Under MCAR, the test statistic follows a chi-square distribution.
    A significant p-value (typically p < 0.05) suggests data is NOT MCAR.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with potential missing values

    Returns
    -------
    Dict
        Dictionary containing:
        - chi_square: Test statistic
        - dof: Degrees of freedom
        - p_value: P-value for the test
        - is_mcar: Boolean indicating if null hypothesis (MCAR) is accepted at alpha=0.05
        - n_patterns: Number of unique missing data patterns
        - interpretation: Human-readable interpretation

    References
    ----------
    Little, R.J.A. (1988). A Test of Missing Completely at Random for
    Multivariate Data with Missing Values. Journal of the American
    Statistical Association, 83(404), 1198-1202.

    Notes
    -----
    - Requires at least 2 variables with missing data
    - Test may be unreliable with very small samples or many missing patterns
    - Non-significant result does not prove MCAR, only fails to reject it
    """
    numeric_data = data.select_dtypes(include=[np.number]).copy()

    if numeric_data.empty:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": None,
            "n_patterns": 0,
            "interpretation": "No numeric columns available for MCAR test",
        }

    missing_mask = numeric_data.isna()
    n_missing_cols = missing_mask.any().sum()

    if n_missing_cols == 0:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": True,
            "n_patterns": 1,
            "interpretation": "No missing data - MCAR test not applicable",
        }

    if n_missing_cols < 2:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": None,
            "n_patterns": 1,
            "interpretation": (
                "Less than 2 variables with missing data - test requires multiple variables"
            ),
        }

    pattern_strings = pd.Series(
        ["".join(row) for row in missing_mask.astype(int).astype(str).values],
        index=missing_mask.index,
    )
    unique_patterns = pattern_strings.unique()
    n_patterns = len(unique_patterns)

    if n_patterns == 1:
        return {
            "chi_square": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "is_mcar": None,
            "n_patterns": 1,
            "interpretation": "Only one missing data pattern - test requires multiple patterns",
        }

    n_vars = numeric_data.shape[1]
    overall_means = numeric_data.mean()
    cov_matrix = numeric_data.cov()

    # Check if covariance matrix is singular
    try:
        np.linalg.inv(cov_matrix.values)
    except np.linalg.LinAlgError:
        # Use pseudo-inverse for singular matrices
        warnings.warn(
            "Covariance matrix is singular; using pseudoinverse. "
            "MCAR test results may be unreliable.",
            stacklevel=2,
        )

    chi_square = 0.0
    df_total = 0

    for pattern in unique_patterns:
        # Get indices for this pattern
        pattern_mask = pattern_strings == pattern
        n_pattern = pattern_mask.sum()

        if n_pattern < 2:
            continue

        # Get the pattern data
        pattern_data = numeric_data.loc[pattern_mask]

        # Identify which variables are observed (not missing) in this pattern
        observed_vars = []
        for i, char in enumerate(pattern):
            if char == "0":  # 0 means observed (not missing)
                observed_vars.append(numeric_data.columns[i])

        if len(observed_vars) < 1:
            continue

        # Calculate mean of observed variables for this pattern
        pattern_means = pattern_data[observed_vars].mean()

        # Get the submatrix of covariance for observed variables
        cov_sub = cov_matrix.loc[observed_vars, observed_vars]

        try:
            cov_sub_inv = np.linalg.inv(cov_sub.values)
        except np.linalg.LinAlgError:
            cov_sub_inv = np.linalg.pinv(cov_sub.values)

        # Calculate deviation from overall means
        mean_diff = pattern_means.values - overall_means[observed_vars].values

        # Contribution to chi-square: n * (x_bar - mu)' * Sigma^-1 * (x_bar - mu)
        chi_sq_contribution = n_pattern * np.dot(np.dot(mean_diff, cov_sub_inv), mean_diff)

        chi_square += chi_sq_contribution
        df_total += len(observed_vars)

    # Adjust degrees of freedom
    # df = sum of observed variables across patterns - number of variables
    df = max(1, df_total - n_vars)

    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi_square, df)

    # Interpretation
    alpha = 0.05
    is_mcar = p_value > alpha

    if is_mcar:
        interpretation = (
            f"Little's MCAR test: chi-square({df}) = {chi_square:.3f}, p = {p_value:.4f}. "
            f"The null hypothesis (MCAR) is NOT rejected at alpha={alpha}. "
            f"Missing data appears to be Missing Completely at Random."
        )
    else:
        interpretation = (
            f"Little's MCAR test: chi-square({df}) = {chi_square:.3f}, p = {p_value:.4f}. "
            f"The null hypothesis (MCAR) IS rejected at alpha={alpha}. "
            f"Missing data is NOT Missing Completely at Random. "
            f"Consider MAR-appropriate methods (e.g., multiple imputation, maximum likelihood)."
        )

    return {
        "chi_square": float(chi_square),
        "dof": int(df),
        "p_value": float(p_value),
        "is_mcar": bool(is_mcar),
        "n_patterns": int(n_patterns),
        "interpretation": interpretation,
    }


class DataQualityAssessor:
    """
    Performs comprehensive data quality assessment including missing data
    analysis, outlier detection, correlation analysis, and variance checks.
    """

    def __init__(self, config: PhenoClusterConfig):
        """
        Initialize the data quality assessor.

        Parameters
        ----------
        config : PhenoClusterConfig
            Configuration object
        """
        self.config = config
        self.logger = get_logger("data_quality", config)

        self.quality_report = {}

    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive data quality assessment.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        Dict
            Dictionary containing quality metrics and issues
        """
        self.logger.info("DATA QUALITY ASSESSMENT")

        quality_report = {}

        # 1. Missing data analysis
        quality_report["missing_data"] = self._assess_missing_data(df)

        # 1b. Little's MCAR test (if there is missing data)
        if quality_report["missing_data"]["columns_with_missing"] > 0:
            quality_report["mcar_test"] = self._assess_mcar(df)

        # 2. Outlier analysis (if enabled)
        if self.config.outlier.enabled:
            quality_report["outliers"] = self._assess_outliers(df)

        # 3. Correlation analysis
        quality_report["correlation"] = self._assess_correlation(df)

        # 4. Variance analysis
        quality_report["variance"] = self._assess_variance(df)

        # 5. Data types and uniqueness
        quality_report["datatypes"] = self._assess_datatypes(df)

        # 6. Summary statistics
        quality_report["summary"] = self._generate_summary(quality_report)

        self.quality_report = quality_report

        # Log summary
        self._log_quality_summary(quality_report)

        return quality_report

    def _assess_missing_data(self, df: pd.DataFrame) -> Dict:
        """Assess missing data patterns."""
        self.logger.info("1. Missing Data Analysis")

        feature_cols = self.config.continuous_columns + self.config.categorical_columns
        missing_info = {}

        for col in feature_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100

                missing_info[col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_pct),
                    "has_missing": missing_count > 0,
                }

                if missing_pct > self.config.data_quality.missing_threshold * 100:
                    self.logger.warning(
                        f"  {col}: {missing_pct:.2f}% missing "
                        f"(exceeds threshold of "
                        f"{self.config.data_quality.missing_threshold * 100}%)"
                    )

        total_missing = sum(info["count"] for info in missing_info.values())
        total_cells = len(df) * len(feature_cols)
        overall_missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0

        result = {
            "by_column": missing_info,
            "overall_missing_count": int(total_missing),
            "overall_missing_percentage": float(overall_missing_pct),
            "columns_with_missing": sum(1 for info in missing_info.values() if info["has_missing"]),
            "high_missing_columns": [
                col
                for col, info in missing_info.items()
                if info["percentage"] > self.config.data_quality.missing_threshold * 100
            ],
        }

        self.logger.info(f"  Overall missing: {overall_missing_pct:.2f}%")
        self.logger.info(
            f"  Columns with missing data: {result['columns_with_missing']}/{len(feature_cols)}"
        )

        return result

    def _assess_mcar(self, df: pd.DataFrame) -> Dict:
        """
        Perform Little's MCAR test on the data.

        Tests the null hypothesis that missing data is Missing Completely
        at Random (MCAR). This is important for validating imputation
        assumptions.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe

        Returns
        -------
        Dict
            Dictionary containing MCAR test results
        """
        self.logger.info("1b. Little's MCAR Test")

        # Little's MCAR test is only meaningful for continuous variables;
        # integer-encoded categoricals produce meaningless mean comparisons
        continuous_cols_present = [
            col for col in self.config.continuous_columns if col in df.columns
        ]
        test_data = df[continuous_cols_present].copy()

        # Run Little's MCAR test
        mcar_result = littles_mcar_test(test_data)

        # Log results
        if mcar_result["p_value"] is not None and not np.isnan(mcar_result["p_value"]):
            self.logger.info(f"  Chi-square statistic: {mcar_result['chi_square']:.3f}")
            self.logger.info(f"  Degrees of freedom: {mcar_result['dof']}")
            self.logger.info(f"  P-value: {mcar_result['p_value']:.4f}")
            self.logger.info(f"  Number of missing patterns: {mcar_result['n_patterns']}")

            if mcar_result["is_mcar"]:
                self.logger.info("  Result: Data appears to be MCAR (null hypothesis not rejected)")
            else:
                self.logger.warning("  Result: Data is NOT MCAR (null hypothesis rejected)")
                self.logger.warning("  Consider using MAR-appropriate imputation methods")
        else:
            self.logger.info(f"  {mcar_result['interpretation']}")

        return mcar_result

    def _assess_outliers(self, df: pd.DataFrame) -> Dict:
        """Assess outliers in continuous variables."""
        self.logger.info("2. Outlier Analysis")

        outlier_info = {}

        for col in self.config.continuous_columns:
            if col not in df.columns:
                continue

            data = df[col].dropna()

            if len(data) == 0:
                continue

            # Calculate IQR-based outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data < lower_bound) | (data > upper_bound)]
            n_outliers = len(outliers)
            outlier_pct = (n_outliers / len(data)) * 100

            outlier_info[col] = {
                "count": int(n_outliers),
                "percentage": float(outlier_pct),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
            }

            if n_outliers > 0:
                self.logger.info(
                    f"  {col}: {n_outliers} outliers ({outlier_pct:.2f}%), "
                    f"range=[{lower_bound:.2f}, {upper_bound:.2f}]"
                )

        total_outliers = sum(info["count"] for info in outlier_info.values())

        result = {
            "by_column": outlier_info,
            "total_outlier_count": int(total_outliers),
            "columns_with_outliers": sum(1 for info in outlier_info.values() if info["count"] > 0),
        }

        self.logger.info(f"  Total outliers detected: {total_outliers}")

        return result

    def _assess_correlation(self, df: pd.DataFrame) -> Dict:
        """Assess correlations between continuous variables."""
        self.logger.info("3. Correlation Analysis")

        if len(self.config.continuous_columns) < 2:
            self.logger.info("  Not enough continuous columns for correlation analysis")
            return {"high_correlations": []}

        cont_data = df[self.config.continuous_columns].dropna()

        if len(cont_data) == 0:
            return {"high_correlations": []}

        corr_matrix = cont_data.corr(method="spearman")

        high_correlations = []
        threshold = self.config.data_quality.correlation_threshold

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) > threshold:
                    high_correlations.append(
                        {
                            "variable1": corr_matrix.columns[i],
                            "variable2": corr_matrix.columns[j],
                            "correlation": float(corr_value),
                        }
                    )

                    self.logger.warning(
                        f"  High correlation: {corr_matrix.columns[i]} <-> "
                        f"{corr_matrix.columns[j]}: r={corr_value:.3f}"
                    )

        if not high_correlations:
            self.logger.info("  No high correlations detected")

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": high_correlations,
            "n_high_correlations": len(high_correlations),
        }

    def _assess_variance(self, df: pd.DataFrame) -> Dict:
        """Assess variance in continuous variables."""
        self.logger.info("4. Variance Analysis")

        variance_info = {}
        low_variance_cols = []
        threshold = self.config.data_quality.variance_threshold

        for col in self.config.continuous_columns:
            if col not in df.columns:
                continue

            data = df[col].dropna()

            if len(data) == 0:
                continue

            variance = data.var()
            std = data.std()

            variance_info[col] = {
                "variance": float(variance),
                "std": float(std),
                "coefficient_of_variation": float(std / abs(data.mean()))
                if abs(data.mean()) > 1e-10
                else np.inf,
            }

            if variance < threshold:
                low_variance_cols.append(col)
                self.logger.warning(f"  Low variance: {col}: var={variance:.4f}")

        if not low_variance_cols:
            self.logger.info("  All variables have sufficient variance")

        return {
            "by_column": variance_info,
            "low_variance_columns": low_variance_cols,
            "n_low_variance": len(low_variance_cols),
        }

    def _assess_datatypes(self, df: pd.DataFrame) -> Dict:
        """Assess data types and uniqueness."""
        self.logger.info("5. Data Types and Uniqueness")

        datatype_info = {}

        feature_cols = self.config.continuous_columns + self.config.categorical_columns

        for col in feature_cols:
            if col not in df.columns:
                continue

            data = df[col]
            n_unique = data.nunique()
            uniqueness_ratio = n_unique / len(data)

            datatype_info[col] = {
                "dtype": str(data.dtype),
                "n_unique": int(n_unique),
                "uniqueness_ratio": float(uniqueness_ratio),
                "is_constant": n_unique == 1,
            }

            if n_unique == 1:
                self.logger.warning(f"  Constant variable: {col}")

        return {
            "by_column": datatype_info,
            "constant_columns": [col for col, info in datatype_info.items() if info["is_constant"]],
        }

    def _generate_summary(self, quality_report: Dict) -> Dict:
        """Generate overall quality summary."""
        issues = []

        # Check for high missing data
        if "missing_data" in quality_report:
            if quality_report["missing_data"]["high_missing_columns"]:
                issues.append(
                    f"{len(quality_report['missing_data']['high_missing_columns'])} "
                    f"columns with high missing data"
                )

        # Check MCAR test result
        if "mcar_test" in quality_report:
            mcar_result = quality_report["mcar_test"]
            if mcar_result.get("is_mcar") is False:
                issues.append(
                    f"Missing data is NOT MCAR (p={mcar_result.get('p_value', 'N/A'):.4f}) - "
                    f"consider MAR-appropriate imputation"
                )

        # Check for outliers
        if "outliers" in quality_report:
            if quality_report["outliers"]["total_outlier_count"] > 0:
                issues.append(
                    f"{quality_report['outliers']['total_outlier_count']} outliers detected"
                )

        # Check for high correlations
        if "correlation" in quality_report:
            if quality_report["correlation"]["n_high_correlations"] > 0:
                issues.append(
                    f"{quality_report['correlation']['n_high_correlations']} "
                    f"highly correlated variable pairs"
                )

        # Check for low variance
        if "variance" in quality_report:
            if quality_report["variance"]["low_variance_columns"]:
                issues.append(
                    f"{len(quality_report['variance']['low_variance_columns'])} "
                    f"low-variance columns"
                )

        # Check for constant columns
        if "datatypes" in quality_report:
            if quality_report["datatypes"]["constant_columns"]:
                issues.append(
                    f"{len(quality_report['datatypes']['constant_columns'])} constant columns"
                )

        # Extract overall missing rate from quality report
        overall_missing_rate = 0.0
        if "missing_data" in quality_report:
            overall_missing_rate = (
                quality_report["missing_data"].get("overall_missing_percentage", 0) / 100
            )

        return {
            "n_issues": len(issues),
            "issues": issues,
            "overall_missing_rate": overall_missing_rate,
        }

    def _log_quality_summary(self, quality_report: Dict):
        """Log quality summary."""
        if "summary" not in quality_report:
            return

        summary = quality_report["summary"]

        self.logger.info("DATA QUALITY SUMMARY")
        self.logger.info(f"Issues Found: {summary['n_issues']}")

        if summary["issues"]:
            self.logger.info("Issues:")
            for issue in summary["issues"]:
                self.logger.info(f"  - {issue}")
        else:
            self.logger.info("  No major issues detected")

    def create_missing_data_figure(self) -> Optional[go.Figure]:
        """
        Create missing data visualization.

        Returns
        -------
        Optional[go.Figure]
            Plotly figure with missing data visualization, or None if no data
        """
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
        """
        Create outlier distribution visualization.
        Only creates figure if outlier analysis was enabled.

        Returns
        -------
        Optional[go.Figure]
            Plotly figure with outlier visualization, or None if outliers not enabled
        """
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
        """
        Create correlation heatmap visualization.
        Only creates figure if high correlations were detected.
        Shows lower triangular matrix without diagonal.

        Returns
        -------
        Optional[go.Figure]
            Plotly figure with correlation heatmap, or None if no high correlations
        """
        if not self.quality_report or "correlation" not in self.quality_report:
            return None

        if not self.quality_report["correlation"]["high_correlations"]:
            return None

        high_corrs = self.quality_report["correlation"]["high_correlations"]

        # Get variables with high correlations
        vars_with_high_corr = set()
        for corr in high_corrs:
            vars_with_high_corr.add(corr["variable1"])
            vars_with_high_corr.add(corr["variable2"])

        if not vars_with_high_corr:
            return None

        # Create correlation matrix subset
        corr_matrix_full = pd.DataFrame(self.quality_report["correlation"]["correlation_matrix"])
        vars_list = sorted(list(vars_with_high_corr))
        corr_matrix_subset = corr_matrix_full.loc[vars_list, vars_list]

        # Create lower triangular mask (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix_subset, dtype=bool), k=1)
        corr_matrix_masked = corr_matrix_subset.copy()
        corr_matrix_masked.values[mask] = np.nan

        # Create text matrix for annotations
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
        """
        Create variance distribution visualization.

        Returns
        -------
        Optional[go.Figure]
            Plotly figure with variance visualization, or None if no data
        """
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

    def save_quality_report(self, output_dir: Path):
        """Save quality report to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        import json

        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        report_clean = convert_types(self.quality_report)

        with open(output_dir / "data_quality_report.json", "w") as f:
            json.dump(report_clean, f, indent=2)

        self.logger.info(f"Quality report saved to {output_dir / 'data_quality_report.json'}")

        # Save individual visualizations if enabled
        if self.config.data_quality.generate_report:
            figures_saved = []

            # 1. Missing data figure
            fig_missing = self.create_missing_data_figure()
            if fig_missing:
                fig_missing.write_html(output_dir / "data_quality_missing_data.html")
                figures_saved.append("data_quality_missing_data.html")
                self.logger.info(
                    f"Missing data visualization saved to "
                    f"{output_dir / 'data_quality_missing_data.html'}"
                )

            # 2. Outlier figure (only if enabled)
            fig_outlier = self.create_outlier_figure()
            if fig_outlier:
                fig_outlier.write_html(output_dir / "data_quality_outliers.html")
                figures_saved.append("data_quality_outliers.html")
                self.logger.info(
                    f"Outlier visualization saved to {output_dir / 'data_quality_outliers.html'}"
                )

            # 3. Correlation figure (only if high correlations found)
            fig_corr = self.create_correlation_figure()
            if fig_corr:
                fig_corr.write_html(output_dir / "data_quality_correlation.html")
                figures_saved.append("data_quality_correlation.html")
                self.logger.info(
                    f"Correlation visualization saved to "
                    f"{output_dir / 'data_quality_correlation.html'}"
                )

            # 4. Variance figure
            fig_var = self.create_variance_figure()
            if fig_var:
                fig_var.write_html(output_dir / "data_quality_variance.html")
                figures_saved.append("data_quality_variance.html")
                self.logger.info(
                    f"Variance visualization saved to {output_dir / 'data_quality_variance.html'}"
                )

            if figures_saved:
                self.logger.info(f"Total {len(figures_saved)} data quality visualization(s) saved")
            else:
                self.logger.warning("No data quality visualizations were generated")
