"""Data quality assessor — orchestrates assessment and reporting."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...config import PhenoClusterConfig
from ...utils.logging import get_logger
from .figures import QualityFigureGenerator
from .mcar import littles_mcar_test


class DataQualityAssessor:
    """
    Performs comprehensive data quality assessment including missing data
    analysis, outlier detection, correlation analysis, and variance checks.
    """

    def __init__(self, config: PhenoClusterConfig):
        self.config = config
        self.logger = get_logger("data_quality", config)
        self.quality_report = {}
        self._figures = None

    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive data quality assessment."""
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
        self._figures = QualityFigureGenerator(self.config, quality_report)

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
        """Perform Little's MCAR test on the data."""
        self.logger.info("1b. Little's MCAR Test")

        continuous_cols_present = [
            col for col in self.config.continuous_columns if col in df.columns
        ]
        test_data = df[continuous_cols_present].copy()

        mcar_result = littles_mcar_test(test_data)

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

        if "missing_data" in quality_report:
            if quality_report["missing_data"]["high_missing_columns"]:
                issues.append(
                    f"{len(quality_report['missing_data']['high_missing_columns'])} "
                    f"columns with high missing data"
                )

        if "mcar_test" in quality_report:
            mcar_result = quality_report["mcar_test"]
            if mcar_result.get("is_mcar") is False:
                issues.append(
                    f"Missing data is NOT MCAR "
                    f"(p={mcar_result.get('p_value', 'N/A'):.4f}) - "
                    f"consider MAR-appropriate imputation"
                )

        if "outliers" in quality_report:
            if quality_report["outliers"]["total_outlier_count"] > 0:
                issues.append(
                    f"{quality_report['outliers']['total_outlier_count']} outliers detected"
                )

        if "correlation" in quality_report:
            if quality_report["correlation"]["n_high_correlations"] > 0:
                issues.append(
                    f"{quality_report['correlation']['n_high_correlations']} "
                    f"highly correlated variable pairs"
                )

        if "variance" in quality_report:
            if quality_report["variance"]["low_variance_columns"]:
                issues.append(
                    f"{len(quality_report['variance']['low_variance_columns'])} "
                    f"low-variance columns"
                )

        if "datatypes" in quality_report:
            if quality_report["datatypes"]["constant_columns"]:
                issues.append(
                    f"{len(quality_report['datatypes']['constant_columns'])} constant columns"
                )

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
        """Create missing data visualization."""
        if self._figures is None:
            return None
        return self._figures.create_missing_data_figure()

    def create_outlier_figure(self) -> Optional[go.Figure]:
        """Create outlier distribution visualization."""
        if self._figures is None:
            return None
        return self._figures.create_outlier_figure()

    def create_correlation_figure(self) -> Optional[go.Figure]:
        """Create correlation heatmap visualization."""
        if self._figures is None:
            return None
        return self._figures.create_correlation_figure()

    def create_variance_figure(self) -> Optional[go.Figure]:
        """Create variance distribution visualization."""
        if self._figures is None:
            return None
        return self._figures.create_variance_figure()

    def save_quality_report(self, output_dir: Path):
        """Save quality report to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        if self.config.data_quality.generate_report:
            self._save_figures(output_dir)

    def _save_figures(self, output_dir: Path):
        """Save individual data quality visualizations."""
        figures_saved = []

        fig_missing = self.create_missing_data_figure()
        if fig_missing:
            fig_missing.write_html(output_dir / "data_quality_missing_data.html")
            figures_saved.append("data_quality_missing_data.html")

        fig_outlier = self.create_outlier_figure()
        if fig_outlier:
            fig_outlier.write_html(output_dir / "data_quality_outliers.html")
            figures_saved.append("data_quality_outliers.html")

        fig_corr = self.create_correlation_figure()
        if fig_corr:
            fig_corr.write_html(output_dir / "data_quality_correlation.html")
            figures_saved.append("data_quality_correlation.html")

        fig_var = self.create_variance_figure()
        if fig_var:
            fig_var.write_html(output_dir / "data_quality_variance.html")
            figures_saved.append("data_quality_variance.html")

        if figures_saved:
            self.logger.info(f"Total {len(figures_saved)} data quality visualization(s) saved")
        else:
            self.logger.warning("No data quality visualizations were generated")
