"""Pipeline result serialization and I/O."""

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd


def _numpy_encoder(obj):
    """JSON encoder for numpy types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series, pd.Index)):
        return obj.tolist()
    return str(obj)


def save_pipeline_results(results, config, preprocessor, feature_selector,
                          reference_phenotype, logger,
                          output_dir: Optional[str] = None) -> None:
    """Save pipeline results to disk."""
    if not results:
        logger.warning("No results to save. Run fit() first.")
        return

    if output_dir is None:
        output_dir = config.output_dir

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "data"
    results_dir = output_path / "results"
    plots_dir = output_path / "plots"
    artifacts_dir = output_path / "artifacts"
    for d in [data_dir, results_dir, plots_dir, artifacts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSaving results to {output_dir}...")

    _save_data_files(results, data_dir, logger)
    _save_json_results(results, results_dir, reference_phenotype, logger)
    _save_model_selection(results, data_dir, results_dir, logger)
    _save_artifacts(config, preprocessor, feature_selector, artifacts_dir, logger)
    _save_plots(results, plots_dir, config, logger)
    _generate_report(output_path, config, logger)

    logger.info("All results saved successfully")


def _save_data_files(results, data_dir, logger):
    """Save processed data CSV files."""
    if "data" in results:
        results["data"].to_csv(data_dir / "phenotypes_data.csv", index=False)
        logger.info("  Saved data/phenotypes_data.csv")

    if "posterior_proba" in results:
        n_clusters = results["n_clusters"]
        pd.DataFrame(
            results["posterior_proba"],
            columns=[f"Cluster_{i}_Prob" for i in range(n_clusters)],
        ).to_csv(data_dir / "posterior_probabilities.csv", index=False)
        logger.info("  Saved data/posterior_probabilities.csv")

    if "model_fit_metrics" in results:
        pd.DataFrame([results["model_fit_metrics"]]).to_csv(
            data_dir / "model_fit_metrics.csv", index=False
        )
        logger.info("  Saved data/model_fit_metrics.csv")


def _save_json_results(results, results_dir, reference_phenotype, logger):
    """Save JSON result files."""
    _json_pairs = [
        ("cluster_stats", "cluster_statistics.json"),
        ("outcome_results", "outcome_results.json"),
        ("survival_results", "survival_results.json"),
        ("multistate_results", "multistate_results.json"),
        ("feature_importance", "feature_importance.json"),
        ("split_info", "split_info.json"),
        ("classification_quality", "classification_quality.json"),
        ("classification_quality_test", "classification_quality_test.json"),
        ("feature_selection", "feature_selection.json"),
    ]
    for key, filename in _json_pairs:
        if key in results and results[key]:
            with open(results_dir / filename, "w") as f:
                json.dump(results[key], f, indent=2, default=_numpy_encoder)
            logger.info(f"  Saved results/{filename}")

    # Stability results (also save consensus matrix)
    if "stability_results" in results and results["stability_results"]:
        with open(results_dir / "stability_results.json", "w") as f:
            json.dump(results["stability_results"], f, indent=2, default=_numpy_encoder)
        if "consensus_matrix" in results["stability_results"]:
            np.save(
                results_dir / "consensus_matrix.npy",
                results["stability_results"]["consensus_matrix"],
            )
        logger.info("  Saved results/stability_results.json")

    # Validation report
    if "validation_metrics" in results and results["validation_metrics"]:
        validation_data = {
            **results["validation_metrics"],
            "reference_phenotype": reference_phenotype,
        }
        with open(results_dir / "validation_report.json", "w") as f:
            json.dump(validation_data, f, indent=2, default=_numpy_encoder)
        logger.info("  Saved results/validation_report.json")

    # External validation (exclude plot objects)
    if "external_validation_results" in results and results["external_validation_results"]:
        ext_data = {
            k: v for k, v in results["external_validation_results"].items() if k != "plots"
        }
        with open(results_dir / "external_validation_results.json", "w") as f:
            json.dump(ext_data, f, indent=2, default=_numpy_encoder)
        logger.info("  Saved results/external_validation_results.json")


def _save_model_selection(results, data_dir, results_dir, logger):
    """Save model selection results."""
    if "model_selection" not in results or not results["model_selection"]:
        return

    model_sel = results["model_selection"]
    criterion_name = model_sel.get("criterion_used", model_sel.get("criterion", "BIC"))
    best_k = model_sel.get("best_n_clusters")

    best_criterion_value = None
    for r in model_sel.get("all_results", []):
        if r.get("n_clusters") == best_k:
            best_criterion_value = r.get(criterion_name)
            break

    selection_summary = {
        "best_n_clusters": best_k,
        "criterion_used": criterion_name,
        "best_criterion_value": best_criterion_value,
    }

    comparison_table = model_sel.get("comparison_table")
    if comparison_table is not None and not comparison_table.empty:
        cv_results = model_sel.get("cv_results")
        if cv_results is not None:
            _ic = {"BIC", "AIC", "CAIC", "SABIC", "ICL"}
            sign = -1.0 if criterion_name.upper() in _ic else 1.0
            all_results = []
            for _, row in comparison_table.iterrows():
                all_results.append({
                    "n_clusters": int(row["n_clusters"]),
                    "mean_score": float(row["mean_score"]) * sign,
                    "std_score": abs(float(row["std_score"])),
                    "rank": int(row["rank"]),
                })
            pd.DataFrame(all_results).to_csv(
                data_dir / "model_selection_results.csv", index=False
            )
            logger.info("  Saved data/model_selection_results.csv")

    with open(results_dir / "model_selection_summary.json", "w") as f:
        json.dump(selection_summary, f, indent=2)
    logger.info("  Saved results/model_selection_summary.json")


def _save_artifacts(config, preprocessor, feature_selector, artifacts_dir, logger):
    """Save model artifacts."""
    if feature_selector is not None:
        try:
            joblib.dump(feature_selector, artifacts_dir / "feature_selector.joblib")
            logger.info("  Saved artifacts/feature_selector.joblib")
        except Exception as e:
            logger.warning(f"  Could not save feature_selector: {e}")

    config.save(artifacts_dir / "config_used.yaml", format="yaml")
    logger.info("  Saved artifacts/config_used.yaml")

    if preprocessor is not None:
        try:
            if preprocessor.label_encoders:
                joblib.dump(
                    preprocessor.label_encoders,
                    artifacts_dir / "label_encoders.joblib",
                )
                logger.info("  Saved artifacts/label_encoders.joblib")
            joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
            logger.info("  Saved artifacts/preprocessor.joblib")
        except Exception as e:
            logger.warning(f"  Could not save preprocessor: {e}")


def _save_plots(results, plots_dir, config, logger):
    """Save plot HTML files."""
    if "plots" in results and config.visualization.save_plots:
        for plot_name, fig in results["plots"].items():
            if fig is not None:
                try:
                    fig.write_html(plots_dir / f"{plot_name}.html")
                    logger.info(f"  Saved plots/{plot_name}.html")
                except Exception as e:
                    logger.warning(f"  Could not save {plot_name}.html: {e}")

    ext_plots = results.get("external_validation_results", {}).get("plots", {})
    if ext_plots and config.visualization.save_plots:
        for plot_name, fig in ext_plots.items():
            if fig is not None:
                try:
                    fig.write_html(plots_dir / f"{plot_name}.html")
                except Exception as e:
                    logger.warning(f"  Could not save {plot_name}.html: {e}")
        logger.info(f"  Saved {len(ext_plots)} external validation plots")


def _generate_report(output_path, config, logger):
    """Generate HTML report."""
    try:
        from ..utils.report import generate_html_report
        generate_html_report(
            results_dir=output_path,
            title=f"{config.project_name} Analysis Report",
        )
        logger.info("  Generated analysis_report.html")
    except Exception as e:
        logger.warning(f"  Could not generate HTML report: {e}")
