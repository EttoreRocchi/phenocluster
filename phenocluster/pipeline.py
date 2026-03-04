"""
PhenoCluster Pipeline Module
=============================

Main pipeline orchestrator that coordinates all components.
"""

import traceback
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import numpy as np
import pandas as pd


@contextmanager
def _suppress_stepmix_warnings() -> Iterator[None]:
    """Suppress expected numerical warnings from StepMix EM optimization.

    These occur when random initializations produce degenerate clusters
    (empty clusters, zero variance, 0/0 in categorical probabilities).
    StepMix handles these internally and selects the best valid solution.
    Scoped to a context manager to avoid masking warnings in user code.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"stepmix\.emission\.gaussian",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"stepmix\.emission\.categorical",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"scipy\.special\._logsumexp",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"numpy\.core\._methods",
        )
        warnings.filterwarnings(
            "ignore",
            module=r"stepmix",
            message=r"Initializations did not converge",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"One or more of the test scores are non-finite",
            module=r"sklearn",
        )
        yield


from .cache import ArtifactCache, _compute_data_fingerprint
from .config import PhenoClusterConfig
from .data import DataPreprocessor, DataSplitter
from .evaluation import ClusterEvaluator, DataQualityAssessor, StabilityAnalyzer
from .feature_selection import MixedDataFeatureSelector
from .model_selection import StepMixModelSelector
from .utils.logging import get_logger
from .utils.results_io import save_pipeline_results
from .visualization import Visualizer


def _compute_classification_quality(proba: np.ndarray, labels: np.ndarray) -> dict:
    """Compute classification quality metrics from posterior probabilities.

    Parameters
    ----------
    proba : np.ndarray
        N x K matrix of posterior probabilities.
    labels : np.ndarray
        Length-N array of modal class assignments.

    Returns
    -------
    dict
        Classification quality metrics including overall_avepp,
        overall_relative_entropy, n_clusters, n_samples,
        assignment_confidence, and per_phenotype breakdown.
    """
    if len(proba) == 0:
        return {}
    n_clusters = proba.shape[1]
    max_proba = np.max(proba, axis=1)
    eps = 1e-15
    sample_entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    max_entropy = np.log(n_clusters)
    rel_entropy = sample_entropy / max_entropy if max_entropy > 0 else sample_entropy

    result = {
        "overall_avepp": float(np.mean(max_proba)),
        "overall_relative_entropy": float(np.mean(rel_entropy)),
        "n_clusters": n_clusters,
        "n_samples": len(labels),
        "assignment_confidence": {
            "above_90": float(np.mean(max_proba > 0.9) * 100),
            "above_80": float(np.mean(max_proba > 0.8) * 100),
            "above_70": float(np.mean(max_proba > 0.7) * 100),
            "below_70": float(np.mean(max_proba <= 0.7) * 100),
        },
        "per_phenotype": {},
    }

    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() > 0:
            cluster_proba = max_proba[mask]
            cluster_entropy = rel_entropy[mask]
            result["per_phenotype"][int(k)] = {
                "n": int(mask.sum()),
                "avepp": float(np.mean(cluster_proba)),
                "avepp_std": float(np.std(cluster_proba)),
                "min_pp": float(np.min(cluster_proba)),
                "median_pp": float(np.median(cluster_proba)),
                "mean_entropy": float(np.mean(cluster_entropy)),
            }

    return result


class PhenoClusterPipeline:
    """
    Main pipeline for clinical phenotype discovery using latent class analysis.

    This class orchestrates the complete workflow:
    - Data preprocessing with multiple imputation
    - Latent class modeling with automatic model selection
    - Cluster validation and characterization
    - Outcome association analysis
    - Advanced visualizations
    """

    def __init__(self, config: Union[PhenoClusterConfig, str, Path]):
        """
        Initialize the pipeline.

        Parameters
        ----------
        config : PhenoClusterConfig or str or Path
            Configuration object or path to configuration file
        """
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            if config_path.suffix in [".yaml", ".yml"]:
                self.config = PhenoClusterConfig.from_yaml(config_path)
            elif config_path.suffix == ".json":
                self.config = PhenoClusterConfig.from_json(config_path)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        elif isinstance(config, PhenoClusterConfig):
            self.config = config
        else:
            raise TypeError("config must be PhenoClusterConfig, str, or Path")

        self.logger = get_logger("pipeline", self.config)

        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.data_splitter = DataSplitter(self.config.data_split)  # Mandatory for sound evaluation
        self.feature_selector = None  # Initialized during fit if enabled
        self.model_selector = None  # StepMixModelSelector
        self.evaluator = None
        self.visualizer = None

        # Results storage
        self.results = {}
        self.split_result = None

        # Cache (initialized in fit())
        self._cache: Optional[ArtifactCache] = None
        self._data_hash: Optional[str] = None

    def _resolve_reference_phenotype(self, labels: np.ndarray, data_processed: pd.DataFrame) -> int:
        """Determine which phenotype serves as the comparison reference.

        Parameters
        ----------
        labels : np.ndarray
            Cluster assignments (already reordered by size).
        data_processed : pd.DataFrame
            Processed data with outcome columns available.

        Returns
        -------
        int
            Phenotype ID to use as reference.
        """
        ref_cfg = self.config.reference_phenotype
        n_clusters = len(np.unique(labels))

        if ref_cfg.strategy == "largest":
            return 0  # After size-reordering, phenotype 0 is always largest

        elif ref_cfg.strategy == "healthiest":
            outcome_col = ref_cfg.health_outcome
            if not outcome_col or outcome_col not in data_processed.columns:
                self.logger.warning(
                    f"health_outcome '{outcome_col}' not found in data; "
                    f"falling back to strategy='largest'"
                )
                return 0
            event_rates = {}
            for k in range(n_clusters):
                mask = labels == k
                vals = data_processed.loc[mask, outcome_col].dropna()
                event_rates[k] = vals.mean() if len(vals) > 0 else 1.0
            ref = min(event_rates, key=event_rates.get)
            self.logger.info(
                f"Reference = Phenotype {ref} "
                f"(lowest event rate for '{outcome_col}': "
                f"{event_rates[ref]:.1%})"
            )
            return ref

        elif ref_cfg.strategy == "specific":
            if ref_cfg.specific_id is None or ref_cfg.specific_id not in range(n_clusters):
                self.logger.warning(
                    f"specific_id={ref_cfg.specific_id} is invalid for "
                    f"{n_clusters} clusters; falling back to strategy='largest'"
                )
                return 0
            return ref_cfg.specific_id

        else:
            self.logger.warning(
                f"Unknown reference strategy '{ref_cfg.strategy}'; falling back to 'largest'"
            )
            return 0

    def _build_measurement_dict(
        self,
        has_missing: bool = False,
        n_continuous: int | None = None,
        n_categorical: int | None = None,
    ) -> dict:
        """
        Build StepMix measurement specification from config.

        Parameters
        ----------
        has_missing : bool
            Whether the data has missing values. If True and imputation
            is disabled, uses '_nan' measurement types to handle NaN values.
        n_continuous : int, optional
            Number of continuous features. If None, uses
            len(config.continuous_columns).
        n_categorical : int, optional
            Number of categorical features. If None, uses
            len(config.categorical_columns).

        Returns
        -------
        dict
            Measurement specification for StepMix model
        """
        # Determine if we need NaN-aware measurement types
        # Use _nan suffix when imputation is disabled AND data has missing values
        use_nan_models = has_missing and not self.config.imputation.enabled

        n_cont = n_continuous if n_continuous is not None else len(self.config.continuous_columns)
        n_cat = n_categorical if n_categorical is not None else len(self.config.categorical_columns)

        measurement = {}
        if n_cont > 0:
            model_type = "continuous_nan" if use_nan_models else "continuous"
            measurement["continuous"] = {
                "model": model_type,
                "n_columns": n_cont,
            }
        if n_cat > 0:
            model_type = "categorical_nan" if use_nan_models else "categorical"
            measurement["categorical"] = {
                "model": model_type,
                "n_columns": n_cat,
            }
        return measurement

    def fit(self, data: pd.DataFrame, force_rerun: bool = False) -> Dict:
        """
        Execute the complete phenotype discovery pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Input clinical dataset
        force_rerun : bool
            If True, ignore cached artifacts and re-run all steps.
            New artifacts are still saved for future runs.

        Returns
        -------
        Dict
            Dictionary containing:
                - 'data': Processed dataset with phenotype labels
                - 'posterior_proba': Posterior probability matrix
                - 'model': Fitted model
                - 'model_selection': Model selection results (if enabled)
                - 'model_fit_metrics': Model fit metrics (log-likelihood, BIC, AIC, etc.)
                - 'cluster_stats': Descriptive statistics per cluster
                - 'outcome_results': Outcome association results
                - 'plots': Dictionary of generated visualizations
                - 'config': Configuration used
        """
        self.config.validate()

        self.logger.info(f"{self.config.project_name.upper()} - PHENOTYPE DISCOVERY PIPELINE")
        self.logger.info(f"Start time: {pd.Timestamp.now()}")
        self.logger.info(f"Configuration: {self.config.project_name}")

        # Initialize artifact cache
        self._init_cache(data, force_rerun)

        # Steps 0-6: Filter, quality, split, impute, outliers, preprocess
        ctx = self._preprocess(data)

        # Step 7: Feature selection (if enabled)
        self._select_features(ctx)

        # Step 8: Model selection and training (on train set)
        self._train_model(ctx)

        # Steps 9-10: Phase 1 (test evaluation) + Phase 2 (full-cohort refit)
        self._evaluate_model(ctx)

        # Step 9: Stability analysis (expensive, cached separately)
        self._run_stability(ctx)

        # Steps 10-10c: Outcomes, survival, multistate
        self._run_analyses(ctx)

        # Steps 11-13: Feature importance, visualizations, compile results
        self._finalize(ctx)

        return self.results

    def _init_cache(self, data: pd.DataFrame, force_rerun: bool) -> None:
        """Initialize the artifact cache if enabled."""
        if not self.config.cache.enabled or force_rerun:
            self._cache = None
            self._data_hash = None
            if force_rerun:
                self.logger.info("CACHE: force_rerun=True, all steps will re-run")
            elif not self.config.cache.enabled:
                self.logger.info("CACHE: disabled in config")
            return

        artifacts_dir = Path(self.config.output_dir) / "artifacts"
        self._data_hash = _compute_data_fingerprint(data)
        self._cache = ArtifactCache(
            artifacts_dir=artifacts_dir,
            config_dict=self.config.to_dict(),
            compress_level=self.config.cache.compress_level,
        )
        self.logger.info(f"CACHE: enabled (data fingerprint: {self._data_hash[:12]}...)")

    def _preprocess(self, data: pd.DataFrame) -> Dict:
        """Steps 0-6: Row filter, quality, missing detection, train/test split,
        imputation, outlier handling, and encoding+scaling.

        Phase 1 fits preprocessing on training data only (for model selection).
        Phase 2 (in _evaluate_model) refits on the full cohort for final analysis.

        Returns a mutable context dict carrying intermediate results.
        """
        if self._cache and self._cache.is_step_valid("preprocess", self._data_hash):
            self.logger.info("CACHE HIT: Loading preprocess from cache")
            cached = self._cache.load_step_artifacts("preprocess")
            self.preprocessor = cached["preprocessor"]
            self.split_result = cached["split_result"]
            return cached["ctx"]

        ctx = {}
        ctx["n_rows_original"] = len(data)

        # Step 0: Row-level missing data filter
        if self.config.row_filter.enabled and self.config.row_filter.max_missing_pct < 1.0:
            feature_cols = [
                c
                for c in self.config.continuous_columns + self.config.categorical_columns
                if c in data.columns
            ]
            if feature_cols:
                missing_frac = data[feature_cols].isna().mean(axis=1)
                data = data[missing_frac <= self.config.row_filter.max_missing_pct].reset_index(
                    drop=True
                )
                n_removed = ctx["n_rows_original"] - len(data)
                self.logger.info(
                    f"ROW FILTER: Removed {n_removed} of {ctx['n_rows_original']} rows "
                    f"({n_removed / ctx['n_rows_original'] * 100:.1f}%) with "
                    f">{self.config.row_filter.max_missing_pct * 100:.0f}% missing values. "
                    f"{len(data)} rows remaining."
                )
            else:
                self.logger.warning(
                    "ROW FILTER: No feature columns found in data, skipping filter."
                )

        ctx["data_filtered"] = data

        # Validate sufficient samples
        min_samples_needed = self.config.model_selection.min_clusters * 5
        if len(data) < min_samples_needed:
            raise ValueError(
                f"Insufficient samples after filtering: {len(data)} "
                f"remaining, but at least {min_samples_needed} needed "
                f"(min_clusters={self.config.model_selection.min_clusters} x 5). "
                f"Consider relaxing row_filter.max_missing_pct or reviewing data."
            )

        # Step 1: Data Quality Assessment
        ctx["quality_report"] = {}
        if self.config.data_quality.enabled:
            quality_assessor = DataQualityAssessor(self.config)
            ctx["quality_report"] = quality_assessor.assess_data_quality(data)
            quality_assessor.save_quality_report(Path(self.config.output_dir) / "quality")

        # Step 2: Detect missing values
        ctx["missing_info"] = self.preprocessor.detect_missing_values(data)

        # Step 3: Split into train/test BEFORE preprocessing
        self.logger.info("Splitting data into train/test sets...")
        self.split_result = self.data_splitter.split(
            data, stratify_column=self.config.data_split.stratify_by
        )
        ctx["split_info"] = {
            "train_size": self.split_result.n_train,
            "test_size": self.split_result.n_test,
            "train_fraction": self.split_result.train_fraction,
            "split_type": "random",
        }
        self.logger.info(
            f"  Train: {ctx['split_info']['train_size']} "
            f"({ctx['split_info']['train_fraction']:.1%})"
        )
        self.logger.info(
            f"  Test: {ctx['split_info']['test_size']} (held out for unbiased evaluation)"
        )

        data_train_raw = self.split_result.train
        data_test_raw = self.split_result.test

        # Step 4: Impute - fit on train, transform train and test
        self.preprocessor.fit_imputer(data_train_raw)
        data_train_imp = self.preprocessor.transform_impute(data_train_raw)
        data_test_imp = self.preprocessor.transform_impute(data_test_raw)

        # Step 5: Outlier handling - fit on train, transform train and test
        self.preprocessor.fit_outlier_handler(data_train_imp)
        data_train_out = self.preprocessor.transform_outliers(data_train_imp)
        data_test_out = self.preprocessor.transform_outliers(data_test_imp)

        # Step 6: Encode + scale - fit on train, transform train and test
        self.preprocessor.fit_preprocessor(data_train_out)
        data_train_proc, X_train = self.preprocessor.transform_preprocess(data_train_out)
        data_test_proc, X_test = self.preprocessor.transform_preprocess(data_test_out)

        ctx["X_train"] = X_train
        ctx["X_test"] = X_test
        ctx["data_train"] = data_train_proc
        ctx["data_test"] = data_test_proc

        # Save raw filtered data for Phase 2 refit on full cohort
        ctx["data_raw"] = data

        if self._cache:
            self._cache.save_step_artifacts(
                "preprocess",
                {"ctx": ctx, "preprocessor": self.preprocessor, "split_result": self.split_result},
                self._data_hash,
            )

        return ctx

    def _select_features(self, ctx: Dict) -> None:
        """Step 7: Feature selection (if enabled). Mutates ctx in place.

        Fits on training data only.  The selected feature subset is stored
        so that Phase 2 (full-cohort refit) can reuse the same columns.
        """
        if self._cache and self._cache.is_step_valid("feature_select", self._data_hash):
            self.logger.info("CACHE HIT: Loading feature_select from cache")
            cached = self._cache.load_step_artifacts("feature_select")
            self.feature_selector = cached.get("feature_selector")
            ctx["feature_selection_report"] = cached["feature_selection_report"]
            ctx["X_train"] = cached["X_train"]
            ctx["X_test"] = cached["X_test"]
            return

        ctx["feature_selection_report"] = {}
        if self.config.feature_selection.enabled:
            self.logger.info(
                f"Performing feature selection (method: {self.config.feature_selection.method})..."
            )
            self.feature_selector = MixedDataFeatureSelector(
                self.config.feature_selection,
                continuous_cols=self.config.continuous_columns,
                categorical_cols=self.config.categorical_columns,
            )

            # Extract target variable for supervised methods (lasso, mutual_info)
            y_fs = None
            if self.config.feature_selection.require_target:
                target_col = self.config.feature_selection.target_column
                if target_col not in ctx["data_train"].columns:
                    raise ValueError(
                        f"Feature selection target column '{target_col}' not found in data. "
                        f"Available outcome columns: {self.config.outcome_columns}"
                    )
                y_fs = ctx["data_train"][target_col].values

            # Filter to feature columns only - prevent outcome/survival columns
            # from influencing LASSO/mutual_info feature rankings
            feature_cols = self.config.continuous_columns + self.config.categorical_columns
            cols_present = [c for c in feature_cols if c in ctx["data_train"].columns]
            data_for_fs = ctx["data_train"][cols_present].copy()
            data_train_selected = self.feature_selector.fit_transform(data_for_fs, y=y_fs)
            ctx["feature_selection_report"] = self.feature_selector.get_selection_report()

            self.logger.info(
                f"  Selected {ctx['feature_selection_report']['n_selected']} of "
                f"{ctx['feature_selection_report']['n_original']} features"
            )
            self.logger.info(
                f"  Removed: {ctx['feature_selection_report']['removed_features'][:5]}"
                f"{'...' if len(ctx['feature_selection_report']['removed_features']) > 5 else ''}"
            )

            selected_features = self.feature_selector.get_selected_features()
            selected_continuous = [
                c for c in self.config.continuous_columns if c in selected_features
            ]
            selected_categorical = [
                c for c in self.config.categorical_columns if c in selected_features
            ]

            ctx["X_train"] = self.preprocessor.get_feature_matrix(
                data_train_selected, selected_continuous, selected_categorical
            )
            ctx["X_test"] = self.preprocessor.get_feature_matrix(
                ctx["data_test"], selected_continuous, selected_categorical
            )

        if self._cache:
            self._cache.save_step_artifacts(
                "feature_select",
                {
                    "feature_selector": self.feature_selector,
                    "feature_selection_report": ctx["feature_selection_report"],
                    "X_train": ctx["X_train"],
                    "X_test": ctx["X_test"],
                },
                self._data_hash,
            )

    def _train_model(self, ctx: Dict) -> None:
        """Step 6: Model selection and training. Stores model and selection_results in ctx."""
        if self._cache and self._cache.is_step_valid("train_model", self._data_hash):
            self.logger.info("CACHE HIT: Loading train_model from cache")
            cached = self._cache.load_step_artifacts("train_model")
            ctx["model"] = cached["model"]
            ctx["selection_results"] = cached["selection_results"]
            self.model_selector = cached.get("model_selector")
            return

        self.logger.info("Starting model fitting...")
        X_train = ctx["X_train"]

        has_missing_values = (
            np.isnan(X_train).any()
            if isinstance(X_train, np.ndarray)
            else X_train.isnull().any().any()
        )

        # Derive actual column counts from X_train shape.
        # This handles one-hot expansion, feature selection, and all encoding methods.
        n_total = X_train.shape[1]
        if self.feature_selector is not None:
            selected = self.feature_selector.get_selected_features()
            n_cont = sum(1 for c in self.config.continuous_columns if c in selected)
        else:
            n_cont = len(self.config.continuous_columns)
        n_cat = n_total - n_cont

        measurement = self._build_measurement_dict(
            has_missing=has_missing_values,
            n_continuous=n_cont,
            n_categorical=n_cat,
        )

        if has_missing_values and not self.config.imputation.enabled:
            self.logger.info("Using NaN-aware measurement models (continuous_nan, categorical_nan)")

        with _suppress_stepmix_warnings():
            if self.config.model_selection.enabled:
                self.logger.info(
                    "Using information criterion for model selection (on training data)"
                )
                self.model_selector = StepMixModelSelector(
                    config=self.config.model_selection,
                    measurement=measurement,
                    stepmix_params={
                        "random_state": self.config.random_state,
                        "max_iter": self.config.stepmix.max_iter,
                        "abs_tol": self.config.stepmix.abs_tol,
                        "rel_tol": self.config.stepmix.rel_tol,
                    },
                )
                model, _selection_result = self.model_selector.select_best_model(X_train)
                ctx["selection_results"] = self.model_selector.get_selection_results()

                comparison_table = self.model_selector.get_comparison_table()
                # Un-negate IC scores for display (scorers negate for sklearn maximization)
                _ic = {"BIC", "AIC", "CAIC", "SABIC", "ICL"}
                criterion_name = self.config.model_selection.criterion
                if criterion_name.upper() in _ic:
                    display_table = comparison_table.copy()
                    display_table["mean_score"] = -display_table["mean_score"]
                else:
                    display_table = comparison_table
                self.logger.info(f"Model comparison:\n{display_table.to_string()}")
            else:
                from stepmix.stepmix import StepMix

                self.logger.info(f"Using fixed number of clusters: {self.config.n_clusters}")
                model = StepMix(
                    n_components=self.config.n_clusters,
                    measurement=measurement,
                    random_state=self.config.random_state,
                    n_init=self.config.model_selection.n_init[0]
                    if self.config.model_selection.n_init
                    else 10,
                    verbose=0,
                    progress_bar=0,
                )
                model.fit(X_train)
                ctx["selection_results"] = {}

        ctx["model"] = model

        if self._cache:
            self._cache.save_step_artifacts(
                "train_model",
                {
                    "model": model,
                    "selection_results": ctx["selection_results"],
                    "model_selector": self.model_selector,
                },
                self._data_hash,
            )

    def _evaluate_model(self, ctx: Dict) -> None:
        """Phase 1: Test-set evaluation + validation metrics.
        Phase 2: Refit preprocessing and model on full cohort for final analysis."""
        if self._cache and self._cache.is_step_valid("evaluate_model", self._data_hash):
            self.logger.info("CACHE HIT: Loading evaluate_model from cache")
            cached = self._cache.load_step_artifacts("evaluate_model")
            ctx["labels"] = cached["labels"]
            ctx["labels_test"] = cached["labels_test"]
            ctx["proba"] = cached["proba"]
            ctx["proba_test"] = cached["proba_test"]
            ctx["cluster_stats"] = cached["cluster_stats"]
            ctx["model_fit_metrics"] = cached["model_fit_metrics"]
            ctx["test_metrics"] = cached["test_metrics"]
            ctx["n_clusters"] = cached["n_clusters"]
            ctx["validation_metrics"] = cached.get("validation_metrics", {})
            ctx["classification_quality"] = cached.get("classification_quality", {})
            ctx["classification_quality_test"] = cached.get("classification_quality_test", {})
            ctx["data_processed"] = cached["data_processed"]
            ctx["X"] = cached["X"]
            ctx["original_continuous_data"] = cached["original_continuous_data"]
            self.preprocessor = cached.get("full_preprocessor", self.preprocessor)
            self.reference_phenotype = cached["reference_phenotype"]
            self.evaluator = ClusterEvaluator(self.config, ctx["model"])
            return

        train_model = ctx["model"]
        X_train, X_test = ctx["X_train"], ctx["X_test"]
        n_clusters = train_model.n_components

        # Phase 1: Evaluate train-fitted model on held-out test set
        self.logger.info("Phase 1: Evaluating model on held-out test set...")
        labels_train = train_model.predict(X_train)
        labels_test = train_model.predict(X_test)
        proba_test = train_model.predict_proba(X_test)

        ctx["test_metrics"] = {
            "log_likelihood": train_model.score(X_test),
            "n_samples": len(X_test),
            "n_clusters": n_clusters,
        }
        self.logger.info(f"  Test set log-likelihood: {ctx['test_metrics']['log_likelihood']:.2f}")

        # Validation metrics: train vs test cluster proportions
        train_cluster_sizes = np.bincount(labels_train, minlength=n_clusters)
        test_cluster_sizes = np.bincount(labels_test, minlength=n_clusters)

        ctx["validation_metrics"] = {
            "train_cluster_sizes": {int(k): int(v) for k, v in enumerate(train_cluster_sizes)},
            "test_cluster_sizes": {int(k): int(v) for k, v in enumerate(test_cluster_sizes)},
            "train_cluster_pcts": {
                int(k): round(float(v / len(labels_train) * 100), 1)
                for k, v in enumerate(train_cluster_sizes)
            },
            "test_cluster_pcts": {
                int(k): round(float(v / len(labels_test) * 100), 1)
                for k, v in enumerate(test_cluster_sizes)
            },
            "train_log_likelihood": train_model.score(X_train),
            "test_log_likelihood": ctx["test_metrics"]["log_likelihood"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_clusters": n_clusters,
        }

        # Phase 2: Refit full preprocessing + model on full cohort
        self.logger.info("Phase 2: Refitting preprocessing and model on full cohort...")
        data_raw = ctx["data_raw"]

        # Refit preprocessing on full data
        full_preprocessor = DataPreprocessor(self.config)
        full_preprocessor.fit_imputer(data_raw)
        data_full_imp = full_preprocessor.transform_impute(data_raw)
        full_preprocessor.fit_outlier_handler(data_full_imp)
        data_full_out = full_preprocessor.transform_outliers(data_full_imp)

        # Save original continuous data before scaling
        ctx["original_continuous_data"] = (
            data_full_out[self.config.continuous_columns].copy()
            if self.config.continuous_columns
            else pd.DataFrame()
        )

        full_preprocessor.fit_preprocessor(data_full_out)
        data_processed, X = full_preprocessor.transform_preprocess(data_full_out)

        # If feature selection was active, subset X to selected features
        if self.feature_selector is not None:
            selected_features = self.feature_selector.get_selected_features()
            selected_continuous = [
                c for c in self.config.continuous_columns if c in selected_features
            ]
            selected_categorical = [
                c for c in self.config.categorical_columns if c in selected_features
            ]
            X = full_preprocessor.get_feature_matrix(
                data_processed, selected_continuous, selected_categorical
            )

        ctx["data_processed"] = data_processed
        ctx["X"] = X

        # Replace preprocessor with full-cohort-fitted version
        self.preprocessor = full_preprocessor

        # Refit model on full data with selected K
        self.logger.info(f"  Refitting StepMix (K={n_clusters}) on full cohort...")
        from stepmix.stepmix import StepMix

        has_missing = np.isnan(X).any() if isinstance(X, np.ndarray) else X.isnull().any().any()
        if self.feature_selector is not None:
            selected = self.feature_selector.get_selected_features()
            n_cont = sum(1 for c in self.config.continuous_columns if c in selected)
            n_cat = sum(1 for c in self.config.categorical_columns if c in selected)
            measurement = self._build_measurement_dict(
                has_missing=has_missing, n_continuous=n_cont, n_categorical=n_cat
            )
        else:
            measurement = self._build_measurement_dict(has_missing=has_missing)

        with _suppress_stepmix_warnings():
            model = StepMix(
                n_components=n_clusters,
                measurement=measurement,
                random_state=self.config.random_state,
                n_init=self.config.model_selection.n_init[0]
                if self.config.model_selection.n_init
                else 10,
                max_iter=self.config.stepmix.max_iter,
                abs_tol=self.config.stepmix.abs_tol,
                rel_tol=self.config.stepmix.rel_tol,
                verbose=0,
                progress_bar=0,
            )
            model.fit(X)
        ctx["model"] = model

        # Predictions on full cohort
        labels = model.predict(X)
        proba = model.predict_proba(X)

        # Reorder phenotypes by size (largest = 0)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        size_order = np.argsort(-cluster_sizes)
        if not np.array_equal(size_order, np.arange(n_clusters)):
            self.logger.info("Reordering phenotypes by size (largest = Phenotype 0)...")
            label_map = {old: new for new, old in enumerate(size_order)}
            labels = np.array([label_map[lbl] for lbl in labels])
            proba = proba[:, size_order]
            self.logger.info(f"  Phenotype sizes: {[int(cluster_sizes[i]) for i in size_order]}")

        # Derive test labels from full-cohort predictions (consistent label space)
        labels_test = labels[self.split_result.test_indices]
        proba_test = proba[self.split_result.test_indices]

        # Classification quality metrics from posterior probabilities
        classification_quality = _compute_classification_quality(proba, labels)
        ctx["classification_quality"] = classification_quality
        self.logger.info(
            "  Classification quality (full cohort): overall AvePP="
            f"{classification_quality['overall_avepp']:.3f}, "
            f"{classification_quality['assignment_confidence']['above_80']:.1f}% assigned "
            "with >80% confidence"
        )

        # Test-set classification quality
        classification_quality_test = _compute_classification_quality(proba_test, labels_test)
        ctx["classification_quality_test"] = classification_quality_test
        self.logger.info(
            "  Classification quality (test set): overall AvePP="
            f"{classification_quality_test['overall_avepp']:.3f}, "
            f"{classification_quality_test['assignment_confidence']['above_80']:.1f}% assigned "
            "with >80% confidence"
        )

        # Resolve reference phenotype
        self.reference_phenotype = self._resolve_reference_phenotype(labels, data_processed)
        self.logger.info(f"Reference phenotype: {self.reference_phenotype}")

        # Fit metrics
        self.evaluator = ClusterEvaluator(self.config, model)

        ctx["model_fit_metrics"] = {
            "train_log_likelihood": ctx["validation_metrics"]["train_log_likelihood"],
            "test_log_likelihood": ctx["validation_metrics"]["test_log_likelihood"],
            "full_log_likelihood": model.score(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_full": len(X),
            "n_clusters": n_clusters,
        }

        ctx["cluster_stats"] = self.evaluator.compute_cluster_statistics(
            data_processed, labels, original_df=ctx["original_continuous_data"]
        )

        ctx["labels"] = labels
        ctx["labels_test"] = labels_test
        ctx["proba"] = proba
        ctx["proba_test"] = proba_test
        ctx["n_clusters"] = n_clusters

        self.logger.info("Phase 2 complete: full-cohort model ready")

        if self._cache:
            self._cache.save_step_artifacts(
                "evaluate_model",
                {
                    "labels": labels,
                    "labels_test": labels_test,
                    "proba": proba,
                    "proba_test": proba_test,
                    "cluster_stats": ctx["cluster_stats"],
                    "model_fit_metrics": ctx["model_fit_metrics"],
                    "test_metrics": ctx["test_metrics"],
                    "n_clusters": n_clusters,
                    "reference_phenotype": self.reference_phenotype,
                    "validation_metrics": ctx["validation_metrics"],
                    "classification_quality": ctx["classification_quality"],
                    "classification_quality_test": ctx["classification_quality_test"],
                    "data_processed": data_processed,
                    "X": X,
                    "original_continuous_data": ctx["original_continuous_data"],
                    "full_preprocessor": self.preprocessor,
                },
                self._data_hash,
            )

    def _run_stability(self, ctx: Dict) -> None:
        """Step 9: Cluster stability analysis (expensive, cached separately)."""
        if self._cache and self._cache.is_step_valid("stability", self._data_hash):
            self.logger.info("CACHE HIT: Loading stability from cache")
            cached = self._cache.load_step_artifacts("stability")
            ctx["stability_results"] = cached["stability_results"]
            return

        labels = ctx["labels"]
        n_clusters = ctx["n_clusters"]

        ctx["stability_results"] = {}
        if self.config.stability.enabled:
            stability_analyzer = StabilityAnalyzer(self.config)
            ctx["stability_results"] = stability_analyzer.analyze_stability(
                ctx["X"], ctx["model"], labels
            )
            cluster_stability = stability_analyzer.analyze_cluster_stability(
                ctx["X"], ctx["model"], labels, n_clusters
            )
            ctx["stability_results"]["cluster_stability"] = cluster_stability

        if self._cache:
            self._cache.save_step_artifacts(
                "stability",
                {"stability_results": ctx["stability_results"]},
                self._data_hash,
            )

    def _run_analyses(self, ctx: Dict) -> None:
        """Steps 10-10c: Outcome, survival, and multistate analyses."""
        if self._cache and self._cache.is_step_valid("run_analyses", self._data_hash):
            self.logger.info("CACHE HIT: Loading run_analyses from cache")
            cached = self._cache.load_step_artifacts("run_analyses")
            ctx["outcome_results"] = cached["outcome_results"]
            ctx["survival_results"] = cached["survival_results"]
            ctx["multistate_results"] = cached["multistate_results"]
            return

        data_processed = ctx["data_processed"]
        labels = ctx["labels"]
        proba = ctx["proba"]
        n_clusters = ctx["n_clusters"]
        ref = self.reference_phenotype

        # Step 10: Outcome association analysis
        ctx["outcome_results"] = {}
        if self.config.outcome.enabled:
            data_train = data_processed.iloc[self.split_result.train_indices].reset_index(drop=True)
            labels_train = labels[self.split_result.train_indices]
            outcome_results = self.evaluator.analyze_outcomes(
                data_train, labels_train, reference_phenotype=ref
            )
            data_test = data_processed.iloc[self.split_result.test_indices].reset_index(drop=True)
            labels_test_for_outcomes = labels[self.split_result.test_indices]
            outcome_results_test = self.evaluator.analyze_outcomes(
                data_test, labels_test_for_outcomes, reference_phenotype=ref
            )
            ctx["outcome_results"] = {
                "train": outcome_results,
                "test": outcome_results_test,
                "full_cohort": self.evaluator.analyze_outcomes(
                    data_processed, labels, reference_phenotype=ref
                ),
            }
        else:
            self.logger.info("Outcome analysis disabled - skipping")

        # Step 10b: Survival analysis
        ctx["survival_results"] = {}
        if self.config.survival.enabled and self.config.survival.targets:
            self.logger.info("Running survival analysis (full cohort)...")
            from .evaluation.survival import SurvivalAnalyzer

            survival_analyzer = SurvivalAnalyzer(self.config, n_clusters, reference_phenotype=ref)

            for target in self.config.survival.targets:
                try:
                    result = survival_analyzer.analyze_survival(
                        data=data_processed,
                        labels=labels,
                        time_column=target.time_column,
                        event_column=target.event_column,
                    )
                    ctx["survival_results"][target.name] = result
                    self.logger.info(f"  {target.name} survival analysis completed")
                except Exception as e:
                    self.logger.warning(
                        f"  Survival analysis failed for {target.name} ({type(e).__name__}): {e}"
                    )
                    self.logger.warning(traceback.format_exc())

            # FDR correction across ALL survival pairwise p-values globally
            if self.config.inference.enabled and self.config.inference.fdr_correction:
                from .evaluation.stats_utils import apply_fdr_correction

                all_p_entries = []
                for target_name, target_data in ctx["survival_results"].items():
                    comparison = target_data.get("comparison", {})
                    for key, val in comparison.items():
                        if isinstance(val, dict) and "p_value" in val:
                            all_p_entries.append((target_name, key))
                if all_p_entries:
                    raw_p = [
                        ctx["survival_results"][t]["comparison"][k]["p_value"]
                        for t, k in all_p_entries
                    ]
                    adjusted = apply_fdr_correction(raw_p)
                    for (t, k), q in zip(all_p_entries, adjusted):
                        ctx["survival_results"][t]["comparison"][k]["p_value_fdr"] = q

            if self.config.survival.use_weighted:
                self.logger.info(
                    "  Running weighted survival analysis (posterior probability weights)..."
                )
                for target in self.config.survival.targets:
                    try:
                        weighted_result = survival_analyzer.analyze_weighted_survival(
                            data=data_processed,
                            posterior_probs=proba,
                            time_column=target.time_column,
                            event_column=target.event_column,
                        )
                        ctx["survival_results"][f"{target.name}_weighted"] = weighted_result
                    except Exception as e:
                        self.logger.warning(
                            f"  Weighted survival analysis failed for {target.name} "
                            f"({type(e).__name__}): {e}"
                        )
                        self.logger.warning(traceback.format_exc())

            if ctx["survival_results"]:
                self.logger.info(
                    f"  Completed survival analysis for {len(ctx['survival_results'])} endpoint(s)"
                )

        # Step 10c: Multistate analysis
        ctx["multistate_results"] = {}
        if self.config.multistate.enabled:
            self.logger.info("Running multistate analysis (full cohort)...")
            try:
                from .evaluation.multistate import MultistateAnalyzer

                multistate_analyzer = MultistateAnalyzer(
                    self.config, n_clusters, reference_phenotype=ref
                )

                ms_results = multistate_analyzer.run_full_analysis(
                    data_processed,
                    labels,
                )

                ctx["multistate_results"] = multistate_analyzer.results_to_dict(ms_results)

                n_trans = len(ms_results.transition_results)
                n_paths = len(ms_results.pathway_results)
                self.logger.info(f"  Fitted {n_trans} transition model(s)")
                self.logger.info(f"  Observed {n_paths} unique pathway(s)")
                if ms_results.state_occupation_probabilities:
                    mc_n = ms_results.state_occupation_probabilities.n_simulations
                    self.logger.info(f"  Monte Carlo: {mc_n:,} simulations per phenotype")
                self.logger.info("  Multistate analysis completed")
            except Exception as e:
                self.logger.warning(f"  Multistate analysis failed ({type(e).__name__}): {e}")
                self.logger.warning(traceback.format_exc())

        if self._cache:
            self._cache.save_step_artifacts(
                "run_analyses",
                {
                    "outcome_results": ctx["outcome_results"],
                    "survival_results": ctx["survival_results"],
                    "multistate_results": ctx["multistate_results"],
                },
                self._data_hash,
            )

    def _finalize(self, ctx: Dict) -> None:
        """Steps 11-13: Feature importance, visualizations, and result compilation."""
        data_processed = ctx["data_processed"]
        labels = ctx["labels"]
        n_clusters = ctx["n_clusters"]
        proba = ctx["proba"]

        # Step 11: Feature importance
        feature_importance = self.evaluator.compute_feature_importance(data_processed, labels)
        if feature_importance:
            feature_importance["top_features_per_cluster"] = (
                self.evaluator.get_top_features_per_cluster(
                    feature_importance,
                    feature_char_config=self.config.feature_characterization,
                )
            )

        # Step 11b: External validation (if configured)
        self._run_external_validation(ctx)

        # Step 12: Create visualizations
        self.visualizer = Visualizer(self.config, n_clusters)

        selection_results = (
            self.model_selector.get_selection_results() if self.model_selector is not None else None
        )
        plots = self.visualizer.create_all_plots(
            data_processed,
            labels,
            selection_results,
            ctx["stability_results"],
            outcome_results=ctx["outcome_results"],
            survival_results=ctx["survival_results"],
            multistate_results=ctx["multistate_results"],
            posterior_probs=proba,
            posterior_probs_test=ctx["proba_test"],
            labels_test=ctx["labels_test"],
            reference_phenotype=self.reference_phenotype,
        )

        # Step 13: Add cluster labels to data
        data_result = data_processed.copy()
        data_result["phenotype"] = labels
        for i in range(n_clusters):
            data_result[f"phenotype_prob_{i}"] = proba[:, i]

        # Compile results
        split_info = ctx["split_info"]
        feature_selection_report = ctx["feature_selection_report"]
        self.results = {
            "data": data_result,
            "posterior_proba": proba,
            "model": ctx["model"],
            "model_selection": ctx["selection_results"],
            "model_fit_metrics": ctx["model_fit_metrics"],
            "test_metrics": ctx["test_metrics"],
            "cluster_stats": ctx["cluster_stats"],
            "stability_results": ctx["stability_results"],
            "outcome_results": ctx["outcome_results"],
            "survival_results": ctx["survival_results"],
            "multistate_results": ctx["multistate_results"],
            "validation_metrics": ctx.get("validation_metrics", {}),
            "classification_quality": ctx.get("classification_quality", {}),
            "classification_quality_test": ctx.get("classification_quality_test", {}),
            "feature_importance": feature_importance,
            "external_validation_results": ctx.get("external_validation_results", {}),
            "quality_report": ctx["quality_report"],
            "split_info": split_info,
            "feature_selection": feature_selection_report,
            "plots": plots,
            "missing_info": ctx["missing_info"],
            "row_filter_info": {
                "enabled": self.config.row_filter.enabled,
                "max_missing_pct": self.config.row_filter.max_missing_pct,
                "n_rows_original": ctx["n_rows_original"],
                "n_rows_after_filter": len(ctx["data_filtered"]),
            },
            "config": self.config.to_dict(),
            "n_clusters": n_clusters,
            "n_samples": len(ctx["X"]),
            "reference_phenotype": self.reference_phenotype,
        }

        self.logger.info("PIPELINE COMPLETE")
        self.logger.info(f"End time: {pd.Timestamp.now()}")
        self.logger.info("Results include:")
        self.logger.info(f"  - Phenotype labels ({n_clusters} clusters)")
        self.logger.info("  - Posterior probability matrix")
        self.logger.info("  - Fitted model")
        if ctx["quality_report"]:
            self.logger.info("  - Data quality assessment")
        self.logger.info(
            f"  - Data split: train={split_info['train_size']}, test={split_info['test_size']}"
        )
        if feature_selection_report:
            self.logger.info(
                f"  - Feature selection: "
                f"{feature_selection_report['n_selected']}/"
                f"{feature_selection_report['n_original']} features"
            )
        if ctx["selection_results"]:
            self.logger.info("  - Model selection results")
        self.logger.info("  - Model fit metrics")
        self.logger.info("  - Cluster statistics")
        if ctx["stability_results"]:
            self.logger.info("  - Stability analysis results")
        if ctx["outcome_results"]:
            self.logger.info("  - Outcome association results")
        if ctx["survival_results"]:
            self.logger.info(
                f"  - Survival analysis results ({len(ctx['survival_results'])} endpoint(s))"
            )
        if ctx["multistate_results"]:
            n_trans = len(ctx["multistate_results"].get("transition_results", {}))
            n_paths = len(ctx["multistate_results"].get("pathway_results", []))
            self.logger.info(f"  - Multistate analysis ({n_trans} transitions, {n_paths} pathways)")
        self.logger.info("  - Feature importance analysis")
        if ctx.get("external_validation_results"):
            self.logger.info("  - External validation results")
        self.logger.info(f"  - Visualizations ({len(plots)} plots)")

    def _run_external_validation(self, ctx: Dict) -> None:
        """Run external validation on independent cohort if configured."""
        ev_config = self.config.external_validation
        if not ev_config.enabled or not ev_config.external_data_path:
            return

        path = Path(ev_config.external_data_path)
        if not path.exists():
            self.logger.warning(f"External data file not found: {path}")
            return

        self.logger.info(f"EXTERNAL VALIDATION: Loading {path}...")
        external_raw = pd.read_csv(path)

        # Preprocess using full-cohort-fitted preprocessor (same as Phase 2)
        try:
            ext_imp = self.preprocessor.transform_impute(external_raw)
            ext_out = self.preprocessor.transform_outliers(ext_imp)
            ext_processed, X_ext = self.preprocessor.transform_preprocess(ext_out)

            # If feature selection active, subset to selected features
            if self.feature_selector is not None:
                selected = self.feature_selector.get_selected_features()
                sel_cont = [c for c in self.config.continuous_columns if c in selected]
                sel_cat = [c for c in self.config.categorical_columns if c in selected]
                X_ext = self.preprocessor.get_feature_matrix(ext_processed, sel_cont, sel_cat)
        except Exception as e:
            self.logger.warning(
                f"External validation preprocessing failed: {e}\n{traceback.format_exc()}"
            )
            return

        # Run validation
        from .evaluation.external_validation import ExternalValidator

        validator = ExternalValidator(self.config, ctx["n_clusters"])
        results = validator.validate_with_model(
            X_external=X_ext,
            model=ctx["model"],
            derivation_labels=ctx["labels"],
            derivation_outcomes=ctx.get("outcome_results"),
            n_external=len(external_raw),
            external_df=ext_processed,
        )

        # Recover external labels for downstream analyses
        external_labels = np.array(results["external_labels"])
        ref = self.reference_phenotype

        # Classification quality for external cohort
        external_proba = ctx["model"].predict_proba(X_ext)
        ext_cq = _compute_classification_quality(external_proba, external_labels)
        results["classification_quality"] = ext_cq
        self.logger.info(f"  External classification quality: AvePP={ext_cq['overall_avepp']:.3f}")

        # External survival analysis (if enabled and columns present)
        ext_survival_results = {}
        if self.config.survival.enabled and self.config.survival.targets:
            self.logger.info("Running survival analysis on external cohort...")
            from .evaluation.survival import SurvivalAnalyzer

            ext_surv_analyzer = SurvivalAnalyzer(
                self.config, ctx["n_clusters"], reference_phenotype=ref
            )
            for target in self.config.survival.targets:
                if (
                    target.time_column in ext_processed.columns
                    and target.event_column in ext_processed.columns
                ):
                    try:
                        ext_surv = ext_surv_analyzer.analyze_survival(
                            data=ext_processed,
                            labels=external_labels,
                            time_column=target.time_column,
                            event_column=target.event_column,
                        )
                        ext_survival_results[target.name] = ext_surv
                        self.logger.info(f"  External survival: {target.name} OK")
                    except Exception as e:
                        self.logger.warning(f"External survival failed for {target.name}: {e}")
                else:
                    self.logger.warning(
                        f"External data missing columns for survival target '{target.name}'"
                    )
            if ext_survival_results:
                results["survival_results"] = ext_survival_results

        # External multistate analysis (if enabled)
        if self.config.multistate.enabled:
            self.logger.info("Running multistate analysis on external cohort...")
            try:
                from .evaluation.multistate import MultistateAnalyzer

                ext_ms_analyzer = MultistateAnalyzer(
                    self.config, ctx["n_clusters"], reference_phenotype=ref
                )
                ext_ms_results = ext_ms_analyzer.run_full_analysis(ext_processed, external_labels)
                results["multistate_results"] = ext_ms_analyzer.results_to_dict(ext_ms_results)
                self.logger.info("  External multistate analysis OK")
            except Exception as e:
                self.logger.warning(
                    f"External multistate analysis failed: {e}\n{traceback.format_exc()}"
                )

        # Create external plots
        ext_plots = {}
        ms_res = results.get("multistate_results")
        if ext_survival_results or ms_res:
            from .visualization.plots import Visualizer

            ext_visualizer = Visualizer(self.config, ctx["n_clusters"])

            for target_name, surv_result in ext_survival_results.items():
                try:
                    km = ext_visualizer.create_kaplan_meier_plot(
                        surv_result, target_name=target_name
                    )
                    if km:
                        ext_plots[f"ext_kaplan_meier_{target_name}"] = km
                    na = ext_visualizer.create_nelson_aalen_plot(
                        surv_result, target_name=target_name
                    )
                    if na:
                        ext_plots[f"ext_nelson_aalen_{target_name}"] = na
                except Exception as e:
                    self.logger.warning(f"External survival plot failed for {target_name}: {e}")

        if ms_res:
            trans_res = ms_res.get("transition_results", {})
            if trans_res:
                try:
                    fig = ext_visualizer.create_transition_hazard_forest_plot(
                        trans_res, reference_phenotype=ref
                    )
                    if fig:
                        ext_plots["ext_multistate_transition_hazards"] = fig
                except Exception as e:
                    self.logger.warning(f"External multistate HR plot failed: {e}")
            state_occ = ms_res.get("state_occupation_probabilities")
            if state_occ:
                try:
                    fig = ext_visualizer.create_state_occupation_uncertainty_plot(state_occ)
                    if fig:
                        ext_plots["ext_multistate_state_occupation_uncertainty"] = fig
                except Exception as e:
                    self.logger.warning(f"External state occupation plot failed: {e}")

        if ext_plots:
            results["plots"] = ext_plots

        ctx["external_validation_results"] = results

    def save_results(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Save pipeline results to disk."""
        save_pipeline_results(
            self.results,
            self.config,
            self.preprocessor,
            self.feature_selector,
            self.reference_phenotype,
            self.logger,
            output_dir,
        )

        self.logger.info("All results saved successfully")


def run_pipeline(
    data: pd.DataFrame,
    config: Union[PhenoClusterConfig, str, Path],
    force_rerun: bool = False,
) -> Dict:
    """
    Convenience function to run the complete pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Input clinical dataset
    config : PhenoClusterConfig or str or Path
        Configuration object or path to configuration file
    force_rerun : bool
        If True, ignore cached artifacts and re-run all steps.

    Returns
    -------
    Dict
        Pipeline results

    Examples
    --------
    >>> import pandas as pd
    >>> from phenocluster import run_pipeline
    >>>
    >>> # Load data
    >>> df = pd.read_csv('clinical_data.csv')
    >>>
    >>> # Run pipeline with config file
    >>> results = run_pipeline(df, 'config.yaml')
    >>>
    >>> # Save results
    >>> from phenocluster import PhenoClusterPipeline
    >>> pipeline = PhenoClusterPipeline('config.yaml')
    >>> results = pipeline.fit(df)
    >>> pipeline.save_results()
    """
    pipeline = PhenoClusterPipeline(config)
    results = pipeline.fit(data, force_rerun=force_rerun)
    pipeline.save_results()
    return results
