"""
PhenoCluster Pipeline Orchestrator
====================================

Thin orchestrator that coordinates pipeline stages and manages caching.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from ..cache import ArtifactCache, _compute_data_fingerprint
from ..config import PhenoClusterConfig
from ..data import DataPreprocessor, DataSplitter
from ..evaluation import StabilityAnalyzer
from ..utils.logging import get_logger
from ..utils.results_io import save_pipeline_results
from .context import PipelineContext
from .stages import (
    AnalysisStage,
    EvaluationStage,
    FeatureSelectionStage,
    FinalizationStage,
    PreprocessingStage,
    TrainingStage,
)


class PhenoClusterPipeline:
    """
    Main pipeline for clinical phenotype discovery using latent class analysis.

    This class orchestrates the complete workflow by delegating to
    focused stage classes:

    - PreprocessingStage: data filtering, quality, splitting, preprocessing
    - FeatureSelectionStage: optional feature selection
    - TrainingStage: model selection and training
    - EvaluationStage: test evaluation, full-cohort refit, classification quality
    - AnalysisStage: outcome, survival, multistate analyses
    - FinalizationStage: feature importance, visualization, result compilation
    """

    def __init__(self, config: Union[PhenoClusterConfig, str, Path]):
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
        self.data_splitter = DataSplitter(self.config.data_split)
        self.feature_selector = None
        self.model_selector = None
        self.reference_phenotype = 0

        # Results storage
        self.results = {}
        self.split_result = None

        # Cache
        self._cache: Optional[ArtifactCache] = None
        self._data_hash: Optional[str] = None

    def fit(self, data: pd.DataFrame, force_rerun: bool = False) -> Dict:
        """
        Execute the complete phenotype discovery pipeline.

        Parameters
        ----------
        data : pd.DataFrame
            Input clinical dataset
        force_rerun : bool
            If True, ignore cached artifacts and re-run all steps.

        Returns
        -------
        Dict
            Dictionary containing all pipeline results.
        """
        self.config.validate()
        self.logger.info(f"{self.config.project_name.upper()} - PHENOTYPE DISCOVERY PIPELINE")
        self.logger.info(f"Start time: {pd.Timestamp.now()}")
        self.logger.info(f"Configuration: {self.config.project_name}")

        # Initialize artifact cache
        self._init_cache(data, force_rerun)

        ctx = PipelineContext(data_raw=data)

        # Steps 0-6: Preprocessing
        self._run_preprocessing(ctx)

        # Step 7: Feature selection
        self._run_feature_selection(ctx)

        # Step 8: Model training
        self._run_training(ctx)

        # Steps 9-10: Evaluation (test + full-cohort refit)
        self._run_evaluation(ctx)

        # Step 9: Stability analysis
        self._run_stability(ctx)

        # Steps 10-10c: Analyses
        self._run_analyses(ctx)

        # Steps 11-13: Finalization
        self._run_finalization(ctx)

        return self.results

    def _init_cache(self, data: pd.DataFrame, force_rerun: bool) -> None:
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

    def _run_preprocessing(self, ctx: PipelineContext) -> None:
        if self._cache and self._cache.is_step_valid("preprocess", self._data_hash):
            self.logger.info("CACHE HIT: Loading preprocess from cache")
            cached = self._cache.load_step_artifacts("preprocess")
            self.preprocessor = cached["preprocessor"]
            self.split_result = cached["split_result"]
            self._restore_ctx_from_cached(ctx, cached["ctx_data"])
            return

        stage = PreprocessingStage(self.config, self.preprocessor, self.data_splitter, self.logger)
        stage.run(ctx)
        self.preprocessor = stage.preprocessor
        self.split_result = stage.split_result

        if self._cache:
            self._cache.save_step_artifacts(
                "preprocess",
                {
                    "ctx_data": self._extract_ctx_data(ctx, "preprocess"),
                    "preprocessor": self.preprocessor,
                    "split_result": self.split_result,
                },
                self._data_hash,
            )

    def _run_feature_selection(self, ctx: PipelineContext) -> None:
        if self._cache and self._cache.is_step_valid("feature_select", self._data_hash):
            self.logger.info("CACHE HIT: Loading feature_select from cache")
            cached = self._cache.load_step_artifacts("feature_select")
            self.feature_selector = cached.get("feature_selector")
            ctx.feature_selection_report = cached["feature_selection_report"]
            ctx.X_train = cached["X_train"]
            ctx.X_test = cached["X_test"]
            return

        stage = FeatureSelectionStage(self.config, self.preprocessor, self.logger)
        stage.run(ctx)
        self.feature_selector = stage.feature_selector

        if self._cache:
            self._cache.save_step_artifacts(
                "feature_select",
                {
                    "feature_selector": self.feature_selector,
                    "feature_selection_report": ctx.feature_selection_report,
                    "X_train": ctx.X_train,
                    "X_test": ctx.X_test,
                },
                self._data_hash,
            )

    def _run_training(self, ctx: PipelineContext) -> None:
        if self._cache and self._cache.is_step_valid("train_model", self._data_hash):
            self.logger.info("CACHE HIT: Loading train_model from cache")
            cached = self._cache.load_step_artifacts("train_model")
            ctx.model = cached["model"]
            ctx.selection_results = cached["selection_results"]
            self.model_selector = cached.get("model_selector")
            return

        stage = TrainingStage(self.config, self.logger)
        stage.run(ctx, feature_selector=self.feature_selector)
        self.model_selector = stage.model_selector

        if self._cache:
            self._cache.save_step_artifacts(
                "train_model",
                {
                    "model": ctx.model,
                    "selection_results": ctx.selection_results,
                    "model_selector": self.model_selector,
                },
                self._data_hash,
            )

    def _run_evaluation(self, ctx: PipelineContext) -> None:
        if self._cache and self._cache.is_step_valid("evaluate_model", self._data_hash):
            self.logger.info("CACHE HIT: Loading evaluate_model from cache")
            cached = self._cache.load_step_artifacts("evaluate_model")
            ctx.labels = cached["labels"]
            ctx.labels_test = cached["labels_test"]
            ctx.proba = cached["proba"]
            ctx.proba_test = cached["proba_test"]
            ctx.cluster_stats = cached["cluster_stats"]
            ctx.model_fit_metrics = cached["model_fit_metrics"]
            ctx.test_metrics = cached["test_metrics"]
            ctx.n_clusters = cached["n_clusters"]
            ctx.validation_metrics = cached.get("validation_metrics", {})
            ctx.classification_quality = cached.get("classification_quality", {})
            ctx.classification_quality_test = cached.get("classification_quality_test", {})
            ctx.data_processed = cached["data_processed"]
            ctx.X = cached["X"]
            ctx.original_continuous_data = cached["original_continuous_data"]
            self.preprocessor = cached.get("full_preprocessor", self.preprocessor)
            self.reference_phenotype = cached["reference_phenotype"]
            from ..evaluation import ClusterEvaluator

            self._evaluator = ClusterEvaluator(self.config, ctx.model)
            return

        stage = EvaluationStage(self.config, self.logger)
        stage.run(ctx, self.split_result, feature_selector=self.feature_selector)
        self.preprocessor = stage.preprocessor
        self.reference_phenotype = stage.reference_phenotype
        self._evaluator = stage.evaluator

        if self._cache:
            self._cache.save_step_artifacts(
                "evaluate_model",
                {
                    "labels": ctx.labels,
                    "labels_test": ctx.labels_test,
                    "proba": ctx.proba,
                    "proba_test": ctx.proba_test,
                    "cluster_stats": ctx.cluster_stats,
                    "model_fit_metrics": ctx.model_fit_metrics,
                    "test_metrics": ctx.test_metrics,
                    "n_clusters": ctx.n_clusters,
                    "reference_phenotype": self.reference_phenotype,
                    "validation_metrics": ctx.validation_metrics,
                    "classification_quality": ctx.classification_quality,
                    "classification_quality_test": ctx.classification_quality_test,
                    "data_processed": ctx.data_processed,
                    "X": ctx.X,
                    "original_continuous_data": ctx.original_continuous_data,
                    "full_preprocessor": self.preprocessor,
                },
                self._data_hash,
            )

    def _run_stability(self, ctx: PipelineContext) -> None:
        if self._cache and self._cache.is_step_valid("stability", self._data_hash):
            self.logger.info("CACHE HIT: Loading stability from cache")
            cached = self._cache.load_step_artifacts("stability")
            ctx.stability_results = cached["stability_results"]
            return

        if self.config.stability.enabled:
            stability_analyzer = StabilityAnalyzer(self.config)
            ctx.stability_results = stability_analyzer.analyze_stability(
                ctx.X, ctx.model, ctx.labels
            )
            cluster_stability = stability_analyzer.analyze_cluster_stability(
                ctx.X, ctx.model, ctx.labels, ctx.n_clusters
            )
            ctx.stability_results["cluster_stability"] = cluster_stability

        if self._cache:
            self._cache.save_step_artifacts(
                "stability",
                {"stability_results": ctx.stability_results},
                self._data_hash,
            )

    def _run_analyses(self, ctx: PipelineContext) -> None:
        if self._cache and self._cache.is_step_valid("run_analyses", self._data_hash):
            self.logger.info("CACHE HIT: Loading run_analyses from cache")
            cached = self._cache.load_step_artifacts("run_analyses")
            ctx.outcome_results = cached["outcome_results"]
            ctx.survival_results = cached["survival_results"]
            ctx.multistate_results = cached["multistate_results"]
            return

        stage = AnalysisStage(self.config, self.logger)
        stage.run(ctx, self._evaluator, self.split_result, self.reference_phenotype)

        if self._cache:
            self._cache.save_step_artifacts(
                "run_analyses",
                {
                    "outcome_results": ctx.outcome_results,
                    "survival_results": ctx.survival_results,
                    "multistate_results": ctx.multistate_results,
                },
                self._data_hash,
            )

    def _run_finalization(self, ctx: PipelineContext) -> None:
        stage = FinalizationStage(self.config, self.logger)
        self.results = stage.run(
            ctx,
            evaluator=self._evaluator,
            model_selector=self.model_selector,
            preprocessor=self.preprocessor,
            feature_selector=self.feature_selector,
            split_result=self.split_result,
            reference_phenotype=self.reference_phenotype,
        )

    def _extract_ctx_data(self, ctx, stage):
        """Extract ctx fields relevant to a stage for caching."""
        if stage == "preprocess":
            return {
                "n_rows_original": ctx.n_rows_original,
                "data_filtered": ctx.data_filtered,
                "quality_report": ctx.quality_report,
                "missing_info": ctx.missing_info,
                "split_info": ctx.split_info,
                "X_train": ctx.X_train,
                "X_test": ctx.X_test,
                "data_train": ctx.data_train,
                "data_test": ctx.data_test,
                "data_raw": ctx.data_raw,
            }
        return {}

    def _restore_ctx_from_cached(self, ctx, cached_data):
        """Restore ctx fields from cached data."""
        for key, value in cached_data.items():
            setattr(ctx, key, value)

    def save_results(self, output_dir=None) -> None:
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
        Dictionary containing all pipeline results.
    """
    pipeline = PhenoClusterPipeline(config)
    return pipeline.fit(data, force_rerun=force_rerun)
