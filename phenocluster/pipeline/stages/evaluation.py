"""Evaluation stage: test-set evaluation, full-cohort refit, phenotype reordering."""

import numpy as np
import pandas as pd

from ...data import DataPreprocessor
from ...evaluation import ClusterEvaluator
from ..context import PipelineContext
from ..quality import compute_classification_quality
from ..warnings import suppress_stepmix_warnings


class EvaluationStage:
    """Steps 9-10: Test-set evaluation + full-cohort refit + classification quality."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.evaluator = None
        self.reference_phenotype = 0
        self.preprocessor = None  # Updated after full-cohort refit

    def run(self, ctx: PipelineContext, split_result, feature_selector=None) -> None:
        """Execute evaluation. Mutates ctx in place."""
        n_clusters = ctx.model.n_components

        self._evaluate_on_test(ctx, n_clusters)
        self._refit_full_cohort(ctx, n_clusters, split_result, feature_selector)
        self._reorder_and_classify(ctx, n_clusters, split_result, feature_selector)

        self.logger.info("Phase 2 complete: full-cohort model ready")

    def _evaluate_on_test(self, ctx: PipelineContext, n_clusters: int) -> None:
        """Phase 1: Evaluate train-fitted model on held-out test set."""
        train_model = ctx.model
        X_train, X_test = ctx.X_train, ctx.X_test

        self.logger.info("Phase 1: Evaluating model on held-out test set...")
        labels_train = train_model.predict(X_train)
        labels_test = train_model.predict(X_test)

        ctx.test_metrics = {
            "log_likelihood": train_model.score(X_test),
            "n_samples": len(X_test),
            "n_clusters": n_clusters,
        }
        self.logger.info(f"  Test set log-likelihood: {ctx.test_metrics['log_likelihood']:.2f}")

        train_cluster_sizes = np.bincount(labels_train, minlength=n_clusters)
        test_cluster_sizes = np.bincount(labels_test, minlength=n_clusters)

        ctx.validation_metrics = {
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
            "test_log_likelihood": ctx.test_metrics["log_likelihood"],
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_clusters": n_clusters,
        }

    def _refit_full_cohort(self, ctx, n_clusters, split_result, feature_selector):
        """Phase 2: Refit preprocessing and model on full cohort."""
        self.logger.info("Phase 2: Refitting preprocessing and model on full cohort...")
        data_raw = ctx.data_raw

        full_preprocessor = DataPreprocessor(self.config)
        full_preprocessor.fit_imputer(data_raw)
        data_full_imp = full_preprocessor.transform_impute(data_raw)
        full_preprocessor.fit_outlier_handler(data_full_imp)
        data_full_out = full_preprocessor.transform_outliers(data_full_imp)

        ctx.original_continuous_data = (
            data_full_out[self.config.continuous_columns].copy()
            if self.config.continuous_columns
            else pd.DataFrame()
        )

        full_preprocessor.fit_preprocessor(data_full_out)
        data_processed, X = full_preprocessor.transform_preprocess(data_full_out)

        if feature_selector is not None:
            selected_features = feature_selector.get_selected_features()
            selected_continuous = [
                c for c in self.config.continuous_columns if c in selected_features
            ]
            selected_categorical = [
                c for c in self.config.categorical_columns if c in selected_features
            ]
            X = full_preprocessor.get_feature_matrix(
                data_processed, selected_continuous, selected_categorical
            )

        ctx.data_processed = data_processed
        ctx.X = X
        self.preprocessor = full_preprocessor

        self.logger.info(f"  Refitting StepMix (K={n_clusters}) on full cohort...")
        from stepmix.stepmix import StepMix

        has_missing = np.isnan(X).any() if isinstance(X, np.ndarray) else X.isnull().any().any()

        # Build measurement dict
        if feature_selector is not None:
            selected = feature_selector.get_selected_features()
            n_cont = sum(1 for c in self.config.continuous_columns if c in selected)
            n_cat = sum(1 for c in self.config.categorical_columns if c in selected)
        else:
            n_cont = len(self.config.continuous_columns)
            n_total = X.shape[1]
            n_cat = n_total - n_cont

        use_nan = has_missing and not self.config.imputation.enabled
        measurement = {}
        if n_cont > 0:
            measurement["continuous"] = {
                "model": "continuous_nan" if use_nan else "continuous",
                "n_columns": n_cont,
            }
        if n_cat > 0:
            measurement["categorical"] = {
                "model": "categorical_nan" if use_nan else "categorical",
                "n_columns": n_cat,
            }

        with suppress_stepmix_warnings():
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
        ctx.model = model

    def _reorder_and_classify(self, ctx, n_clusters, split_result, feature_selector):
        """Reorder phenotypes by size, compute classification quality, resolve reference."""
        model = ctx.model
        X = ctx.X
        data_processed = ctx.data_processed

        labels = model.predict(X)
        proba = model.predict_proba(X)

        # Reorder phenotypes by size (largest = 0)
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        size_order = np.argsort(-cluster_sizes)
        if not np.array_equal(size_order, np.arange(n_clusters)):
            self.logger.info("Reordering phenotypes by size (largest = Phenotype 0)...")
            mapping_arr = np.empty(n_clusters, dtype=int)
            for new, old in enumerate(size_order):
                mapping_arr[old] = new
            labels = mapping_arr[labels]
            proba = proba[:, size_order]
            self.logger.info(f"  Phenotype sizes: {[int(cluster_sizes[i]) for i in size_order]}")

        # Derive test labels from full-cohort predictions
        labels_test = labels[split_result.test_indices]
        proba_test = proba[split_result.test_indices]

        # Classification quality metrics
        classification_quality = compute_classification_quality(proba, labels)
        ctx.classification_quality = classification_quality
        self.logger.info(
            "  Classification quality (full cohort): overall AvePP="
            f"{classification_quality['overall_avepp']:.3f}, "
            f"{classification_quality['assignment_confidence']['above_80']:.1f}% assigned "
            "with >80% confidence"
        )

        classification_quality_test = compute_classification_quality(proba_test, labels_test)
        ctx.classification_quality_test = classification_quality_test
        self.logger.info(
            "  Classification quality (test set): overall AvePP="
            f"{classification_quality_test['overall_avepp']:.3f}, "
            f"{classification_quality_test['assignment_confidence']['above_80']:.1f}% assigned "
            "with >80% confidence"
        )

        # Resolve reference phenotype
        self.reference_phenotype = self._resolve_reference_phenotype(
            labels, data_processed, n_clusters
        )
        self.logger.info(f"Reference phenotype: {self.reference_phenotype}")

        self.evaluator = ClusterEvaluator(self.config, model)

        ctx.model_fit_metrics = {
            "train_log_likelihood": ctx.validation_metrics["train_log_likelihood"],
            "test_log_likelihood": ctx.validation_metrics["test_log_likelihood"],
            "full_log_likelihood": model.score(X),
            "n_train": len(ctx.X_train),
            "n_test": len(ctx.X_test),
            "n_full": len(X),
            "n_clusters": n_clusters,
        }

        ctx.cluster_stats = self.evaluator.compute_cluster_statistics(
            data_processed, labels, original_df=ctx.original_continuous_data
        )

        ctx.labels = labels
        ctx.labels_test = labels_test
        ctx.proba = proba
        ctx.proba_test = proba_test
        ctx.n_clusters = n_clusters

    def _resolve_reference_phenotype(self, labels, data_processed, n_clusters):
        """Determine which phenotype serves as the comparison reference."""
        ref_cfg = self.config.reference_phenotype

        if ref_cfg.strategy == "largest":
            return 0

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
