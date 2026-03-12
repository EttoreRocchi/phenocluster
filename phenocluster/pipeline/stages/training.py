"""Model training stage."""

import numpy as np

from ...model_selection import StepMixModelSelector
from ..context import PipelineContext
from ..warnings import suppress_stepmix_warnings


class TrainingStage:
    """Step 8: Model selection and training on training data."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model_selector = None

    def run(self, ctx: PipelineContext, feature_selector=None) -> None:
        """Execute model training. Mutates ctx in place."""
        self.logger.info("Starting model fitting...")
        X_train = ctx.X_train

        has_missing_values = (
            np.isnan(X_train).any()
            if isinstance(X_train, np.ndarray)
            else X_train.isnull().any().any()
        )

        n_cont, n_cat = self._count_feature_types(X_train, feature_selector)
        measurement = self._build_measurement_dict(has_missing_values, n_cont, n_cat)

        if has_missing_values and not self.config.imputation.enabled:
            self.logger.info("Using NaN-aware measurement models (continuous_nan, categorical_nan)")

        with suppress_stepmix_warnings():
            if self.config.model_selection.enabled:
                ctx.model = self._train_with_selection(ctx, X_train, measurement)
            else:
                ctx.model = self._train_fixed(X_train, measurement)

    def _count_feature_types(self, X_train, feature_selector):
        """Derive actual column counts from X_train shape."""
        n_total = X_train.shape[1]
        if feature_selector is not None:
            selected = feature_selector.get_selected_features()
            n_cont = sum(1 for c in self.config.continuous_columns if c in selected)
        else:
            n_cont = len(self.config.continuous_columns)
        n_cat = n_total - n_cont
        return n_cont, n_cat

    def _build_measurement_dict(self, has_missing, n_continuous, n_categorical):
        """Build StepMix measurement specification from config."""
        use_nan_models = has_missing and not self.config.imputation.enabled

        measurement = {}
        if n_continuous > 0:
            model_type = "continuous_nan" if use_nan_models else "continuous"
            measurement["continuous"] = {"model": model_type, "n_columns": n_continuous}
        if n_categorical > 0:
            model_type = "categorical_nan" if use_nan_models else "categorical"
            measurement["categorical"] = {"model": model_type, "n_columns": n_categorical}
        return measurement

    def _train_with_selection(self, ctx, X_train, measurement):
        """Train with automatic model selection via information criteria."""
        self.logger.info("Using information criterion for model selection (on training data)")
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
        model, _ = self.model_selector.select_best_model(X_train)
        ctx.selection_results = self.model_selector.get_selection_results()

        comparison_table = self.model_selector.get_comparison_table()
        _ic = {"BIC", "AIC", "CAIC", "SABIC", "ICL"}
        criterion_name = self.config.model_selection.criterion
        if criterion_name.upper() in _ic:
            display_table = comparison_table.copy()
            display_table["mean_score"] = -display_table["mean_score"]
        else:
            display_table = comparison_table
        self.logger.info(f"Model comparison:\n{display_table.to_string()}")
        return model

    def _train_fixed(self, X_train, measurement):
        """Train with fixed number of clusters."""
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
        return model
