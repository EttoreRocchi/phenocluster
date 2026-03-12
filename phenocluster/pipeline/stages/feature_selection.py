"""Feature selection stage."""

from ...feature_selection import MixedDataFeatureSelector
from ..context import PipelineContext


class FeatureSelectionStage:
    """Step 7: Feature selection (if enabled). Fits on training data only."""

    def __init__(self, config, preprocessor, logger):
        self.config = config
        self.preprocessor = preprocessor
        self.logger = logger
        self.feature_selector = None

    def run(self, ctx: PipelineContext) -> None:
        """Execute feature selection. Mutates ctx in place."""
        if not self.config.feature_selection.enabled:
            return

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
            if target_col not in ctx.data_train.columns:
                raise ValueError(
                    f"Feature selection target column '{target_col}' not found in data. "
                    f"Available outcome columns: {self.config.outcome_columns}"
                )
            y_fs = ctx.data_train[target_col].values

        # Filter to feature columns only
        feature_cols = self.config.continuous_columns + self.config.categorical_columns
        cols_present = [c for c in feature_cols if c in ctx.data_train.columns]
        data_for_fs = ctx.data_train[cols_present].copy()
        data_train_selected = self.feature_selector.fit_transform(data_for_fs, y=y_fs)
        ctx.feature_selection_report = self.feature_selector.get_selection_report()

        self.logger.info(
            f"  Selected {ctx.feature_selection_report['n_selected']} of "
            f"{ctx.feature_selection_report['n_original']} features"
        )
        self.logger.info(
            f"  Removed: {ctx.feature_selection_report['removed_features'][:5]}"
            f"{'...' if len(ctx.feature_selection_report['removed_features']) > 5 else ''}"
        )

        selected_features = self.feature_selector.get_selected_features()
        selected_continuous = [c for c in self.config.continuous_columns if c in selected_features]
        selected_categorical = [
            c for c in self.config.categorical_columns if c in selected_features
        ]

        ctx.X_train = self.preprocessor.get_feature_matrix(
            data_train_selected, selected_continuous, selected_categorical
        )
        ctx.X_test = self.preprocessor.get_feature_matrix(
            ctx.data_test, selected_continuous, selected_categorical
        )
