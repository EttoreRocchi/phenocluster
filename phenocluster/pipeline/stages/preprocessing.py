"""Preprocessing stage: row filter, quality, split, impute, outlier, encode/scale."""

from pathlib import Path

from ...evaluation import DataQualityAssessor
from ..context import PipelineContext


class PreprocessingStage:
    """Steps 0-6: Row filter, quality assessment, train/test split,
    imputation, outlier handling, and encoding+scaling.

    Fits preprocessing on training data only (for model selection).
    Phase 2 (in EvaluationStage) refits on the full cohort.
    """

    def __init__(self, config, preprocessor, data_splitter, logger):
        self.config = config
        self.preprocessor = preprocessor
        self.data_splitter = data_splitter
        self.logger = logger

    def run(self, ctx: PipelineContext) -> None:
        """Execute preprocessing steps. Mutates ctx in place."""
        data = ctx.data_raw
        ctx.n_rows_original = len(data)

        # Step 0: Row-level missing data filter
        data = self._filter_rows(ctx, data)
        ctx.data_filtered = data

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
        if self.config.data_quality.enabled:
            quality_assessor = DataQualityAssessor(self.config)
            ctx.quality_report = quality_assessor.assess_data_quality(data)
            quality_assessor.save_quality_report(Path(self.config.output_dir) / "quality")

        # Step 2: Detect missing values
        ctx.missing_info = self.preprocessor.detect_missing_values(data)

        # Step 3: Split into train/test BEFORE preprocessing
        self.split_result = self._split_data(ctx, data)

        # Steps 4-6: Impute, outlier handle, encode+scale
        self._fit_transform_preprocessing(ctx)

        # Save raw filtered data for Phase 2 refit on full cohort
        ctx.data_raw = data

    def _filter_rows(self, ctx, data):
        """Step 0: Row-level missing data filter."""
        if not self.config.row_filter.enabled or self.config.row_filter.max_missing_pct >= 1.0:
            return data

        feature_cols = [
            c
            for c in self.config.continuous_columns + self.config.categorical_columns
            if c in data.columns
        ]
        if not feature_cols:
            self.logger.warning("ROW FILTER: No feature columns found in data, skipping filter.")
            return data

        missing_frac = data[feature_cols].isna().mean(axis=1)
        data = data[missing_frac <= self.config.row_filter.max_missing_pct].reset_index(drop=True)
        n_removed = ctx.n_rows_original - len(data)
        self.logger.info(
            f"ROW FILTER: Removed {n_removed} of {ctx.n_rows_original} rows "
            f"({n_removed / ctx.n_rows_original * 100:.1f}%) with "
            f">{self.config.row_filter.max_missing_pct * 100:.0f}% missing values. "
            f"{len(data)} rows remaining."
        )
        return data

    def _split_data(self, ctx, data):
        """Step 3: Train/test split."""
        self.logger.info("Splitting data into train/test sets...")
        split_result = self.data_splitter.split(
            data, stratify_column=self.config.data_split.stratify_by
        )
        ctx.split_info = {
            "train_size": split_result.n_train,
            "test_size": split_result.n_test,
            "train_fraction": split_result.train_fraction,
            "split_type": "random",
        }
        self.logger.info(
            f"  Train: {ctx.split_info['train_size']} ({ctx.split_info['train_fraction']:.1%})"
        )
        self.logger.info(
            f"  Test: {ctx.split_info['test_size']} (held out for unbiased evaluation)"
        )
        return split_result

    def _fit_transform_preprocessing(self, ctx):
        """Steps 4-6: Impute, outlier handle, encode+scale on train/test."""
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

        ctx.X_train = X_train
        ctx.X_test = X_test
        ctx.data_train = data_train_proc
        ctx.data_test = data_test_proc
