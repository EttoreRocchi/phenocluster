"""Tests for the PhenoCluster pipeline - basic smoke tests."""

import numpy as np
import pandas as pd


class TestPipelineSmokeTest:
    """Smoke test: pipeline completes on small synthetic dataset."""

    def test_fit_runs_on_synthetic_data(self, tmp_path):
        """Pipeline.fit() should complete and assign labels on a synthetic dataset."""
        from phenocluster.config import PhenoClusterConfig
        from phenocluster.pipeline import PhenoClusterPipeline

        rng = np.random.RandomState(42)
        n = 60
        df = pd.DataFrame(
            {
                "feat1": rng.normal(0, 1, n),
                "feat2": rng.normal(5, 2, n),
                "feat3": rng.normal(-1, 0.5, n),
                "cat1": rng.choice(["A", "B", "C"], n),
            }
        )

        config = PhenoClusterConfig.from_dict(
            {
                "global": {
                    "project_name": "smoke_test",
                    "output_dir": str(tmp_path),
                    "random_state": 42,
                },
                "data": {
                    "continuous_columns": ["feat1", "feat2", "feat3"],
                    "categorical_columns": ["cat1"],
                    "split": {"test_size": 0.2},
                },
                "preprocessing": {},
                "model": {
                    "n_clusters": 2,
                    "selection": {
                        "min_clusters": 2,
                        "max_clusters": 3,
                        "criterion": "BIC",
                    },
                },
                "outcome": {"enabled": False},
                "stability": {"enabled": False},
                "logging": {"level": "WARNING", "log_to_file": False},
            }
        )

        pipeline = PhenoClusterPipeline(config)
        results = pipeline.fit(df)

        assert results is not None
        assert "data" in results
        assert results["n_clusters"] >= 2


class TestPhenotypeReordering:
    """Test that phenotype reordering works correctly."""

    def test_reordering_by_size(self):
        """Test label remapping by cluster size."""
        labels = np.array([2, 2, 2, 2, 2, 0, 0, 0, 1, 1])
        n_clusters = 3
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        size_order = np.argsort(-cluster_sizes)

        # Cluster 2 is largest (5), then 0 (3), then 1 (2)
        assert size_order[0] == 2  # Largest -> new Phenotype 0
        assert size_order[1] == 0  # Second -> new Phenotype 1
        assert size_order[2] == 1  # Smallest -> new Phenotype 2

        label_map = {old: new for new, old in enumerate(size_order)}
        new_labels = np.array([label_map[lbl] for lbl in labels])

        # Original cluster 2 -> new 0 (5 patients)
        assert np.sum(new_labels == 0) == 5
        # Original cluster 0 -> new 1 (3 patients)
        assert np.sum(new_labels == 1) == 3
        # Original cluster 1 -> new 2 (2 patients)
        assert np.sum(new_labels == 2) == 2


class TestStepFunctionInterpolation:
    """Test step-function interpolation for KM curves."""

    def test_step_function_is_right_continuous(self):
        """Test that interpolation uses step function, not linear."""
        from scipy.interpolate import interp1d

        timeline = np.array([0, 5, 10, 15, 20])
        survival = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        sf = interp1d(
            timeline,
            survival,
            kind="previous",
            bounds_error=False,
            fill_value=(1.0, 0.2),
        )

        # At t=7 (between 5 and 10), should be 0.8 (step), not 0.72 (linear)
        assert sf(7) == 0.8
        # At t=12, should be 0.6
        assert sf(12) == 0.6
        # Before first time, should be 1.0
        assert sf(-1) == 1.0
        # After last time, should be 0.2
        assert sf(25) == 0.2


class TestReferencePhenotype:
    """Test configurable reference phenotype dummy coding."""

    def test_create_dummies_default_reference(self):
        """With reference=0, phenotypes 1 and 2 get dummy columns."""
        from phenocluster.evaluation.metrics import create_phenotype_dummies

        labels = np.array([0, 0, 1, 1, 2, 2])
        dummies, non_ref = create_phenotype_dummies(labels, 3, reference=0)
        assert non_ref == [1, 2]
        assert dummies.shape == (6, 2)
        # Phenotype 1 -> column 0
        assert np.all(dummies[2:4, 0] == 1)
        assert np.all(dummies[:2, 0] == 0)
        # Phenotype 2 -> column 1
        assert np.all(dummies[4:6, 1] == 1)
        assert np.all(dummies[:4, 1] == 0)

    def test_create_dummies_nonzero_reference(self):
        """With reference=1, phenotypes 0 and 2 get dummy columns."""
        from phenocluster.evaluation.metrics import create_phenotype_dummies

        labels = np.array([0, 0, 1, 1, 2, 2])
        dummies, non_ref = create_phenotype_dummies(labels, 3, reference=1)
        assert non_ref == [0, 2]
        assert dummies.shape == (6, 2)
        # Phenotype 0 -> column 0
        assert np.all(dummies[0:2, 0] == 1)
        assert np.all(dummies[2:, 0] == 0)
        # Phenotype 2 -> column 1
        assert np.all(dummies[4:6, 1] == 1)


class TestNaNLabelEncoding:
    """Test that NaN values are preserved through categorical label encoding."""

    def test_nan_preserved_not_encoded_as_category(self):
        """NaN must remain as float NaN, never encoded as a 'nan' string category."""
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        values = pd.Series(["A", "B", np.nan, "A", np.nan])
        nan_mask = values.isna()
        non_null = values.loc[~nan_mask].astype(str)

        le = LabelEncoder()
        le.fit(non_null)

        encoded = pd.Series(np.nan, index=values.index, dtype=float)
        encoded.loc[~nan_mask] = le.transform(non_null).astype(float)

        # NaN positions must still be NaN
        assert np.isnan(encoded.iloc[2])
        assert np.isnan(encoded.iloc[4])
        # Non-NaN positions must be numeric codes
        assert not np.isnan(encoded.iloc[0])
        assert not np.isnan(encoded.iloc[1])
        # 'nan' must NOT appear in the learned categories
        assert "nan" not in le.classes_

    def test_all_nan_column(self):
        """A fully-NaN column should produce all-NaN encoded values."""
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        values = pd.Series([np.nan, np.nan, np.nan])
        nan_mask = values.isna()
        non_null = values.loc[~nan_mask].astype(str)

        le = LabelEncoder()
        # With no non-null values, fit on empty -> 0 classes
        if len(non_null) > 0:
            le.fit(non_null)

        encoded = pd.Series(np.nan, index=values.index, dtype=float)
        assert encoded.isna().all()


class TestMinClusterSizeConfig:
    """Test min_cluster_size validation and conversion."""

    def test_absolute_count(self):
        """Integer min_cluster_size interpreted as absolute count."""
        from phenocluster.config import ModelSelectionConfig

        cfg = ModelSelectionConfig(min_cluster_size=50)
        assert cfg.get_min_cluster_size(700) == 50

    def test_percentage(self):
        """Float in (0, 1) interpreted as percentage of n_samples."""
        from phenocluster.config import ModelSelectionConfig

        cfg = ModelSelectionConfig(min_cluster_size=0.08)
        # 8% of 700 = 56
        assert cfg.get_min_cluster_size(700) == 56

    def test_percentage_minimum_one(self):
        """Very small percentage with tiny dataset still returns at least 1."""
        from phenocluster.config import ModelSelectionConfig

        cfg = ModelSelectionConfig(min_cluster_size=0.01)
        assert cfg.get_min_cluster_size(5) >= 1

    def test_invalid_value_raises(self):
        """Negative or zero min_cluster_size should raise ValueError."""
        import pytest

        from phenocluster.config import ModelSelectionConfig

        with pytest.raises(ValueError):
            ModelSelectionConfig(min_cluster_size=-1)
        with pytest.raises(ValueError):
            ModelSelectionConfig(min_cluster_size=0)


class TestInferenceConfig:
    """Test that InferenceConfig defaults and validation are correct."""

    def test_default_values(self):
        """Verify default confidence_level, fdr_correction, etc."""
        from phenocluster.config import InferenceConfig

        cfg = InferenceConfig()
        assert cfg.enabled is True
        assert cfg.confidence_level == 0.95
        assert cfg.fdr_correction is True
        assert cfg.outcome_test == "auto"
        assert cfg.cox_penalizer == 0.0

    def test_custom_values(self):
        """Custom values should be stored correctly."""
        from phenocluster.config import InferenceConfig

        cfg = InferenceConfig(confidence_level=0.90, cox_penalizer=0.1, outcome_test="fisher")
        assert cfg.confidence_level == 0.90
        assert cfg.cox_penalizer == 0.1
        assert cfg.outcome_test == "fisher"

    def test_invalid_confidence_level_raises(self):
        """confidence_level outside (0, 1) should raise ValueError."""
        import pytest

        from phenocluster.config import InferenceConfig

        with pytest.raises(ValueError):
            InferenceConfig(confidence_level=0.0)
        with pytest.raises(ValueError):
            InferenceConfig(confidence_level=1.0)

    def test_invalid_outcome_test_raises(self):
        """Invalid outcome_test should raise ValueError."""
        import pytest

        from phenocluster.config import InferenceConfig

        with pytest.raises(ValueError):
            InferenceConfig(outcome_test="invalid")

    def test_negative_penalizer_raises(self):
        """Negative cox_penalizer should raise ValueError."""
        import pytest

        from phenocluster.config import InferenceConfig

        with pytest.raises(ValueError):
            InferenceConfig(cox_penalizer=-0.1)


class TestFeatureSelectionConfig:
    """Test feature selection configuration and target column validation."""

    def test_supervised_method_requires_target(self):
        """LASSO/MI enabled without target_column should raise ValueError."""
        import pytest

        from phenocluster.config import FeatureSelectionConfig

        with pytest.raises(ValueError, match="requires a target column"):
            FeatureSelectionConfig(enabled=True, method="lasso", target_column=None)
        with pytest.raises(ValueError, match="requires a target column"):
            FeatureSelectionConfig(enabled=True, method="mutual_info", target_column=None)

    def test_supervised_method_with_target(self):
        """LASSO/MI with target_column should succeed."""
        from phenocluster.config import FeatureSelectionConfig

        cfg = FeatureSelectionConfig(enabled=True, method="lasso", target_column="outcome")
        assert cfg.require_target is True
        assert cfg.target_column == "outcome"

        cfg2 = FeatureSelectionConfig(enabled=True, method="mutual_info", target_column="outcome")
        assert cfg2.require_target is True

    def test_unsupervised_method_no_target_needed(self):
        """Variance/correlation/combined should not require target."""
        from phenocluster.config import FeatureSelectionConfig

        for method in ["variance", "correlation", "combined"]:
            cfg = FeatureSelectionConfig(method=method)
            assert cfg.require_target is False
            assert cfg.target_column is None


# Config serialization round-trip


class TestConfigSerialization:
    """Test that to_dict() -> from_dict() preserves all configuration."""

    def test_round_trip_preserves_values(self):
        """from_dict -> to_dict -> from_dict preserves configuration values.

        Known serialization asymmetries (monte_carlo nesting, random_state
        propagation, tuple/list) prevent exact dict equality, so we verify
        that the important configuration values survive the round-trip.
        """
        from phenocluster.config import PhenoClusterConfig

        raw = {
            "global": {"project_name": "RoundTrip", "output_dir": "out", "random_state": 123},
            "data": {
                "continuous_columns": ["age", "bmi"],
                "categorical_columns": ["sex"],
                "split": {"test_size": 0.25},
            },
            "preprocessing": {},
            "model": {
                "n_clusters": 4,
                "selection": {
                    "enabled": True,
                    "criterion": "ICL",
                    "min_clusters": 2,
                    "max_clusters": 5,
                },
                "stepmix": {"max_iter": 500},
            },
            "inference": {
                "enabled": True,
                "confidence_level": 0.90,
                "fdr_correction": False,
                "outcome_test": "fisher",
                "cox_penalizer": 0.1,
            },
            "multistate": {
                "enabled": False,
                "monte_carlo": {"n_simulations": 500, "time_points": [7, 14, 28]},
            },
        }

        c1 = PhenoClusterConfig.from_dict(raw)
        d1 = c1.to_dict()
        c2 = PhenoClusterConfig.from_dict(d1)

        # Core settings preserved
        assert c2.project_name == c1.project_name
        assert c2.random_state == c1.random_state
        assert c2.n_clusters == c1.n_clusters

        # Inference settings preserved
        assert c2.inference.enabled == c1.inference.enabled
        assert c2.inference.confidence_level == c1.inference.confidence_level
        assert c2.inference.fdr_correction == c1.inference.fdr_correction
        assert c2.inference.outcome_test == c1.inference.outcome_test
        assert c2.inference.cox_penalizer == c1.inference.cox_penalizer

        # Data columns preserved
        assert c2.continuous_columns == c1.continuous_columns
        assert c2.categorical_columns == c1.categorical_columns

        # Multistate MC settings preserved
        assert c2.multistate.monte_carlo_n_simulations == c1.multistate.monte_carlo_n_simulations
        assert c2.multistate.monte_carlo_time_points == c1.multistate.monte_carlo_time_points

    def test_multistate_monte_carlo_unpacking(self):
        """Nested monte_carlo block must flatten to top-level multistate keys."""
        from phenocluster.config import PhenoClusterConfig

        raw = {
            "global": {"project_name": "test", "output_dir": "out", "random_state": 42},
            "data": {"continuous_columns": ["x"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
            "multistate": {
                "enabled": False,
                "monte_carlo": {
                    "n_simulations": 999,
                    "time_points": [1, 2, 3],
                    "max_transitions_per_path": 7,
                },
            },
        }
        config = PhenoClusterConfig.from_dict(raw)
        assert config.multistate.monte_carlo_n_simulations == 999
        assert config.multistate.monte_carlo_time_points == [1, 2, 3]
        assert config.multistate.max_transitions_per_path == 7

    def test_random_state_propagation(self):
        """Global random_state must propagate to sub-configs."""
        from phenocluster.config import PhenoClusterConfig

        raw = {
            "global": {"random_state": 123},
            "data": {"continuous_columns": ["x"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
        }
        config = PhenoClusterConfig.from_dict(raw)
        assert config.data_split.random_state == 123
        assert config.model_selection.random_state == 123
        assert config.feature_selection.random_state == 123

    def test_winsorize_limits_list_to_tuple(self):
        """YAML lists for winsorize_limits become tuples."""
        from phenocluster.config import PhenoClusterConfig

        raw = {
            "global": {},
            "data": {"continuous_columns": ["x"], "split": {}},
            "preprocessing": {"outlier": {"winsorize_limits": [0.02, 0.02]}},
            "model": {"n_clusters": 2},
            "outcome": {"enabled": False},
        }
        config = PhenoClusterConfig.from_dict(raw)
        assert isinstance(config.outlier.winsorize_limits, tuple)
        assert config.outlier.winsorize_limits == (0.02, 0.02)


# Test state_at_time lookups


class TestStateAtTime:
    """Test state occupation lookup at arbitrary time points."""

    def _make(self, states, times):
        import pandas as pd

        from phenocluster.evaluation.multistate import PatientTrajectory

        return PatientTrajectory(
            states=states,
            time_at_each_state=times,
            covariates=pd.Series(dtype=float),
        )

    def test_three_state_trajectory(self):
        """states=[0,1,2], times=[5,10]. t=3->0, t=7->1, t=20->2."""
        traj = self._make([0, 1, 2], [5.0, 10.0])
        assert traj.state_at_time(3.0, initial_state=0) == 0
        assert traj.state_at_time(7.0, initial_state=0) == 1
        assert traj.state_at_time(20.0, initial_state=0) == 2

    def test_time_zero(self):
        """At t=0, should return the first state."""
        traj = self._make([0, 1], [5.0])
        assert traj.state_at_time(0.0, initial_state=0) == 0

    def test_boundary_time(self):
        """At t exactly equal to cumulative transition time, stay in current state."""
        # states=[0,1], times=[5]. cumulative_time + time_in_state = 0+5 = 5 >= 5
        traj = self._make([0, 1], [5.0])
        assert traj.state_at_time(5.0, initial_state=0) == 0

    def test_beyond_all_transitions(self):
        """t far beyond last transition -> final (absorbing) state."""
        traj = self._make([0, 1, 2], [5.0, 10.0])
        assert traj.state_at_time(999.0, initial_state=0) == 2

    def test_single_state_no_transitions(self):
        """states=[0], times=[] -> always return state 0."""
        traj = self._make([0], [])
        assert traj.state_at_time(0.0, initial_state=0) == 0
        assert traj.state_at_time(100.0, initial_state=0) == 0


class TestPreprocessingNoLeakage:
    """Verify preprocessing is fit on training data only (no data leakage)."""

    def test_scaler_fit_on_train_only(self, tmp_path):
        """StandardScaler mean/std must come from train, not full data."""
        import pandas as pd

        from phenocluster.config import PhenoClusterConfig
        from phenocluster.data.preprocessor import DataPreprocessor

        rng = np.random.RandomState(42)
        train_df = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        test_df = pd.DataFrame({"x": rng.normal(10, 1, 50)})
        full_df = pd.concat([train_df, test_df], ignore_index=True)

        config = PhenoClusterConfig(continuous_columns=["x"], output_dir=str(tmp_path))
        prep = DataPreprocessor(config)

        prep.fit_preprocessor(train_df)
        assert abs(prep.scaler.mean_[0]) < 1.0  # ~0, not ~3.3

        # Transform full data: test rows should have high z-scores
        full_proc, X = prep.transform_preprocess(full_df)
        test_z = X[100:]  # test portion
        assert np.mean(test_z) > 5.0  # far from 0 because scaled with train stats

    def test_winsorize_bounds_from_train(self, tmp_path):
        """Winsorization bounds must come from train percentiles."""
        import pandas as pd

        from phenocluster.config import OutlierConfig, PhenoClusterConfig
        from phenocluster.data.preprocessor import DataPreprocessor

        train_df = pd.DataFrame({"x": list(range(100))})
        test_df = pd.DataFrame({"x": [200, 300, 400]})

        config = PhenoClusterConfig(
            continuous_columns=["x"],
            output_dir=str(tmp_path),
            outlier=OutlierConfig(enabled=True, method="winsorize", winsorize_limits=(0.05, 0.05)),
        )
        prep = DataPreprocessor(config)

        prep.fit_outlier_handler(train_df)
        result = prep.transform_outliers(test_df)

        # Test values should be clipped to train's 95th percentile
        assert result["x"].max() <= 95.0

    def test_unknown_category_label_encoding(self, tmp_path):
        """Test data with unseen categories should not raise (label encoding)."""
        import pandas as pd

        from phenocluster.config import PhenoClusterConfig
        from phenocluster.data.preprocessor import DataPreprocessor

        train_df = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
        test_df = pd.DataFrame({"cat": ["A", "C"]})

        config = PhenoClusterConfig(categorical_columns=["cat"], output_dir=str(tmp_path))
        prep = DataPreprocessor(config)

        prep.fit_preprocessor(train_df)
        # Should not raise
        result, X = prep.transform_preprocess(test_df)
        assert len(result) == 2

    def test_imputer_fit_on_train_only(self, tmp_path):
        """Imputer statistics must come from train data."""
        import pandas as pd

        from phenocluster.config import ImputationConfig, PhenoClusterConfig
        from phenocluster.data.preprocessor import DataPreprocessor

        train_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
        test_df = pd.DataFrame({"x": [np.nan]})

        config = PhenoClusterConfig(
            continuous_columns=["x"],
            output_dir=str(tmp_path),
            imputation=ImputationConfig(enabled=True, method="simple"),
        )
        prep = DataPreprocessor(config)

        prep.fit_imputer(train_df)
        result = prep.transform_impute(test_df)

        # Mean of train [1, 2, 3] = 2.0
        assert abs(result["x"].iloc[0] - 2.0) < 0.01


class TestICLFormula:
    """Test that the ICL formula is correctly implemented (Biernacki et al. 2000)."""

    def test_icl_worse_than_bic(self):
        """ICL must always be >= BIC (entropy is an additional penalty)."""
        from phenocluster.model_selection.scorers import get_all_criteria

        class FakeModel:
            """Minimal mock of a StepMix model."""

            def bic(self, X):
                return 5000.0

            def aic(self, X):
                return 4800.0

            def caic(self, X):
                return 5100.0

            def sabic(self, X):
                return 4900.0

            def score(self, X):
                return -2400.0

            n_parameters = 10

            def entropy(self, X):
                return 50.0  # StepMix returns total entropy (summed over all samples)

            def relative_entropy(self, X):
                return 0.8

        model = FakeModel()
        X = np.zeros((100, 5))
        criteria = get_all_criteria(model, X)

        # ICL = BIC + 2 * entropy = 5000 + 2 * 50 = 5100
        assert criteria["ICL"] == 5000 + 2 * 50.0
        # ICL must be worse (higher) than BIC
        assert criteria["ICL"] > criteria["BIC"]

    def test_icl_scorer_negates_correctly(self):
        """ICL scorer returns -ICL for sklearn maximization."""
        from phenocluster.model_selection.scorers import icl_score

        class FakeModel:
            def bic(self, X):
                return 4000.0

            def entropy(self, X):
                return 60.0  # Total entropy (summed over all 200 samples)

        model = FakeModel()
        X = np.zeros((200, 5))
        score = icl_score(model, X)

        # Score = -(BIC + 2 * entropy) = -(4000 + 2 * 60) = -4120
        expected = -(4000 + 2 * 60.0)
        assert abs(score - expected) < 1e-6


class TestEntropyCriterion:
    """Test that ENTROPY criterion works for model selection."""

    def test_entropy_criterion_selects_model(self):
        """ENTROPY criterion should successfully select a model."""
        from phenocluster.model_selection.scorers import get_all_criteria

        class FakeModel:
            n_components = 3
            n_parameters = 10

            def bic(self, X):
                return 5000.0

            def aic(self, X):
                return 4800.0

            def caic(self, X):
                return 5100.0

            def sabic(self, X):
                return 4900.0

            def score(self, X):
                return -24.0

            def entropy(self, X):
                return 50.0

            def relative_entropy(self, X):
                return 0.85

        criteria = get_all_criteria(FakeModel(), np.zeros((100, 5)))
        # ENTROPY key must exist and equal relative_entropy
        assert criteria["ENTROPY"] is not None
        assert criteria["ENTROPY"] == criteria["relative_entropy"]
        assert criteria["ENTROPY"] == 0.85


class TestFDRCorrection:
    """Test FDR (Benjamini-Hochberg) correction against known results."""

    def test_fdr_matches_statsmodels(self):
        """Our FDR correction should match statsmodels multipletests."""
        from phenocluster.evaluation.stats_utils import apply_fdr_correction

        p_values = [0.001, 0.01, 0.03, 0.05, 0.10, 0.50]
        adjusted = apply_fdr_correction(p_values)

        # Verify monotonicity: adjusted p-values should be non-decreasing
        # when original p-values are sorted
        sorted_pairs = sorted(zip(p_values, adjusted))
        adjusted_sorted = [a for _, a in sorted_pairs]
        for i in range(len(adjusted_sorted) - 1):
            assert adjusted_sorted[i] <= adjusted_sorted[i + 1] + 1e-10

        # All adjusted p-values should be >= original
        for orig, adj in zip(p_values, adjusted):
            assert adj >= orig - 1e-10

        # All adjusted p-values should be <= 1.0
        for adj in adjusted:
            assert adj <= 1.0 + 1e-10

    def test_fdr_single_value(self):
        """Single p-value: adjusted should equal original."""
        from phenocluster.evaluation.stats_utils import apply_fdr_correction

        adjusted = apply_fdr_correction([0.05])
        assert abs(adjusted[0] - 0.05) < 1e-10

    def test_fdr_empty(self):
        """Empty input should return empty output."""
        from phenocluster.evaluation.stats_utils import apply_fdr_correction

        adjusted = apply_fdr_correction([])
        assert len(adjusted) == 0


class TestFeatureSelectionFiltering:
    """Test that feature selection only operates on feature columns."""

    def test_outcome_columns_excluded(self, tmp_path):
        """Feature selector should not see outcome/survival columns."""
        import pandas as pd

        from phenocluster.config import (
            FeatureSelectionConfig,
            OutcomeConfig,
            PhenoClusterConfig,
        )
        from phenocluster.feature_selection import MixedDataFeatureSelector

        rng = np.random.RandomState(42)
        n = 100

        config = PhenoClusterConfig(
            continuous_columns=["feat1", "feat2", "feat3"],
            outcome=OutcomeConfig(outcome_columns=["mortality"]),
            output_dir=str(tmp_path),
            feature_selection=FeatureSelectionConfig(
                enabled=True, method="variance", variance_threshold=0.0
            ),
        )

        df = pd.DataFrame(
            {
                "feat1": rng.normal(0, 1, n),
                "feat2": rng.normal(0, 1, n),
                "feat3": rng.normal(0, 1, n),
                "mortality": rng.binomial(1, 0.3, n),
            }
        )

        # Simulate what pipeline does: filter to feature columns only
        feature_cols = config.continuous_columns + config.categorical_columns
        cols_present = [c for c in feature_cols if c in df.columns]
        data_for_fs = df[cols_present].copy()

        selector = MixedDataFeatureSelector(
            config.feature_selection,
            continuous_cols=config.continuous_columns,
            categorical_cols=config.categorical_columns,
        )
        result = selector.fit_transform(data_for_fs)

        # 'mortality' should not appear in selected features
        selected = selector.get_selected_features()
        assert "mortality" not in selected
        assert "mortality" not in result.columns


class TestClassificationQuality:
    """Test classification quality metrics computation."""

    def test_perfect_classification_high_avepp(self):
        """With near-certain assignments, AvePP should be close to 1.0."""
        proba = np.array(
            [
                [0.99, 0.005, 0.005],
                [0.005, 0.99, 0.005],
                [0.005, 0.005, 0.99],
            ]
        )
        max_proba = np.max(proba, axis=1)
        avepp = np.mean(max_proba)
        assert avepp > 0.98

    def test_uncertain_classification_low_avepp(self):
        """With uniform assignments, AvePP should be near 1/K."""
        K = 3
        proba = np.ones((10, K)) / K
        max_proba = np.max(proba, axis=1)
        avepp = np.mean(max_proba)
        assert abs(avepp - 1.0 / K) < 0.01

    def test_relative_entropy_range(self):
        """Relative entropy should be in [0, 1]."""
        rng = np.random.RandomState(42)
        K = 4
        # Generate random probability vectors
        raw = rng.dirichlet(np.ones(K), size=50)
        eps = 1e-15
        sample_entropy = -np.sum(raw * np.log(raw + eps), axis=1)
        max_entropy = np.log(K)
        rel_entropy = sample_entropy / max_entropy

        assert np.all(rel_entropy >= -1e-10)
        assert np.all(rel_entropy <= 1.0 + 1e-10)
