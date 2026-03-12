"""Tests for the artifact cache module."""

import numpy as np
import pandas as pd
import pytest

from phenocluster.cache import (
    PIPELINE_STEPS,
    ArtifactCache,
    _compute_data_fingerprint,
)


@pytest.fixture
def sample_config():
    """Minimal config dict matching the nested to_dict() output format."""
    return {
        "global": {"project_name": "test", "output_dir": "results", "random_state": 42},
        "data": {
            "continuous_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "outcome_columns": ["y"],
            "split": {"test_size": 0.2},
        },
        "preprocessing": {
            "imputation": {"enabled": False},
            "categorical_encoding": {"method": "label"},
            "outlier": {"enabled": False},
            "row_filter": {"enabled": False},
            "feature_selection": {"enabled": False},
        },
        "model": {
            "n_clusters": 3,
            "selection": {"enabled": True, "min_clusters": 2},
            "stepmix": {"n_init": 10},
        },
        "reference_phenotype": {"strategy": "largest"},
        "stability": {"enabled": True, "n_runs": 50},
        "survival": {"enabled": True},
        "multistate": {"enabled": False},
        "inference": {"enabled": True, "confidence_level": 0.95},
    }


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "artifacts"


class TestDataFingerprint:
    def test_deterministic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        h1 = _compute_data_fingerprint(df)
        h2 = _compute_data_fingerprint(df)
        assert h1 == h2

    def test_sensitive_to_data_change(self):
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 99], "b": [4, 5, 6]})
        assert _compute_data_fingerprint(df1) != _compute_data_fingerprint(df2)

    def test_sensitive_to_column_change(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "c": [3, 4]})
        assert _compute_data_fingerprint(df1) != _compute_data_fingerprint(df2)


class TestConfigHashing:
    def test_same_config_same_hash(self, cache_dir, sample_config):
        c1 = ArtifactCache(cache_dir, sample_config)
        c2 = ArtifactCache(cache_dir, sample_config)
        for step in PIPELINE_STEPS:
            assert c1.compute_step_hash(step) == c2.compute_step_hash(step)

    def test_different_config_different_hash(self, cache_dir, sample_config):
        c1 = ArtifactCache(cache_dir, sample_config)

        modified = {**sample_config, "stability": {"enabled": True, "n_runs": 200}}
        c2 = ArtifactCache(cache_dir, modified)

        # stability step should differ
        assert c1.compute_step_hash("stability") != c2.compute_step_hash("stability")

        # upstream steps should be identical
        assert c1.compute_step_hash("preprocess") == c2.compute_step_hash("preprocess")
        assert c1.compute_step_hash("train_model") == c2.compute_step_hash("train_model")

    def test_upstream_change_cascades(self, cache_dir, sample_config):
        c1 = ArtifactCache(cache_dir, sample_config)

        modified_model = {**sample_config["model"], "stepmix": {"n_init": 999}}
        modified = {**sample_config, "model": modified_model}
        c2 = ArtifactCache(cache_dir, modified)

        # train_model should differ (stepmix changed)
        assert c1.compute_step_hash("train_model") != c2.compute_step_hash("train_model")

        # downstream steps should also differ (chained)
        assert c1.compute_step_hash("evaluate_model") != c2.compute_step_hash("evaluate_model")
        assert c1.compute_step_hash("stability") != c2.compute_step_hash("stability")
        assert c1.compute_step_hash("run_analyses") != c2.compute_step_hash("run_analyses")


class TestCascadeInvalidation:
    def test_invalidate_evaluate_model(self, cache_dir, sample_config):
        cache = ArtifactCache(cache_dir, sample_config)
        data_hash = "test123"

        # Populate all steps
        for step in PIPELINE_STEPS:
            cache.save_step_artifacts(step, {"data": 42}, data_hash)
            assert cache.is_step_valid(step, data_hash)

        # Invalidate evaluate_model -> stability + run_analyses also invalid
        cache.invalidate_from("evaluate_model")

        assert cache.is_step_valid("preprocess", data_hash)
        assert cache.is_step_valid("feature_select", data_hash)
        assert cache.is_step_valid("train_model", data_hash)
        assert not cache.is_step_valid("evaluate_model", data_hash)
        assert not cache.is_step_valid("stability", data_hash)
        assert not cache.is_step_valid("run_analyses", data_hash)

    def test_fork_independence(self, cache_dir, sample_config):
        """Changing stability config should NOT invalidate run_analyses."""
        cache = ArtifactCache(cache_dir, sample_config)
        data_hash = "test456"

        for step in PIPELINE_STEPS:
            cache.save_step_artifacts(step, {"data": 42}, data_hash)

        # Only invalidate stability
        cache.invalidate_from("stability")

        assert cache.is_step_valid("run_analyses", data_hash)
        assert not cache.is_step_valid("stability", data_hash)


class TestSaveLoad:
    def test_roundtrip(self, cache_dir, sample_config):
        cache = ArtifactCache(cache_dir, sample_config)
        data_hash = "abc"

        artifacts = {
            "array": np.array([1, 2, 3]),
            "dict": {"key": "value"},
            "number": 42,
        }
        cache.save_step_artifacts("preprocess", artifacts, data_hash)

        loaded = cache.load_step_artifacts("preprocess")
        np.testing.assert_array_equal(loaded["array"], artifacts["array"])
        assert loaded["dict"] == artifacts["dict"]
        assert loaded["number"] == 42

    def test_missing_file_invalidates(self, cache_dir, sample_config):
        cache = ArtifactCache(cache_dir, sample_config)
        data_hash = "abc"

        cache.save_step_artifacts("preprocess", {"data": 1}, data_hash)
        assert cache.is_step_valid("preprocess", data_hash)

        # Delete the file
        (cache_dir / "cache_preprocess.joblib").unlink()
        assert not cache.is_step_valid("preprocess", data_hash)

    def test_data_hash_mismatch_invalidates(self, cache_dir, sample_config):
        cache = ArtifactCache(cache_dir, sample_config)
        cache.save_step_artifacts("preprocess", {"data": 1}, "hash_a")

        # Different data hash -> invalid
        assert not cache.is_step_valid("preprocess", "hash_b")


class TestManifestPersistence:
    def test_manifest_survives_reload(self, cache_dir, sample_config):
        cache1 = ArtifactCache(cache_dir, sample_config)
        cache1.save_step_artifacts("preprocess", {"x": 1}, "h1")

        # New cache instance reads the saved manifest
        cache2 = ArtifactCache(cache_dir, sample_config)
        assert cache2.is_step_valid("preprocess", "h1")
