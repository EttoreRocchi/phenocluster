"""Tests for stability analysis module."""

from unittest.mock import MagicMock

import numpy as np

from phenocluster.evaluation.stability import (
    StabilityAnalyzer,
    _align_labels,
    _run_single_cluster_stability_iteration,
    _run_single_consensus_iteration,
)


class TestAlignLabels:
    def test_identity(self):
        ref = np.array([0, 0, 1, 1, 2, 2])
        result = _align_labels(ref, ref.copy(), n_clusters=3)
        np.testing.assert_array_equal(result, ref)

    def test_two_cluster_swap(self):
        ref = np.array([0, 0, 0, 1, 1, 1])
        pred = np.array([1, 1, 1, 0, 0, 0])  # swapped
        result = _align_labels(ref, pred, n_clusters=2)
        np.testing.assert_array_equal(result, ref)

    def test_three_cluster_permutation(self):
        ref = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([2, 2, 0, 0, 1, 1])  # permuted: 0->2, 1->0, 2->1
        result = _align_labels(ref, pred, n_clusters=3)
        np.testing.assert_array_equal(result, ref)


class _FakeModel:
    """Minimal model stub for stability tests."""

    def __init__(
        self,
        n_components=2,
        measurement="continuous",
        random_state=None,
        n_init=1,
        max_iter=10,
        abs_tol=1e-4,
        rel_tol=1e-4,
        verbose=0,
        progress_bar=0,
    ):
        self.n_components = n_components
        self.measurement = measurement
        self._labels = None

    def fit(self, X):
        n = len(X)
        self._labels = np.array([i % self.n_components for i in range(n)])

    def predict(self, X):
        n = len(X)
        return np.array([i % self.n_components for i in range(n)])


class TestRunSingleConsensusIteration:
    def test_valid_return(self):
        X = np.random.RandomState(0).randn(40, 3)
        model = _FakeModel(n_components=2)
        indices, labels, status = _run_single_consensus_iteration(
            i=0,
            X=X,
            model=model,
            n_samples=40,
            subsample_size=30,
            min_cluster_size=2,
            random_state=42,
            n_init=1,
            max_iter=10,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )
        assert status == "valid"
        assert indices is not None
        assert labels is not None
        assert len(indices) == 30
        assert len(labels) == 30

    def test_wrong_clusters(self):
        """Model that always returns 1 cluster."""

        class OneClusterModel(_FakeModel):
            def predict(self, X):
                return np.zeros(len(X), dtype=int)

        X = np.random.RandomState(0).randn(40, 3)
        model = OneClusterModel(n_components=2)
        _, _, status = _run_single_consensus_iteration(
            i=0,
            X=X,
            model=model,
            n_samples=40,
            subsample_size=30,
            min_cluster_size=2,
            random_state=42,
            n_init=1,
            max_iter=10,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )
        assert status == "wrong_clusters"

    def test_exception_returns_failed(self):
        """Model whose fit raises."""

        class FailModel(_FakeModel):
            def fit(self, X):
                raise RuntimeError("boom")

        X = np.random.RandomState(0).randn(40, 3)
        model = FailModel(n_components=2)
        _, _, status = _run_single_consensus_iteration(
            i=0,
            X=X,
            model=model,
            n_samples=40,
            subsample_size=30,
            min_cluster_size=2,
            random_state=42,
            n_init=1,
            max_iter=10,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )
        assert status == "failed"


class TestRunSingleClusterStabilityIteration:
    def test_valid_return(self):
        X = np.random.RandomState(0).randn(40, 3)
        model = _FakeModel(n_components=2)
        original_labels = np.array([i % 2 for i in range(40)])
        result = _run_single_cluster_stability_iteration(
            i=0,
            X=X,
            model=model,
            original_labels=original_labels,
            n_samples=40,
            subsample_size=30,
            n_clusters=2,
            min_cluster_size=2,
            random_state=42,
            n_init=1,
            max_iter=10,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )
        assert result is not None
        assert 0 in result and 1 in result
        assert 0.0 <= result[0] <= 1.0

    def test_failure_returns_none(self):
        class FailModel(_FakeModel):
            def fit(self, X):
                raise RuntimeError("boom")

        X = np.random.RandomState(0).randn(40, 3)
        model = FailModel(n_components=2)
        original_labels = np.array([i % 2 for i in range(40)])
        result = _run_single_cluster_stability_iteration(
            i=0,
            X=X,
            model=model,
            original_labels=original_labels,
            n_samples=40,
            subsample_size=30,
            n_clusters=2,
            min_cluster_size=2,
            random_state=42,
            n_init=1,
            max_iter=10,
            abs_tol=1e-4,
            rel_tol=1e-4,
        )
        assert result is None


class TestStabilityAnalyzerStatics:
    def test_accumulate_results_valid(self):
        indices = np.array([0, 1, 2])
        labels = np.array([0, 0, 1])
        results = [(indices, labels, "valid")]
        cooc, cosamp, counts = StabilityAnalyzer._accumulate_results(results, 5)
        assert cooc.shape == (5, 5)
        assert counts["valid"] == 1

    def test_accumulate_results_failed(self):
        results = [(None, None, "failed"), (None, None, "wrong_clusters")]
        _, _, counts = StabilityAnalyzer._accumulate_results(results, 5)
        assert counts["valid"] == 0
        assert counts["failed"] == 1
        assert counts["wrong_clusters"] == 1

    def test_build_consensus_matrix(self):
        cooc = np.array([[2.0, 1.0], [1.0, 2.0]])
        cosamp = np.array([[2.0, 2.0], [2.0, 2.0]])
        consensus = StabilityAnalyzer._build_consensus_matrix(cooc, cosamp)
        assert consensus.shape == (2, 2)
        np.testing.assert_allclose(consensus[0, 1], 0.5)
        np.testing.assert_allclose(np.diag(consensus), 1.0)

    def test_build_consensus_matrix_zero_cosampling(self):
        cooc = np.zeros((3, 3))
        cosamp = np.zeros((3, 3))
        consensus = StabilityAnalyzer._build_consensus_matrix(cooc, cosamp)
        np.testing.assert_allclose(np.diag(consensus), 1.0)
        # Off-diagonal should be 0 (no co-sampling)
        assert consensus[0, 1] == 0.0

    def test_compute_consensus_stats(self):
        consensus = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
        # Create fake results with valid entries
        indices = np.array([0, 1, 2])
        labels = np.array([0, 0, 1])
        results = [(indices, labels, "valid")]
        stats = StabilityAnalyzer._compute_consensus_stats(consensus, results)
        assert "mean" in stats
        assert "std" in stats
        assert "ci_lower" in stats
        assert "ci_upper" in stats
        assert 0.0 <= stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"] <= 1.0


class TestStabilityAnalyzerDisabled:
    def test_disabled_returns_empty(self, minimal_config):
        # Stability is enabled by default; disable it
        minimal_config.stability.enabled = False
        analyzer = StabilityAnalyzer(minimal_config)
        result = analyzer.analyze_stability(np.zeros((10, 3)), MagicMock(), np.zeros(10, dtype=int))
        assert result == {}

    def test_cluster_stability_disabled_returns_empty(self, minimal_config):
        minimal_config.stability.enabled = False
        analyzer = StabilityAnalyzer(minimal_config)
        result = analyzer.analyze_cluster_stability(
            np.zeros((10, 3)), MagicMock(), np.zeros(10, dtype=int), n_clusters=2
        )
        assert result == {}
