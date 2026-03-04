"""
PhenoCluster Stability Analysis Module
=======================================

Cluster stability analysis using consensus clustering stability analysis.
"""

import sys
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ..config import PhenoClusterConfig
from ..utils.logging import get_logger


def _run_single_consensus_iteration(
    i: int,
    X: np.ndarray,
    model,
    n_samples: int,
    subsample_size: int,
    min_cluster_size: int,
    random_state: Optional[int],
    n_init: int,
    max_iter: int,
    abs_tol: float,
    rel_tol: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Run a single consensus clustering iteration.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], str]
        (indices, labels, status) where status is 'valid', 'wrong_clusters', or 'failed'
    """
    # Create local random generator for this iteration
    seed = random_state + i if random_state is not None else None
    rng = np.random.default_rng(seed)

    # Subsample WITHOUT replacement
    indices = rng.choice(n_samples, size=subsample_size, replace=False)
    X_sub = X[indices]

    try:
        # Fit model on subsample with same parameters as original
        model_sub = type(model)(
            n_components=model.n_components,
            measurement=model.measurement,
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            verbose=0,
            progress_bar=0,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model_sub.fit(X_sub)

        # Get cluster assignments for subsampled data
        labels_sub = model_sub.predict(X_sub)
        n_clusters_found = len(np.unique(labels_sub))

        # Check if correct number of clusters
        if n_clusters_found != model.n_components:
            return None, None, "wrong_clusters"

        # Check minimum cluster size constraint
        unique_labels, counts = np.unique(labels_sub, return_counts=True)
        if np.any(counts < min_cluster_size):
            return None, None, "small_clusters"

        return indices, labels_sub, "valid"

    except Exception:
        return None, None, "failed"


def _align_labels(reference_labels, predicted_labels, n_clusters):
    """Align predicted labels to reference using Hungarian algorithm."""
    # Build confusion matrix
    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    for ref, pred in zip(reference_labels, predicted_labels):
        if ref < n_clusters and pred < n_clusters:
            confusion[ref, pred] += 1
    # Maximise overlap -> minimise negative overlap
    row_ind, col_ind = linear_sum_assignment(-confusion)
    mapping = dict(zip(col_ind, row_ind))
    return np.array([mapping.get(lab, lab) for lab in predicted_labels])


def _run_single_cluster_stability_iteration(
    i: int,
    X: np.ndarray,
    model,
    original_labels: np.ndarray,
    n_samples: int,
    subsample_size: int,
    n_clusters: int,
    min_cluster_size: int,
    random_state: Optional[int],
    n_init: int,
    max_iter: int,
    abs_tol: float,
    rel_tol: float,
) -> Optional[Dict[int, float]]:
    """
    Run a single per-cluster stability iteration.

    Returns
    -------
    Optional[Dict[int, float]]
        Dictionary mapping cluster_id to consistency score, or None if iteration failed
    """
    # Create local random generator for this iteration
    seed = random_state + i if random_state is not None else None
    rng = np.random.default_rng(seed)

    indices = rng.choice(n_samples, size=subsample_size, replace=False)
    X_sub = X[indices]

    try:
        model_sub = type(model)(
            n_components=model.n_components,
            measurement=model.measurement,
            random_state=seed,
            n_init=n_init,
            max_iter=max_iter,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            verbose=0,
            progress_bar=0,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model_sub.fit(X_sub)

        # Check if model converged to correct number of clusters
        sub_labels_sample = model_sub.predict(X_sub)
        n_clusters_found = len(np.unique(sub_labels_sample))

        if n_clusters_found != n_clusters:
            return None

        # Check minimum cluster size constraint
        unique_labels, counts = np.unique(sub_labels_sample, return_counts=True)
        if np.any(counts < min_cluster_size):
            return None

        sub_labels = model_sub.predict(X)
        sub_labels = _align_labels(original_labels, sub_labels, n_clusters)

        # Verify prediction has correct number of clusters
        n_pred_clusters = len(np.unique(sub_labels))
        if n_pred_clusters != n_clusters:
            return None

        # For each original cluster, check assignment consistency
        result = {}
        for cluster_id in range(n_clusters):
            mask = original_labels == cluster_id
            if mask.sum() > 0:
                sub_assignments = sub_labels[mask]
                mode_assignment = np.bincount(sub_assignments).argmax()
                consistency = np.mean(sub_assignments == mode_assignment)
                result[cluster_id] = consistency

        return result

    except Exception:
        return None


class StabilityAnalyzer:
    """
    Performs consensus clustering stability analysis.
    """

    def __init__(self, config: PhenoClusterConfig):
        """
        Initialize the stability analyzer.

        Parameters
        ----------
        config : PhenoClusterConfig
            Configuration object
        """
        self.config = config
        self.logger = get_logger("stability", config)

    def analyze_stability(self, X: np.ndarray, model, original_labels: np.ndarray) -> Dict:
        """
        Perform consensus clustering stability analysis.

        Uses subsampling to build a co-occurrence matrix and measure
        cluster stability. Runs in parallel for faster execution.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        model
            Fitted model (with fit and predict methods)
        original_labels : np.ndarray
            Original cluster assignments

        Returns
        -------
        Dict
            Dictionary with stability metrics
        """
        if not self.config.stability.enabled:
            self.logger.info("Stability analysis disabled")
            return {}

        self.logger.info("CONSENSUS CLUSTERING STABILITY ANALYSIS")
        self.logger.info(f"Subsampling runs: {self.config.stability.n_runs}")
        self.logger.info(f"Subsample fraction: {self.config.stability.subsample_fraction}")

        n_jobs = getattr(self.config.stability, "n_jobs", -1)
        if n_jobs != 1:
            self.logger.info(f"Parallel processing: {n_jobs} jobs (-1 = all cores)")

        n_samples = len(X)

        MAX_CONSENSUS_SAMPLES = 10_000
        if n_samples > MAX_CONSENSUS_SAMPLES:
            self.logger.warning(
                f"Dataset has {n_samples} samples; consensus matrix would require "
                f"{2 * n_samples**2 * 8 / 1e9:.1f} GB. Subsampling to {MAX_CONSENSUS_SAMPLES}."
            )
            rng = np.random.RandomState(self.config.random_state)
            subsample_idx = rng.choice(n_samples, MAX_CONSENSUS_SAMPLES, replace=False)
            X = X[subsample_idx]
            n_samples = MAX_CONSENSUS_SAMPLES

        subsample_size = int(n_samples * self.config.stability.subsample_fraction)

        # Co-occurrence matrix: how often pairs of samples are in same cluster
        cooccurrence_matrix = np.zeros((n_samples, n_samples))
        # Co-sampling matrix: how often pairs were both in the same subsample (Monti et al. 2003)
        cosampling_matrix = np.zeros((n_samples, n_samples))

        self.logger.info("Running consensus clustering analysis...")

        # Get min cluster size
        min_cluster_size = self.config.model_selection.get_min_cluster_size(n_samples)

        # Run iterations in parallel (tqdm wraps result generator for completion tracking)
        gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_run_single_consensus_iteration)(
                i,
                X,
                model,
                n_samples,
                subsample_size,
                min_cluster_size,
                self.config.stepmix.random_state,
                self.config.stepmix.n_init,
                self.config.stepmix.max_iter,
                self.config.stepmix.abs_tol,
                self.config.stepmix.rel_tol,
            )
            for i in range(self.config.stability.n_runs)
        )
        results = list(
            tqdm(
                gen,
                total=self.config.stability.n_runs,
                desc="Consensus clustering",
                file=sys.stdout,
                leave=False,
                disable=self.config.logging.quiet_mode,
            )
        )

        # Process results
        valid_runs = 0
        n_failed = 0
        n_wrong_clusters = 0
        n_small_clusters = 0

        for indices, labels_sub, status in results:
            if status == "valid":
                ix = np.ix_(indices, indices)
                cosampling_matrix[ix] += 1
                same_cluster = labels_sub[:, None] == labels_sub[None, :]
                cooccurrence_matrix[ix] += same_cluster
                valid_runs += 1
            elif status == "wrong_clusters":
                n_wrong_clusters += 1
            elif status == "small_clusters":
                n_small_clusters += 1
            else:
                n_failed += 1

        if valid_runs == 0:
            self.logger.warning("All subsampling runs failed or had wrong cluster counts")
            return {
                "mean_consensus": 0.0,
                "std_consensus": 0.0,
                "ci_95_lower": 0.0,
                "ci_95_upper": 0.0,
                "consensus_matrix": None,
                "n_runs": 0,
                "n_valid": 0,
                "n_failed": n_failed,
                "n_wrong_clusters": n_wrong_clusters,
            }

        # Normalize consensus matrix: proportion of co-samplings where i,j shared a cluster
        # (Monti et al. 2003: C(i,j) = co-occurrence(i,j) / co-sampling(i,j))
        consensus_matrix = np.zeros((n_samples, n_samples))
        mask = cosampling_matrix > 0
        consensus_matrix[mask] = cooccurrence_matrix[mask] / cosampling_matrix[mask]
        np.fill_diagonal(consensus_matrix, 1.0)

        # Calculate consensus score
        # High values (near 0 or 1) are stable, mid values (near 0.5) are unstable
        # Distance from 0.5 (indecision) maps to stability
        stability_values = np.abs(consensus_matrix - 0.5) * 2  # Maps [0,1] to stability
        # Use upper triangle only (matrix is symmetric)
        triu_indices = np.triu_indices_from(stability_values, k=1)
        stability_scores = stability_values[triu_indices]

        mean_consensus = float(np.mean(stability_scores))
        std_consensus = float(np.std(stability_scores))

        # Compute per-run stability scores for CI estimation
        # Each run contributes one scalar: the mean |C_ij - 0.5|*2 over pairs in that run
        per_run_scores = []
        for indices, labels_sub, status in results:
            if status == "valid":
                run_pairs = consensus_matrix[np.ix_(indices, indices)]
                triu = np.triu_indices_from(run_pairs, k=1)
                run_stability = float(np.mean(np.abs(run_pairs[triu] - 0.5) * 2))
                per_run_scores.append(run_stability)

        # Calculate 95% CI from per-run distribution (not per-pair)
        per_run_scores = np.array(per_run_scores)
        n_runs_ci = len(per_run_scores)
        se_consensus = float(np.std(per_run_scores)) / np.sqrt(n_runs_ci)
        ci_95_lower = mean_consensus - 1.96 * se_consensus
        ci_95_upper = mean_consensus + 1.96 * se_consensus

        # Ensure CI is within [0, 1]
        ci_95_lower = max(0.0, float(ci_95_lower))
        ci_95_upper = min(1.0, float(ci_95_upper))

        results = {
            "mean_consensus": mean_consensus,
            "std_consensus": std_consensus,
            "ci_95_lower": ci_95_lower,
            "ci_95_upper": ci_95_upper,
            "consensus_matrix": consensus_matrix,
            "n_runs": self.config.stability.n_runs,
            "n_valid": valid_runs,
            "n_failed": n_failed,
            "n_wrong_clusters": n_wrong_clusters,
        }

        self.logger.info("Consensus Clustering Results:")
        self.logger.info(f"  Valid runs: {valid_runs}/{self.config.stability.n_runs}")
        if n_wrong_clusters > 0:
            self.logger.info(f"  Runs with wrong cluster count: {n_wrong_clusters}")
        if n_failed > 0:
            self.logger.info(f"  Failed runs: {n_failed}")
        self.logger.info(
            f"  Mean consensus score: {mean_consensus:.4f} "
            f"(95% CI: [{ci_95_lower:.4f}, {ci_95_upper:.4f}])"
        )

        return results

    def analyze_cluster_stability(
        self, X: np.ndarray, model, original_labels: np.ndarray, n_clusters: int
    ) -> Dict[int, Dict]:
        """
        Analyze stability for each individual cluster.

        Uses parallel processing for faster execution.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        model
            Fitted model
        original_labels : np.ndarray
            Original cluster assignments
        n_clusters : int
            Number of clusters

        Returns
        -------
        Dict[int, Dict]
            Per-cluster stability metrics
        """
        if not self.config.stability.enabled:
            return {}

        self.logger.info("Per-Cluster Consensus Stability:")

        n_jobs = getattr(self.config.stability, "n_jobs", -1)
        n_samples = len(X)
        subsample_size = int(n_samples * self.config.stability.subsample_fraction)

        # Get min cluster size
        min_cluster_size = self.config.model_selection.get_min_cluster_size(n_samples)

        # Run iterations in parallel (tqdm wraps result generator for completion tracking)
        gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_run_single_cluster_stability_iteration)(
                i,
                X,
                model,
                original_labels,
                n_samples,
                subsample_size,
                n_clusters,
                min_cluster_size,
                self.config.stepmix.random_state,
                self.config.stepmix.n_init,
                self.config.stepmix.max_iter,
                self.config.stepmix.abs_tol,
                self.config.stepmix.rel_tol,
            )
            for i in range(self.config.stability.n_runs)
        )
        iteration_results = list(
            tqdm(
                gen,
                total=self.config.stability.n_runs,
                desc="Per-cluster stability",
                file=sys.stdout,
                leave=False,
                disable=self.config.logging.quiet_mode,
            )
        )

        # Aggregate results
        cluster_consistency = {i: [] for i in range(n_clusters)}
        n_valid_samples = 0

        for result in iteration_results:
            if result is not None:
                n_valid_samples += 1
                for cluster_id, consistency in result.items():
                    cluster_consistency[cluster_id].append(consistency)

        # Compute per-cluster statistics
        results = {}

        if n_valid_samples == 0:
            self.logger.warning("  No valid subsample runs for per-cluster analysis")
            return results

        for cluster_id in range(n_clusters):
            if cluster_consistency[cluster_id]:
                scores = cluster_consistency[cluster_id]
                mean_consistency = np.mean(scores)
                std_consistency = np.std(scores)

                # Calculate 95% CI
                n_scores = len(scores)
                se_consistency = std_consistency / np.sqrt(n_scores)
                ci_95_lower = max(0.0, float(mean_consistency - 1.96 * se_consistency))
                ci_95_upper = min(1.0, float(mean_consistency + 1.96 * se_consistency))

                results[cluster_id] = {
                    "mean_consistency": float(mean_consistency),
                    "std_consistency": float(std_consistency),
                    "ci_95_lower": ci_95_lower,
                    "ci_95_upper": ci_95_upper,
                    "n_samples": int(np.sum(original_labels == cluster_id)),
                    "n_valid_runs": len(scores),
                }

                self.logger.info(
                    f"  Cluster {cluster_id}: "
                    f"{mean_consistency:.4f} (95% CI: [{ci_95_lower:.4f}, {ci_95_upper:.4f}]) "
                    f"(n={results[cluster_id]['n_samples']}, "
                    f"valid_runs={results[cluster_id]['n_valid_runs']})"
                )

        return results
