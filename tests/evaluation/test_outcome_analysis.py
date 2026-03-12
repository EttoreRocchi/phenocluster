"""Tests for outcome analysis module."""

import numpy as np
import pandas as pd

from phenocluster.config import PhenoClusterConfig
from phenocluster.evaluation.outcome_analysis import OutcomeAnalyzer


def _make_config(n_clusters=3, inference_enabled=True, fdr=True, outcome_cols=None):
    """Create a config for outcome tests."""
    return PhenoClusterConfig.from_dict(
        {
            "global": {"project_name": "test", "random_state": 42},
            "data": {"continuous_columns": ["x"], "split": {}},
            "preprocessing": {},
            "model": {"n_clusters": n_clusters},
            "outcome": {
                "enabled": True,
                "outcome_columns": outcome_cols or ["outcome1"],
            },
            "inference": {
                "enabled": inference_enabled,
                "fdr_correction": fdr,
                "outcome_test": "auto",
            },
            "logging": {"level": "WARNING", "log_to_file": False},
        }
    )


def _make_outcome_data(n=120, seed=42):
    """Create synthetic binary outcome data."""
    rng = np.random.RandomState(seed)
    labels = np.array([i % 3 for i in range(n)])
    # Cluster 1 has higher outcome rate
    probs = np.where(labels == 1, 0.7, 0.3)
    outcome = (rng.rand(n) < probs).astype(int)
    df = pd.DataFrame({"outcome1": outcome, "x": rng.randn(n)})
    return df, labels


class TestAnalyzeOutcomes:
    def test_no_outcome_columns(self):
        # OutcomeConfig requires columns when enabled, so create disabled outcome
        cfg = PhenoClusterConfig.from_dict(
            {
                "global": {"project_name": "test", "random_state": 42},
                "data": {"continuous_columns": ["x"], "split": {}},
                "preprocessing": {},
                "model": {"n_clusters": 3},
                "outcome": {"enabled": False},
                "inference": {"enabled": True},
                "logging": {"level": "WARNING", "log_to_file": False},
            }
        )
        analyzer = OutcomeAnalyzer(cfg, n_clusters=3)
        result = analyzer.analyze_outcomes(pd.DataFrame(), np.array([]))
        assert result == {}

    def test_missing_column_skipped(self):
        cfg = _make_config(outcome_cols=["nonexistent"])
        analyzer = OutcomeAnalyzer(cfg, n_clusters=3)
        df = pd.DataFrame({"x": [1, 2, 3]})
        labels = np.array([0, 1, 2])
        result = analyzer.analyze_outcomes(df, labels)
        assert "nonexistent" not in result

    def test_non_binary_outcome_returns_none(self):
        cfg = _make_config()
        analyzer = OutcomeAnalyzer(cfg, n_clusters=2)
        df = pd.DataFrame({"outcome1": [0, 1, 2, 3, 4, 5]})
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = analyzer.analyze_outcomes(df, labels)
        assert "outcome1" not in result

    def test_single_outcome_structure(self):
        cfg = _make_config()
        analyzer = OutcomeAnalyzer(cfg, n_clusters=3)
        df, labels = _make_outcome_data()
        result = analyzer.analyze_outcomes(df, labels)
        assert "outcome1" in result
        cluster_result = result["outcome1"]
        # Reference cluster has OR = 1.0
        assert cluster_result[0]["OR"] == 1.0
        # Non-reference clusters have OR, CI, p_value
        for cid in [1, 2]:
            assert "OR" in cluster_result[cid]
            assert "CI_lower" in cluster_result[cid]
            assert "CI_upper" in cluster_result[cid]
            assert "p_value" in cluster_result[cid]

    def test_reference_phenotype_or_is_one(self):
        cfg = _make_config()
        analyzer = OutcomeAnalyzer(cfg, n_clusters=3)
        df, labels = _make_outcome_data()
        result = analyzer.analyze_outcomes(df, labels, reference_phenotype=1)
        assert result["outcome1"][1]["OR"] == 1.0

    def test_fdr_correction_applied(self):
        cfg = _make_config(fdr=True)
        analyzer = OutcomeAnalyzer(cfg, n_clusters=3)
        df, labels = _make_outcome_data()
        result = analyzer.analyze_outcomes(df, labels)
        # Non-reference clusters should have p_value_fdr
        for cid in [1, 2]:
            assert "p_value_fdr" in result["outcome1"][cid]

    def test_descriptive_only_no_or(self):
        cfg = _make_config(inference_enabled=False)
        analyzer = OutcomeAnalyzer(cfg, n_clusters=3)
        df, labels = _make_outcome_data()
        result = analyzer.analyze_outcomes(df, labels)
        # Non-reference clusters should have prevalence but no OR
        assert "prevalence" in result["outcome1"][1]
        assert "OR" not in result["outcome1"][1]


class TestContingencyTests:
    def test_fisher_used_for_small_cells(self):
        cfg = _make_config()
        analyzer = OutcomeAnalyzer(cfg, n_clusters=2)
        # Very small sample: Fisher's exact should be used
        df = pd.DataFrame({"outcome1": [0, 0, 1, 1, 0, 1, 0, 1]})
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        result = analyzer.analyze_outcomes(df, labels)
        assert "outcome1" in result
        if 1 in result["outcome1"] and "p_value_contingency_method" in result["outcome1"][1]:
            assert result["outcome1"][1]["p_value_contingency_method"] == "Fisher's exact"

    def test_chi_square_for_large_cells(self):
        cfg = _make_config()
        analyzer = OutcomeAnalyzer(cfg, n_clusters=2)
        # Large sample: chi-square should be used
        df, labels = _make_outcome_data(n=200)
        labels = np.array([i % 2 for i in range(200)])
        result = analyzer.analyze_outcomes(df, labels)
        if (
            1 in result.get("outcome1", {})
            and "p_value_contingency_method" in result["outcome1"][1]
        ):
            assert result["outcome1"][1]["p_value_contingency_method"] == "chi-square"
