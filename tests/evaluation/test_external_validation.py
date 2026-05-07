"""Tests for the external validation module."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from phenocluster.config import PhenoClusterConfig
from phenocluster.evaluation.external_validation import ExternalValidator
from phenocluster.utils.report import _generate_external_validation_section


class TestExternalValidator:
    """Tests for ExternalValidator class."""

    @pytest.fixture
    def config(self, tmp_path):
        return PhenoClusterConfig(
            continuous_columns=["a", "b"],
            output_dir=str(tmp_path),
            outcome=MagicMock(outcome_columns=["mortality"]),
        )

    @pytest.fixture
    def mock_model(self):
        """Mock StepMix model."""
        model = MagicMock()
        model.predict.return_value = np.array([0, 1, 0, 1, 2])
        model.predict_proba.return_value = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.7, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.1, 0.8],
            ]
        )
        model.score.return_value = -1.5
        return model

    def test_validate_with_model(self, config, mock_model):
        validator = ExternalValidator(config, n_clusters=3)
        X_external = np.random.randn(5, 2)

        results = validator.validate_with_model(
            X_external=X_external,
            model=mock_model,
            n_external=5,
        )

        assert "external_labels" in results
        assert "cluster_distribution" in results
        assert "log_likelihood" in results
        assert results["n_samples"] == 5
        assert results["log_likelihood"] == -1.5


class TestCompareOutcomesWithExternalDf:
    """Tests for the outcome-comparison branch with external DataFrame."""

    @pytest.fixture
    def config(self, tmp_path):
        from phenocluster.config import OutcomeConfig

        return PhenoClusterConfig(
            continuous_columns=["a", "b"],
            output_dir=str(tmp_path),
            outcome=OutcomeConfig(outcome_columns=["mortality"]),
        )

    def test_external_prevalence_and_chi2_present(self, config):
        from unittest.mock import MagicMock as MM

        validator = ExternalValidator(config, n_clusters=2)
        model = MM()
        model.predict.return_value = np.array([0, 0, 0, 1, 1, 1])
        model.score.return_value = -1.0
        derivation_outcomes = {
            "full_cohort": {
                "mortality": {
                    0: {"prevalence": 0.3, "n_positive": 6, "n_total": 20},
                    1: {"prevalence": 0.7, "n_positive": 14, "n_total": 20},
                }
            }
        }
        external_df = pd.DataFrame({"mortality": [0, 1, 0, 1, 1, 0]})
        results = validator.validate_with_model(
            X_external=np.zeros((6, 2)),
            model=model,
            derivation_outcomes=derivation_outcomes,
            n_external=6,
            external_df=external_df,
        )
        comp = results["outcome_comparison"]["mortality"]
        assert "external_prevalence" in comp
        assert "external_outcome_counts" in comp
        assert "chi2_statistic" in comp
        assert "chi2_p_value" in comp

    def test_outcome_comparison_without_external_df(self, config):
        from unittest.mock import MagicMock as MM

        validator = ExternalValidator(config, n_clusters=2)
        model = MM()
        model.predict.return_value = np.array([0, 1, 1, 0])
        model.score.return_value = -2.0
        derivation_outcomes = {
            "full_cohort": {
                "mortality": {
                    0: {"prevalence": 0.5, "n_positive": 5, "n_total": 10},
                }
            }
        }
        results = validator.validate_with_model(
            X_external=np.zeros((4, 2)),
            model=model,
            derivation_outcomes=derivation_outcomes,
            n_external=4,
        )
        comp = results["outcome_comparison"]["mortality"]
        assert "external_prevalence" not in comp
        assert "derivation_prevalence" in comp


class TestChi2CohortComparison:
    """Direct tests for the static chi-square helper."""

    def test_returns_dict_for_valid_counts(self):
        deriv = {0: {"n_positive": 10, "n_total": 50}, 1: {"n_positive": 30, "n_total": 50}}
        ext = {0: {"n_positive": 5, "n_total": 25}, 1: {"n_positive": 20, "n_total": 25}}
        out = ExternalValidator._chi2_cohort_comparison(deriv, ext)
        assert out is not None
        assert "statistic" in out and "p_value" in out

    def test_no_overlap_returns_none(self):
        out = ExternalValidator._chi2_cohort_comparison(
            {0: {"n_positive": 5, "n_total": 10}},
            {1: {"n_positive": 5, "n_total": 10}},
        )
        assert out is None

    def test_zero_totals_returns_none(self):
        out = ExternalValidator._chi2_cohort_comparison(
            {0: {"n_positive": 0, "n_total": 0}},
            {0: {"n_positive": 0, "n_total": 0}},
        )
        assert out is None


class TestSimplifiedAssignmentModelRemoved:
    """Verify SimplifiedAssignmentModel and related code are removed."""

    def test_no_simplified_assignment_model(self):
        with pytest.raises(ImportError):
            from phenocluster.evaluation.external_validation import (
                SimplifiedAssignmentModel,  # noqa: F401
            )

    def test_no_select_top_features(self):
        with pytest.raises(ImportError):
            from phenocluster.evaluation.external_validation import (
                select_top_features,  # noqa: F401
            )

    def test_no_create_simplified_model(self):
        with pytest.raises(ImportError):
            from phenocluster.evaluation.external_validation import (
                create_simplified_model,  # noqa: F401
            )


class TestExternalSurvivalIntegration:
    """Tests for external survival/multistate result structure."""

    def test_external_survival_results_structure(self):
        """Verify survival_results dict has expected keys when present."""
        results = {
            "external_labels": [0, 1, 0],
            "cluster_distribution": {
                0: {"count": 2, "percentage": 66.7},
                1: {"count": 1, "percentage": 33.3},
            },
            "n_samples": 3,
            "log_likelihood": -1.5,
            "survival_results": {
                "mortality": {
                    "survival_data": {
                        0: {"n_patients": 2, "n_events": 1},
                        1: {"n_patients": 1, "n_events": 0},
                    },
                    "median_survival": {0: 15.0, 1: float("inf")},
                }
            },
        }
        assert "survival_results" in results
        assert "mortality" in results["survival_results"]
        mort = results["survival_results"]["mortality"]
        assert mort["survival_data"][0]["n_patients"] == 2
        assert mort["median_survival"][1] == float("inf")

    def test_external_multistate_results_structure(self):
        """Verify multistate_results dict has expected keys when present."""
        results = {
            "external_labels": [0, 1, 0],
            "n_samples": 3,
            "multistate_results": {
                "transition_results": {
                    "alive_to_dead": {
                        "n_events": 5,
                        "n_at_risk": 10,
                        "phenotype_effects": {
                            0: {"HR": 1.0},
                            1: {"HR": 1.5, "CI_lower": 0.8, "CI_upper": 2.8, "p_value": 0.15},
                        },
                    }
                },
                "pathway_results": [{"state_names": ["alive", "dead"], "total_count": 5}],
            },
        }
        assert "multistate_results" in results
        trans = results["multistate_results"]["transition_results"]
        assert "alive_to_dead" in trans
        assert trans["alive_to_dead"]["n_events"] == 5

    def test_external_results_without_survival(self):
        """External results dict works fine without survival or multistate keys."""
        results = {
            "external_labels": [0, 1],
            "cluster_distribution": {0: {"count": 1, "percentage": 50.0}},
            "n_samples": 2,
            "log_likelihood": -2.0,
        }
        assert "survival_results" not in results
        assert "multistate_results" not in results
        assert results["n_samples"] == 2


class TestExternalValidationReport:
    """Tests for external validation report section rendering."""

    def test_report_empty_when_no_results(self, tmp_path):
        """Section returns empty string when no external validation results."""
        html = _generate_external_validation_section({}, tmp_path)
        assert html == ""

    def test_report_renders_distribution_table(self, tmp_path):
        """Section renders cluster distribution comparison table."""
        data = {
            "external_validation_results": {
                "n_samples": 100,
                "log_likelihood": -2.5,
                "cluster_distribution": {
                    0: {"count": 60, "percentage": 60.0},
                    1: {"count": 40, "percentage": 40.0},
                },
                "derivation_distribution": {
                    0: {"count": 50, "percentage": 50.0},
                    1: {"count": 50, "percentage": 50.0},
                },
            }
        }
        html = _generate_external_validation_section(data, tmp_path)
        assert "External Validation" in html
        assert "Cluster Distribution Comparison" in html
        assert "Phenotype 0" in html
        assert "60.0%" in html

    def test_report_renders_survival_section(self, tmp_path):
        """Section renders survival tables when survival results present."""
        data = {
            "external_validation_results": {
                "n_samples": 50,
                "log_likelihood": -1.8,
                "cluster_distribution": {
                    0: {"count": 30, "percentage": 60.0},
                    1: {"count": 20, "percentage": 40.0},
                },
                "survival_results": {
                    "mortality": {
                        "survival_data": {
                            0: {"n_patients": 30, "n_events": 10},
                            1: {"n_patients": 20, "n_events": 8},
                        },
                        "median_survival": {0: 24.0, 1: 12.0},
                        "comparison": {
                            "1_vs_0": {
                                "HR": 1.8,
                                "CI_lower": 0.9,
                                "CI_upper": 3.5,
                                "p_value": 0.05,
                            }
                        },
                        "logrank_p_value": 0.03,
                    }
                },
            }
        }
        html = _generate_external_validation_section(data, tmp_path)
        assert "Survival Analysis (External Cohort)" in html
        assert "Mortality" in html
        assert "24.0" in html
        assert "log-rank" in html

    def test_report_renders_multistate_section(self, tmp_path):
        """Section renders multistate tables when multistate results present."""
        data = {
            "reference_phenotype": 0,
            "external_validation_results": {
                "n_samples": 50,
                "log_likelihood": -1.8,
                "cluster_distribution": {
                    0: {"count": 30, "percentage": 60.0},
                },
                "multistate_results": {
                    "transition_results": {
                        "alive_to_dead": {
                            "n_events": 15,
                            "n_at_risk": 50,
                            "phenotype_effects": {
                                0: {"HR": 1.0},
                                1: {
                                    "HR": 2.1,
                                    "CI_lower": 1.0,
                                    "CI_upper": 4.2,
                                    "p_value": 0.02,
                                },
                            },
                        }
                    },
                    "pathway_results": [{"state_names": ["alive", "dead"], "total_count": 15}],
                },
            },
        }
        html = _generate_external_validation_section(data, tmp_path)
        assert "Multistate Analysis (External Cohort)" in html
        assert "Alive To Dead" in html
        assert "15/50" in html
